from __future__ import annotations

import base64
import json
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import tifffile as tf
from skimage.segmentation import find_boundaries

from .types import TrackingParams


@dataclass(frozen=True)
class AmbiguousEvent:
    event_id: int
    frame: int
    event_type: str
    original_decision: str
    parent_candidate: int | None
    new_particles: list[int]
    x: float
    y: float
    min_dist_px: float | None
    n_neighbors: int
    parent_track_len: int = 0
    child_track_lens: list[int] | None = None
    note: str = ""


def run_phase1_adjudication(
    movie_path: str,
    masks_filt: np.ndarray,
    tracks: pd.DataFrame,
    tr: TrackingParams,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not tr.adjudication_enabled or tracks.empty:
        return tracks, pd.DataFrame()

    events = _detect_ambiguous_events(tracks, tr)
    if not events:
        return tracks, pd.DataFrame()
    events = events[: int(tr.adjudication_max_events)]

    movie = tf.memmap(str(movie_path))
    out = tracks.copy()
    audit_rows: list[dict[str, Any]] = []

    for e in events:
        crops = _make_context_crops(movie, masks_filt, out, e, tr)
        decision = _adjudicate_event(e, crops, tr)
        before = out
        out, final_action, applied = _apply_decision(out, e, decision, tr)
        if applied:
            out = _recompute_generation_and_fate(out)
        audit_rows.append(
            {
                "event_id": e.event_id,
                "frame": e.frame,
                "event_type": e.event_type,
                "original_decision": e.original_decision,
                "parent_candidate": e.parent_candidate,
                "new_particles": ",".join(str(x) for x in e.new_particles),
                "min_dist_px": e.min_dist_px,
                "n_neighbors": e.n_neighbors,
                "provider": decision.get("provider"),
                "vlm_decision": decision.get("decision"),
                "selected_hypothesis": decision.get("hypothesis"),
                "confidence": float(decision.get("confidence", 0.0)),
                "reason": decision.get("reason", ""),
                "final_applied_action": final_action,
                "applied": bool(applied),
                "rows_before": int(len(before)),
                "rows_after": int(len(out)),
            }
        )

    out = out.sort_values(["particle", "frame"]).drop_duplicates(["particle", "frame"], keep="first").copy()
    return out, pd.DataFrame(audit_rows)


def _detect_ambiguous_events(tracks: pd.DataFrame, tr: TrackingParams) -> list[AmbiguousEvent]:
    if tracks.empty:
        return []
    radius = float(tr.adjudication_radius_px)
    gap = int(tr.adjudication_gap_frames)

    starts = (
        tracks.sort_values("frame")
        .groupby("particle", as_index=False)
        .first()[["particle", "frame", "x", "y"]]
        .rename(columns={"frame": "start_frame", "x": "start_x", "y": "start_y"})
    )
    ends = (
        tracks.sort_values("frame")
        .groupby("particle", as_index=False)
        .last()[["particle", "frame", "x", "y"]]
        .rename(columns={"frame": "end_frame", "x": "end_x", "y": "end_y"})
    )
    spans = tracks.groupby("particle", as_index=False)["frame"].agg(start_frame="min", end_frame="max")
    spans["track_len"] = spans["end_frame"] - spans["start_frame"] + 1
    starts = starts.merge(spans[["particle", "track_len"]], on="particle", how="left")
    starts["track_len"] = starts["track_len"].fillna(1).astype(int)
    ends = ends.merge(spans[["particle", "track_len"]], on="particle", how="left")
    ends["track_len"] = ends["track_len"].fillna(1).astype(int)
    starts = starts[starts["start_frame"] > 0].copy()

    min_persist = _adaptive_min_persistence(spans["track_len"].to_numpy(dtype=int), tr)
    div_min_persist = max(int(tr.adjudication_division_persistence_frames), min_persist)
    min_parent_len = int(tr.adjudication_division_min_parent_track_len)

    by_frame = {int(f): g for f, g in tracks.groupby("frame")}
    handled_births: set[int] = set()
    events: list[AmbiguousEvent] = []
    eidx = 1

    # Pattern A/B: disappearance followed by 1-2 nearby births (potential continue/division)
    for _, d in ends.iterrows():
        p = int(d["particle"])
        te = int(d["end_frame"])
        dx, dy = float(d["end_x"]), float(d["end_y"])
        parent_track_len = int(d.get("track_len", 0))
        cand = starts[(starts["start_frame"] >= te + 1) & (starts["start_frame"] <= te + gap)].copy()
        if cand.empty:
            continue
        cand["dist"] = np.sqrt((cand["start_x"] - dx) ** 2 + (cand["start_y"] - dy) ** 2)
        # Discard ultra-short candidates as likely noise.
        near = cand[(cand["dist"] <= radius) & (cand["track_len"] >= min_persist)].sort_values("dist")
        if near.empty:
            continue

        # Division requires two persistent daughters with close start times.
        near_div = near[near["track_len"] >= div_min_persist].copy()
        if (
            len(near_div) >= 2
            and parent_track_len >= min_parent_len
            and abs(int(near_div.iloc[0]["start_frame"]) - int(near_div.iloc[1]["start_frame"])) <= 1
        ):
            new_ids = [int(x) for x in near_div["particle"].head(2).tolist()]
            child_lens = [int(x) for x in near_div["track_len"].head(2).tolist()]
            evt_type = "potential_division"
            original = "true_division"
            frame = int(min(near_div.iloc[0]["start_frame"], near_div.iloc[1]["start_frame"]))
            x_evt = float(np.mean(near_div["start_x"].head(2).to_numpy(dtype=float)))
            y_evt = float(np.mean(near_div["start_y"].head(2).to_numpy(dtype=float)))
            min_dist = float(near_div.iloc[0]["dist"])
        else:
            # Default to continuity correction with the strongest nearby candidate.
            new_ids = [int(near.iloc[0]["particle"])]
            child_lens = [int(near.iloc[0]["track_len"])]
            evt_type = "disappearance_followed_by_birth"
            original = "continue_same_track"
            frame = int(near.iloc[0]["start_frame"])
            x_evt = float(near.iloc[0]["start_x"])
            y_evt = float(near.iloc[0]["start_y"])
            min_dist = float(near.iloc[0]["dist"])

        for nid in new_ids:
            handled_births.add(nid)
        n_neighbors = _neighbor_count(by_frame.get(frame), x_evt, y_evt, radius)
        events.append(
            AmbiguousEvent(
                event_id=eidx,
                frame=frame,
                event_type=evt_type,
                original_decision=original,
                parent_candidate=p,
                new_particles=new_ids,
                x=x_evt,
                y=y_evt,
                min_dist_px=min_dist,
                n_neighbors=n_neighbors,
                parent_track_len=parent_track_len,
                child_track_lens=child_lens,
                note=f"death@{te}",
            )
        )
        eidx += 1

    # Pattern C: birth near endpoint in crowded neighborhood (possible false new / ID switch)
    for _, b in starts.iterrows():
        bid = int(b["particle"])
        if bid in handled_births:
            continue
        if int(b.get("track_len", 0)) < min_persist:
            continue
        tb = int(b["start_frame"])
        bx, by = float(b["start_x"]), float(b["start_y"])
        cand = ends[(ends["end_frame"] >= tb - gap) & (ends["end_frame"] <= tb - 1)].copy()
        if cand.empty:
            continue
        cand["dist"] = np.sqrt((cand["end_x"] - bx) ** 2 + (cand["end_y"] - by) ** 2)
        near = cand[cand["dist"] <= radius].sort_values("dist")
        if near.empty:
            continue
        n_neighbors = _neighbor_count(by_frame.get(tb), bx, by, radius)
        evt_type = "cluster_overlap_id_change" if n_neighbors >= 3 else "new_near_endpoint"
        events.append(
            AmbiguousEvent(
                event_id=eidx,
                frame=tb,
                event_type=evt_type,
                original_decision="continue_same_track",
                parent_candidate=int(near.iloc[0]["particle"]),
                new_particles=[bid],
                x=bx,
                y=by,
                min_dist_px=float(near.iloc[0]["dist"]),
                n_neighbors=n_neighbors,
                parent_track_len=int(near.iloc[0].get("track_len", 0)),
                child_track_lens=[int(b.get("track_len", 0))],
                note="birth-near-endpoint",
            )
        )
        eidx += 1
    return events


def _adaptive_min_persistence(track_lens: np.ndarray, tr: TrackingParams) -> int:
    if track_lens.size == 0:
        return max(2, int(tr.adjudication_auto_min_persist_frames))
    p10 = int(np.percentile(track_lens, 10))
    floor = int(tr.adjudication_auto_min_persist_frames)
    ceil = int(tr.adjudication_auto_max_min_persist_frames)
    return max(floor, min(ceil, max(2, p10)))


def _neighbor_count(df: pd.DataFrame | None, x: float, y: float, radius: float) -> int:
    if df is None or df.empty:
        return 0
    d = np.sqrt((df["x"] - x) ** 2 + (df["y"] - y) ** 2)
    return int((d <= radius).sum())


def _make_context_crops(
    movie: np.ndarray,
    masks_filt: np.ndarray,
    tracks: pd.DataFrame,
    e: AmbiguousEvent,
    tr: TrackingParams,
) -> list[str]:
    half = int(tr.adjudication_context_half_window)
    sz = int(tr.adjudication_crop_size_px)
    T, H, W = int(movie.shape[0]), int(movie.shape[1]), int(movie.shape[2])
    t0 = max(0, e.frame - half)
    t1 = min(T - 1, e.frame + half)
    cx, cy = int(round(e.x)), int(round(e.y))
    x0 = max(0, cx - sz // 2)
    y0 = max(0, cy - sz // 2)
    x1 = min(W, x0 + sz)
    y1 = min(H, y0 + sz)
    x0 = max(0, x1 - sz)
    y0 = max(0, y1 - sz)

    import cv2

    frames_b64: list[str] = []
    for t in range(t0, t1 + 1):
        g = movie[t][y0:y1, x0:x1].astype(np.float32)
        lo, hi = np.percentile(g, (1, 99))
        if hi <= lo:
            hi = lo + 1.0
        g8 = np.clip((g - lo) / (hi - lo), 0, 1)
        rgb = np.stack([(g8 * 255).astype(np.uint8)] * 3, axis=-1)
        m = masks_filt[t][y0:y1, x0:x1]
        b = find_boundaries(m > 0, mode="outer")
        rgb[b, :] = (0, 255, 0)
        curr = tracks[tracks["frame"] == t]
        for _, r in curr.iterrows():
            xx = int(round(float(r["x"]))) - x0
            yy = int(round(float(r["y"]))) - y0
            if 0 <= xx < rgb.shape[1] and 0 <= yy < rgb.shape[0]:
                cv2.circle(rgb, (xx, yy), 2, (255, 255, 0), -1, lineType=cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if ok:
            frames_b64.append(base64.b64encode(buf.tobytes()).decode("ascii"))
    return frames_b64


def _adjudicate_event(e: AmbiguousEvent, crops_b64: list[str], tr: TrackingParams) -> dict[str, Any]:
    provider = (tr.adjudication_provider or "heuristic").lower()
    if provider == "gemini":
        out = _adjudicate_gemini(e, crops_b64, tr)
        if out is not None:
            return out
        # Defensive fallback if helper returns None unexpectedly.
        return {"provider": "gemini", "decision": "defer", "confidence": 0.0, "reason": "gemini_unavailable_unknown"}
    return _adjudicate_heuristic(e)


def _adjudicate_heuristic(e: AmbiguousEvent) -> dict[str, Any]:
    child_lens = e.child_track_lens or []
    if e.event_type == "potential_division" and len(e.new_particles) >= 2 and len(child_lens) >= 2 and min(child_lens[:2]) >= 2:
        return {"provider": "heuristic", "decision": "true_division", "confidence": 0.72, "reason": "2 persistent nearby births after disappearance"}
    if e.event_type == "cluster_overlap_id_change":
        return {"provider": "heuristic", "decision": "merge_or_touch_no_new_id", "confidence": 0.66, "reason": "crowded overlap near endpoint"}
    if e.min_dist_px is not None and e.min_dist_px <= 20:
        return {"provider": "heuristic", "decision": "continue_same_track", "confidence": 0.70, "reason": "birth very close to endpoint"}
    return {"provider": "heuristic", "decision": "true_new_cell", "confidence": 0.56, "reason": "insufficient continuation evidence"}


def _adjudicate_gemini(e: AmbiguousEvent, crops_b64: list[str], tr: TrackingParams) -> dict[str, Any] | None:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return {"provider": "gemini", "decision": "defer", "confidence": 0.0, "reason": "gemini_unavailable_missing_api_key"}
    model = tr.adjudication_model or "gemini-2.5-flash"
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    hypotheses = _hypotheses_for_event(e)
    schema_prompt = (
        "You are adjudicating ambiguous microscopy cell-tracking events from a short temporal clip. "
        "Return strict JSON only with fields: hypothesis, decision, confidence, reason. "
        f"Choose exactly one hypothesis from this list: {hypotheses}. "
        "decision must be one of: continue_same_track, true_new_cell, true_division, merge_or_touch_no_new_id. "
        "confidence must be a float in [0,1], where >=0.80 means strong visual evidence and <=0.50 means uncertain. "
        f"event_type={e.event_type}; frame={e.frame}; parent_candidate={e.parent_candidate}; "
        f"new_particles={e.new_particles}; min_dist_px={e.min_dist_px}; n_neighbors={e.n_neighbors}; "
        f"parent_track_len={e.parent_track_len}; child_track_lens={e.child_track_lens}. "
        "Use temporal continuity across the sequence: if morphology and centroid continuity suggest the same cell, "
        "favor continue_same_track or merge_or_touch_no_new_id. "
        "For crowded contact/bump events, compare the hypotheses using before-contact, during-contact, and after-contact frames. "
        "Prefer contact_same_identity or fragment_reconnect over true_division when cells merely touch, bump, change direction, "
        "or temporarily merge at the boundary. "
        "Use true_division only when a plausible parent-to-two-daughters transition is visible across frames, "
        "both daughters persist, and timing is consistent. "
        "Do NOT call true_division when the parent remains large/unchanged with only contact-direction changes in crowded regions. "
        "Keep reason concise."
    )
    parts: list[dict[str, Any]] = [{"text": schema_prompt}]
    for b64 in crops_b64[:9]:
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64}})

    payload = {
        "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"},
        "contents": [{"role": "user", "parts": parts}],
    }
    retries = max(0, int(getattr(tr, "adjudication_gemini_retries", 2)))
    timeout_s = max(5, int(getattr(tr, "adjudication_gemini_timeout_s", 30)))
    last_err = "unknown"
    resp: dict[str, Any] | None = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as r:
                resp = json.loads(r.read().decode("utf-8"))
            break
        except Exception as ex:
            last_err = type(ex).__name__
            if attempt < retries:
                time.sleep(0.6 * (attempt + 1))
    if resp is None:
        return {
            "provider": "gemini",
            "decision": "defer",
            "confidence": 0.0,
            "reason": f"gemini_unavailable_request_failed_{last_err}",
        }

    text = ""
    try:
        text = resp["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return {
            "provider": "gemini",
            "decision": "defer",
            "confidence": 0.0,
            "reason": "gemini_unavailable_malformed_response",
        }
    try:
        obj = json.loads(text)
    except Exception:
        return {
            "provider": "gemini",
            "decision": "defer",
            "confidence": 0.0,
            "reason": "gemini_unavailable_non_json_response",
        }
    h = str(obj.get("hypothesis", "")).strip()
    if h not in hypotheses:
        h = "unspecified"
    d = str(obj.get("decision", "defer"))
    d = _decision_from_hypothesis(h, d)
    if d not in {"continue_same_track", "true_new_cell", "true_division", "merge_or_touch_no_new_id"}:
        d = "defer"
    return {
        "provider": "gemini",
        "hypothesis": h,
        "decision": d,
        "confidence": float(obj.get("confidence", 0.0)),
        "reason": str(obj.get("reason", ""))[:500],
    }


def _hypotheses_for_event(e: AmbiguousEvent) -> list[str]:
    if e.event_type == "potential_division":
        base = [
            "true_division_after_separation",
            "contact_same_identity_no_division",
            "contact_id_swap_after_bump",
            "fragment_reconnect_same_cell",
            "defer_uncertain",
        ]
    elif e.event_type in {"cluster_overlap_id_change", "new_near_endpoint"} or int(e.n_neighbors) >= 3:
        base = [
            "contact_same_identity_no_division",
            "contact_id_swap_after_bump",
            "fragment_reconnect_same_cell",
            "true_new_cell",
            "defer_uncertain",
        ]
    else:
        base = [
            "fragment_reconnect_same_cell",
            "true_new_cell",
            "true_division_after_separation",
            "defer_uncertain",
        ]
    return base


def _decision_from_hypothesis(hypothesis: str, model_decision: str) -> str:
    if hypothesis == "true_division_after_separation":
        return "true_division"
    if hypothesis == "fragment_reconnect_same_cell":
        return "continue_same_track"
    if hypothesis == "contact_same_identity_no_division":
        return "merge_or_touch_no_new_id"
    if hypothesis == "true_new_cell":
        return "true_new_cell"
    if hypothesis in {"contact_id_swap_after_bump", "defer_uncertain"}:
        return "defer"
    return model_decision


def _apply_decision(
    tracks: pd.DataFrame,
    e: AmbiguousEvent,
    decision: dict[str, Any],
    tr: TrackingParams,
) -> tuple[pd.DataFrame, str, bool]:
    conf = float(decision.get("confidence", 0.0))
    d = str(decision.get("decision", "defer"))
    min_conf = float(tr.adjudication_confidence_min)
    if conf < min_conf:
        return tracks, f"rejected_low_confidence_{conf:.2f}_lt_{min_conf:.2f}", False

    out = tracks.copy()
    stats = _track_stats_map(out)
    radius_limit = float(tr.adjudication_radius_px) * float(tr.adjudication_consistency_radius_multiplier)
    if d in {"continue_same_track", "merge_or_touch_no_new_id"}:
        if e.parent_candidate is None or not e.new_particles:
            return tracks, "invalid_continue_payload", False
        child = int(e.new_particles[0])
        parent = int(e.parent_candidate)
        if child == parent:
            return tracks, "invalid_continue_payload_same_id", False
        idx = out["particle"] == child
        if not idx.any():
            return tracks, "child_not_found", False
        pstat = stats.get(parent)
        cstat = stats.get(child)
        if pstat is None or cstat is None:
            return tracks, "invalid_continue_payload_missing_stats", False
        gap = int(cstat["start_frame"]) - int(pstat["end_frame"])
        if gap < 0 or gap > int(tr.adjudication_gap_frames) + 1:
            return tracks, f"rejected_continue_gap_{gap}", False
        dist = float(np.hypot(float(cstat["start_x"]) - float(pstat["end_x"]), float(cstat["start_y"]) - float(pstat["end_y"])))
        if dist > radius_limit:
            return tracks, f"rejected_continue_distance_{dist:.1f}_gt_{radius_limit:.1f}", False
        out.loc[idx, "particle"] = parent
        # Continuation should not keep parent link for merged segment rows.
        if "parent" in out.columns:
            out.loc[idx, "parent"] = np.nan
        out.loc[out["parent"] == child, "parent"] = parent
        return out, f"reassigned_track_{child}_to_{parent}", True

    if d == "true_division":
        if e.parent_candidate is None or len(e.new_particles) < 2:
            return tracks, "invalid_division_payload", False
        parent = int(e.parent_candidate)
        kids = [int(k) for k in e.new_particles[:2]]
        if kids[0] == kids[1]:
            return tracks, "invalid_division_payload_duplicate_children", False
        if parent in kids:
            return tracks, "invalid_division_payload_parent_equals_child", False
        particles_present = set(int(x) for x in out["particle"].dropna().unique().tolist())
        missing_kids = [k for k in kids if k not in particles_present]
        if missing_kids:
            return tracks, f"invalid_division_payload_missing_children_{'_'.join(str(x) for x in missing_kids)}", False
        pstat = stats.get(parent)
        if pstat is None:
            return tracks, "invalid_division_payload_missing_parent_stats", False
        if int(pstat["track_len"]) < int(tr.adjudication_division_min_parent_track_len):
            return tracks, "invalid_division_payload_short_parent_track", False
        # Require both children to start at/after event frame to avoid retroactive links.
        starts = out.groupby("particle", as_index=False)["frame"].min().rename(columns={"frame": "start_frame"})
        starts_map = {int(r["particle"]): int(r["start_frame"]) for _, r in starts.iterrows()}
        if any(starts_map.get(k, -10**9) < int(e.frame) for k in kids):
            return tracks, "invalid_division_payload_children_start_before_event", False
        for k in kids:
            kstat = stats.get(k)
            if kstat is None:
                return tracks, f"invalid_division_payload_missing_child_stats_{k}", False
            if int(kstat["track_len"]) < int(tr.adjudication_division_persistence_frames):
                return tracks, f"invalid_division_payload_short_child_track_{k}", False
            dpk = float(np.hypot(float(kstat["start_x"]) - float(pstat["end_x"]), float(kstat["start_y"]) - float(pstat["end_y"])))
            if dpk > radius_limit:
                return tracks, f"invalid_division_payload_child_far_{k}_{dpk:.1f}", False
        if abs(int(starts_map.get(kids[0], 0)) - int(starts_map.get(kids[1], 0))) > 1:
            return tracks, "invalid_division_payload_children_desync", False
        # Crowded scenes with moderate confidence often induce false-positive divisions.
        if int(e.n_neighbors) >= 4 and conf < 0.90:
            return tracks, "rejected_division_crowded_low_confidence", False
        # Biological guard: when parent stays large and daughters are not meaningfully smaller,
        # avoid forcing a split label.
        if "area" in out.columns:
            p_area = float(pstat.get("end_area", np.nan))
            c1_area = float(stats.get(kids[0], {}).get("start_area", np.nan))
            c2_area = float(stats.get(kids[1], {}).get("start_area", np.nan))
            if np.isfinite(p_area) and np.isfinite(c1_area) and np.isfinite(c2_area) and p_area > 0:
                if c1_area >= 0.90 * p_area and c2_area >= 0.90 * p_area:
                    return tracks, "rejected_division_children_not_smaller_than_parent", False
                if (c1_area + c2_area) > 1.80 * p_area:
                    return tracks, "rejected_division_children_area_sum_too_large", False
        if "parent" in out.columns:
            out.loc[out["particle"].isin(kids), "parent"] = parent
        if "fate" in out.columns:
            out.loc[out["particle"] == parent, "fate"] = "DIVIDE"
        return out, f"set_parent_{parent}_for_{kids[0]}_{kids[1]}", True

    if d == "true_new_cell":
        return tracks, "kept_true_new_cell", False

    if d == "defer":
        return tracks, "rejected_defer", False
    return tracks, f"rejected_unknown_decision_{d}", False


def _track_stats_map(tracks: pd.DataFrame) -> dict[int, dict[str, float | int]]:
    if tracks.empty:
        return {}
    first = tracks.sort_values("frame").groupby("particle", as_index=False).first()[["particle", "frame", "x", "y"]]
    first = first.rename(columns={"frame": "start_frame", "x": "start_x", "y": "start_y"})
    last = tracks.sort_values("frame").groupby("particle", as_index=False).last()[["particle", "frame", "x", "y"]]
    last = last.rename(columns={"frame": "end_frame", "x": "end_x", "y": "end_y"})
    spans = tracks.groupby("particle", as_index=False)["frame"].agg(track_len="count")
    if "area" in tracks.columns:
        first_a = tracks.sort_values("frame").groupby("particle", as_index=False).first()[["particle", "area"]]
        first_a = first_a.rename(columns={"area": "start_area"})
        last_a = tracks.sort_values("frame").groupby("particle", as_index=False).last()[["particle", "area"]]
        last_a = last_a.rename(columns={"area": "end_area"})
        merged = (
            first.merge(last, on="particle", how="inner")
            .merge(spans, on="particle", how="left")
            .merge(first_a, on="particle", how="left")
            .merge(last_a, on="particle", how="left")
        )
    else:
        merged = first.merge(last, on="particle", how="inner").merge(spans, on="particle", how="left")
    out: dict[int, dict[str, float | int]] = {}
    for _, r in merged.iterrows():
        pid = int(r["particle"])
        out[pid] = {
            "start_frame": int(r["start_frame"]),
            "end_frame": int(r["end_frame"]),
            "start_x": float(r["start_x"]),
            "start_y": float(r["start_y"]),
            "end_x": float(r["end_x"]),
            "end_y": float(r["end_y"]),
            "track_len": int(r["track_len"]),
            "start_area": float(r["start_area"]) if "start_area" in r and pd.notna(r["start_area"]) else float("nan"),
            "end_area": float(r["end_area"]) if "end_area" in r and pd.notna(r["end_area"]) else float("nan"),
        }
    return out


def _recompute_generation_and_fate(tracks: pd.DataFrame) -> pd.DataFrame:
    if tracks.empty:
        return tracks
    out = tracks.copy()
    if "parent" not in out.columns:
        out["parent"] = np.nan
    pmap = (
        out.groupby("particle")["parent"]
        .first()
        .to_dict()
    )
    # normalize parent map to ints / None
    clean: dict[int, int | None] = {}
    for k, v in pmap.items():
        kk = int(k)
        if pd.isna(v):
            clean[kk] = None
        else:
            clean[kk] = int(v)

    children_count: dict[int, int] = {}
    for pid, par in clean.items():
        if par is not None:
            children_count[par] = children_count.get(par, 0) + 1

    gen_cache: dict[int, int] = {}

    def _gen(pid: int, seen: set[int] | None = None) -> int:
        if pid in gen_cache:
            return gen_cache[pid]
        if seen is None:
            seen = set()
        if pid in seen:
            gen_cache[pid] = 0
            return 0
        seen.add(pid)
        p = clean.get(pid)
        if p is None:
            g = 0
        else:
            g = _gen(int(p), seen) + 1
        gen_cache[pid] = g
        return g

    for pid in clean:
        _gen(pid)

    out["generation"] = out["particle"].map(lambda p: gen_cache.get(int(p), 0))
    out["fate"] = out["particle"].map(lambda p: "DIVIDE" if children_count.get(int(p), 0) >= 2 else "UNDEFINED")
    return out
