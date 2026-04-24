from __future__ import annotations

import base64
import json
import os
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
    starts = starts[starts["start_frame"] > 0].copy()

    by_frame = {int(f): g for f, g in tracks.groupby("frame")}
    handled_births: set[int] = set()
    events: list[AmbiguousEvent] = []
    eidx = 1

    # Pattern A/B: disappearance followed by 1-2 nearby births (potential continue/division)
    for _, d in ends.iterrows():
        p = int(d["particle"])
        te = int(d["end_frame"])
        dx, dy = float(d["end_x"]), float(d["end_y"])
        cand = starts[(starts["start_frame"] >= te + 1) & (starts["start_frame"] <= te + gap)].copy()
        if cand.empty:
            continue
        cand["dist"] = np.sqrt((cand["start_x"] - dx) ** 2 + (cand["start_y"] - dy) ** 2)
        near = cand[cand["dist"] <= radius].sort_values("dist")
        if near.empty:
            continue
        new_ids = [int(x) for x in near["particle"].head(2).tolist()]
        for nid in new_ids:
            handled_births.add(nid)
        evt_type = "potential_division" if len(new_ids) >= 2 else "disappearance_followed_by_birth"
        original = "true_division" if len(new_ids) >= 2 else "continue_same_track"
        frame = int(near.iloc[0]["start_frame"])
        n_neighbors = _neighbor_count(by_frame.get(frame), float(near.iloc[0]["start_x"]), float(near.iloc[0]["start_y"]), radius)
        events.append(
            AmbiguousEvent(
                event_id=eidx,
                frame=frame,
                event_type=evt_type,
                original_decision=original,
                parent_candidate=p,
                new_particles=new_ids,
                x=float(near.iloc[0]["start_x"]),
                y=float(near.iloc[0]["start_y"]),
                min_dist_px=float(near.iloc[0]["dist"]),
                n_neighbors=n_neighbors,
                note=f"death@{te}",
            )
        )
        eidx += 1

    # Pattern C: birth near endpoint in crowded neighborhood (possible false new / ID switch)
    for _, b in starts.iterrows():
        bid = int(b["particle"])
        if bid in handled_births:
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
                note="birth-near-endpoint",
            )
        )
        eidx += 1
    return events


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
        return {"provider": "gemini", "decision": "defer", "confidence": 0.0, "reason": "gemini_unavailable"}
    return _adjudicate_heuristic(e)


def _adjudicate_heuristic(e: AmbiguousEvent) -> dict[str, Any]:
    if e.event_type == "potential_division" and len(e.new_particles) >= 2:
        return {"provider": "heuristic", "decision": "true_division", "confidence": 0.66, "reason": "2 nearby births after disappearance"}
    if e.event_type == "cluster_overlap_id_change":
        return {"provider": "heuristic", "decision": "merge_or_touch_no_new_id", "confidence": 0.60, "reason": "crowded overlap near endpoint"}
    if e.min_dist_px is not None and e.min_dist_px <= 20:
        return {"provider": "heuristic", "decision": "continue_same_track", "confidence": 0.62, "reason": "birth very close to endpoint"}
    return {"provider": "heuristic", "decision": "true_new_cell", "confidence": 0.56, "reason": "insufficient continuation evidence"}


def _adjudicate_gemini(e: AmbiguousEvent, crops_b64: list[str], tr: TrackingParams) -> dict[str, Any] | None:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    model = tr.adjudication_model or "gemini-2.5-flash"
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    schema_prompt = (
        "You are adjudicating ambiguous microscopy cell-tracking events from a short temporal clip. "
        "Return strict JSON only with fields: decision, confidence, reason. "
        "decision must be one of: continue_same_track, true_new_cell, true_division, merge_or_touch_no_new_id. "
        "confidence must be a float in [0,1], where >=0.80 means strong visual evidence and <=0.50 means uncertain. "
        f"event_type={e.event_type}; frame={e.frame}; parent_candidate={e.parent_candidate}; "
        f"new_particles={e.new_particles}; min_dist_px={e.min_dist_px}; n_neighbors={e.n_neighbors}. "
        "Use temporal continuity across the sequence: if morphology and centroid continuity suggest the same cell, "
        "favor continue_same_track or merge_or_touch_no_new_id. "
        "Use true_division only when a plausible parent-to-two-daughters transition is visible across frames, "
        "not just transient touching/overlap in one frame. Keep reason concise."
    )
    parts: list[dict[str, Any]] = [{"text": schema_prompt}]
    for b64 in crops_b64[:9]:
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64}})

    payload = {
        "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"},
        "contents": [{"role": "user", "parts": parts}],
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            resp = json.loads(r.read().decode("utf-8"))
    except Exception:
        return None

    text = ""
    try:
        text = resp["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return None
    try:
        obj = json.loads(text)
    except Exception:
        return None
    d = str(obj.get("decision", "defer"))
    if d not in {"continue_same_track", "true_new_cell", "true_division", "merge_or_touch_no_new_id"}:
        d = "defer"
    return {
        "provider": "gemini",
        "decision": d,
        "confidence": float(obj.get("confidence", 0.0)),
        "reason": str(obj.get("reason", ""))[:500],
    }


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
    if d in {"continue_same_track", "merge_or_touch_no_new_id"}:
        if e.parent_candidate is None or not e.new_particles:
            return tracks, "invalid_continue_payload", False
        child = int(e.new_particles[0])
        parent = int(e.parent_candidate)
        idx = out["particle"] == child
        if not idx.any():
            return tracks, "child_not_found", False
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
        if "parent" in out.columns:
            out.loc[out["particle"].isin(kids), "parent"] = parent
        if "fate" in out.columns:
            out.loc[out["particle"] == parent, "fate"] = "DIVIDE"
        return out, f"set_parent_{parent}_for_{kids[0]}_{kids[1]}", True

    if d == "true_new_cell":
        return tracks, "kept_true_new_cell", False

    return tracks, "no_change", False


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
