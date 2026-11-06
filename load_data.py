"""Dataset, camera topology, query workload, and calibration loaders for ST-GenDB."""

from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from st_gendb.calibration import PredicateCalibrator, SubspaceProjector
from st_gendb.schema import CameraTopology, VisualDetectionSlice
from st_gendb.synthesizer import DeclarativeVideoQuery


def _decode_embedding(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, str):
        raw_bytes = base64.b64decode(value)
        return np.frombuffer(raw_bytes, dtype=np.float32).copy()
    return np.asarray(value, dtype=np.float32)


def _read_json_or_jsonl(file_path: str, list_keys: Tuple[str, ...] = ()) -> List[Any]:
    if file_path.endswith(".jsonl"):
        records: List[Any] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    records.append(json.loads(stripped))
        return records

    with open(file_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in list_keys:
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f"Unsupported record container in {file_path!r}")


def load_topology(
    topology_file: str,
    *,
    v_min_mps: float = 5.5,
    v_max_mps: float = 21.0,
    tau_sig_sec: float = 18.0,
) -> CameraTopology:
    """Loads a directed road-network CameraTopology from a .json or .npz archive."""
    if topology_file.endswith(".npz"):
        with np.load(topology_file, allow_pickle=False) as data:
            cam_ids = [int(c) for c in data["camera_ids"].tolist()]
            dist_mat = np.asarray(data["dist_matrix"], dtype=np.float32)
            adj_mat = np.asarray(data["adjacency"], dtype=np.int32)
            v_min = float(data["v_min_mps"]) if "v_min_mps" in data.files else v_min_mps
            v_max = float(data["v_max_mps"]) if "v_max_mps" in data.files else v_max_mps
            tau_sig = float(data["tau_sig_sec"]) if "tau_sig_sec" in data.files else tau_sig_sec
            fov_ov = float(data["fov_overlap_m"]) if "fov_overlap_m" in data.files else 0.0
            coloc = float(data["co_located_slack_sec"]) if "co_located_slack_sec" in data.files else 0.0
        return CameraTopology(
            num_cameras=len(cam_ids),
            camera_ids=cam_ids,
            dist_matrix=dist_mat,
            adjacency=adj_mat,
            v_min_mps=v_min,
            v_max_mps=v_max,
            tau_sig_sec=tau_sig,
            fov_overlap_m=fov_ov,
            co_located_slack_sec=coloc,
        )

    with open(topology_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    cam_ids = [int(c) for c in meta["camera_ids"]]
    dist_mat = np.asarray(meta["dist_matrix"], dtype=np.float32)
    adj_mat = np.asarray(meta["adjacency"], dtype=np.int32)
    return CameraTopology(
        num_cameras=len(cam_ids),
        camera_ids=cam_ids,
        dist_matrix=dist_mat,
        adjacency=adj_mat,
        v_min_mps=float(meta.get("v_min_mps", v_min_mps)),
        v_max_mps=float(meta.get("v_max_mps", v_max_mps)),
        tau_sig_sec=float(meta.get("tau_sig_sec", tau_sig_sec)),
        fov_overlap_m=float(meta.get("fov_overlap_m", 0.0)),
        co_located_slack_sec=float(meta.get("co_located_slack_sec", 0.0)),
        congestion_speed_floor_mps=float(meta.get("congestion_speed_floor_mps", 0.8)),
        congestion_clearance_scale=float(meta.get("congestion_clearance_scale", 2.2)),
    )


def load_slices(slices_file: str) -> List[VisualDetectionSlice]:
    """Loads candidate camera-time detection slices from a .json or .jsonl file."""
    raw_records = _read_json_or_jsonl(slices_file, list_keys=("slices", "records"))
    slices: List[VisualDetectionSlice] = []
    for rec in raw_records:
        t0_embed = _decode_embedding(rec["tier0_embed"])
        if t0_embed is None:
            raise ValueError(f"Slice {rec.get('slice_id')} is missing required tier0_embed.")
        slices.append(
            VisualDetectionSlice(
                slice_id=int(rec["slice_id"]),
                cam_id=int(rec["cam_id"]),
                t_enter=float(rec["t_enter"]),
                t_exit=float(rec["t_exit"]),
                tier0_embed=t0_embed,
                tier1_embed=_decode_embedding(rec.get("tier1_embed")),
                tier2_embed=_decode_embedding(rec.get("tier2_embed")),
                attribute_scores_t0={str(k): float(v) for k, v in (rec.get("attribute_scores_t0") or {}).items()},
                attribute_scores_t1={str(k): float(v) for k, v in (rec.get("attribute_scores_t1") or {}).items()},
                attribute_scores_t2={str(k): float(v) for k, v in (rec.get("attribute_scores_t2") or {}).items()},
                predicate_scores={str(k): float(v) for k, v in (rec.get("predicate_scores") or {}).items()},
                frame_paths=tuple(str(p) for p in (rec.get("frame_paths") or ())),
                t0_infer_sec=float(rec.get("t0_infer_sec", 0.0)),
                t1_infer_sec=float(rec.get("t1_infer_sec", 0.0)),
                t2_infer_sec=float(rec.get("t2_infer_sec", 0.0)),
                gt_entity_id=int(rec.get("gt_entity_id", -1)),
                gt_attributes={str(a) for a in (rec.get("gt_attributes") or ())},
            )
        )
    return slices


def load_queries(
    queries_file: str,
    target_cameras: Optional[List[int]] = None,
) -> List[DeclarativeVideoQuery]:
    """Loads declarative SQL queries and optional evaluation ground truth from .json or .jsonl."""
    raw_items = _read_json_or_jsonl(queries_file, list_keys=("queries", "records"))
    queries: List[DeclarativeVideoQuery] = []
    for idx, item in enumerate(raw_items):
        if isinstance(item, str):
            queries.append(
                DeclarativeVideoQuery.from_sql(
                    sql_text=item,
                    query_id=f"Q_{idx + 1:03d}",
                    target_cameras=target_cameras,
                )
            )
            continue

        qid = str(item.get("query_id", f"Q_{idx + 1:03d}"))
        q_cams = item.get("target_cameras", target_cameras)
        cam_list = [int(c) for c in q_cams] if q_cams is not None else None
        query = DeclarativeVideoQuery.from_sql(
            sql_text=str(item["sql"]),
            query_id=qid,
            target_cameras=cam_list,
        )
        if item.get("gt_target_entities") is not None:
            query.gt_target_entities = [int(e) for e in item["gt_target_entities"]]
        if item.get("gt_comove_pairs") is not None:
            query.gt_comove_pairs = [(int(pair[0]), int(pair[1])) for pair in item["gt_comove_pairs"]]
        queries.append(query)
    return queries


def load_calibration(
    calibration_file: str,
) -> Tuple[Dict[int, SubspaceProjector], Optional[PredicateCalibrator]]:
    """Loads optional validation subspace projectors and predicate log-odds calibration from .npz."""
    if not calibration_file or not os.path.exists(calibration_file):
        return {}, None

    projectors: Dict[int, SubspaceProjector] = {}
    pred_calibrator: Optional[PredicateCalibrator] = None
    with np.load(calibration_file, allow_pickle=False) as data:
        for tier in (0, 1, 2):
            mean_key = f"tier{tier}_mean"
            proj_key = f"tier{tier}_proj"
            if mean_key in data.files and proj_key in data.files:
                alpha_key = f"tier{tier}_mix_alpha"
                raw_key = f"tier{tier}_raw_basis"
                mix_alpha = float(data[alpha_key]) if alpha_key in data.files else 0.85
                raw_basis = np.asarray(data[raw_key], dtype=np.float32) if raw_key in data.files else None
                projectors[tier] = SubspaceProjector(
                    mean_vec=np.asarray(data[mean_key], dtype=np.float32),
                    proj_matrix=np.asarray(data[proj_key], dtype=np.float32),
                    mix_alpha=mix_alpha,
                    raw_basis=raw_basis,
                )

        if "pred_scale_json" in data.files and "pred_bias_json" in data.files:
            scales = {str(k): float(v) for k, v in json.loads(str(data["pred_scale_json"].item())).items()}
            biases = {str(k): float(v) for k, v in json.loads(str(data["pred_bias_json"].item())).items()}
            pred_calibrator = PredicateCalibrator(scale=scales, bias=biases)

    return projectors, pred_calibrator
