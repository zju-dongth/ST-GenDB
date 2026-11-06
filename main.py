"""Main pipeline entrypoint for ST-GenDB multi-camera video query execution."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional
import yaml

from eval import evaluate_workload
from load_data import load_calibration, load_queries, load_slices, load_topology
from st_gendb import Qwen25VLHardwareBackend, STGenDBEngine, VisualDetectionSlice


def _resolve_path(path_str: str, base_dir: str) -> str:
    if not path_str or os.path.isabs(path_str) or os.path.exists(path_str):
        return path_str
    candidate = os.path.normpath(os.path.join(base_dir, path_str))
    return candidate if os.path.exists(candidate) else path_str


def _apply_slice_calibration(
    slices: List[VisualDetectionSlice],
    projectors_by_tier: Dict[int, Any],
    predicate_calibrator: Optional[Any],
) -> None:
    if not projectors_by_tier and predicate_calibrator is None:
        return
    p0 = projectors_by_tier.get(0)
    p1 = projectors_by_tier.get(1)
    p2 = projectors_by_tier.get(2)
    for s in slices:
        if p0 is not None and s.tier0_embed.shape[0] == p0.mean_vec.shape[0]:
            s.tier0_embed = p0.transform(s.tier0_embed)
        if p1 is not None and s.tier1_embed is not None and s.tier1_embed.shape[0] == p1.mean_vec.shape[0]:
            s.tier1_embed = p1.transform(s.tier1_embed)
        if p2 is not None and s.tier2_embed is not None and s.tier2_embed.shape[0] == p2.mean_vec.shape[0]:
            s.tier2_embed = p2.transform(s.tier2_embed)
        if predicate_calibrator is not None:
            if s.attribute_scores_t0:
                s.attribute_scores_t0 = predicate_calibrator.calibrate_dict(s.attribute_scores_t0)
            if s.attribute_scores_t1:
                s.attribute_scores_t1 = predicate_calibrator.calibrate_dict(s.attribute_scores_t1)
            if s.attribute_scores_t2:
                s.attribute_scores_t2 = predicate_calibrator.calibrate_dict(s.attribute_scores_t2)


def run_pipeline(
    cfg_file: str,
    sql_override: Optional[str] = None,
    mode_override: Optional[str] = None,
    allow_vlm: bool = False,
) -> Dict[str, Any]:
    """Loads prepared data, compiles admission-certified plans, and evaluates the query workload."""
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg_dir = os.path.dirname(os.path.abspath(cfg_file))
    ds_cfg = cfg.get("dataset", {})
    hw_cfg = cfg.get("hardware", {})
    topo_defaults = cfg.get("topology_defaults", {})

    topology_path = _resolve_path(str(ds_cfg.get("topology_file", "")), cfg_dir)
    slices_path = _resolve_path(str(ds_cfg.get("slices_file", "")), cfg_dir)
    queries_path = _resolve_path(str(ds_cfg.get("queries_file", "")), cfg_dir)
    calib_path = _resolve_path(str(ds_cfg.get("calibration_file", "")), cfg_dir)
    vlm_model_path = _resolve_path(str(ds_cfg.get("vlm_model_path", "")), cfg_dir)

    topology = load_topology(
        topology_path,
        v_min_mps=float(topo_defaults.get("v_min_mps", 5.5)),
        v_max_mps=float(topo_defaults.get("v_max_mps", 21.0)),
        tau_sig_sec=float(topo_defaults.get("tau_sig_sec", 18.0)),
    )
    slices = load_slices(slices_path)
    projectors_by_tier, predicate_calibrator = load_calibration(calib_path)

    vlm_backend: Optional[Qwen25VLHardwareBackend] = None
    if allow_vlm:
        if not vlm_model_path:
            raise ValueError("dataset.vlm_model_path must be set when --allow-vlm is enabled.")
        vlm_backend = Qwen25VLHardwareBackend(
            model_path=vlm_model_path,
            kv_bytes_per_token=int(hw_cfg.get("kv_bytes_per_token", 131072)),
            hard_vram_limit_gb=float(hw_cfg.get("hard_vram_limit_gb", 24.0)),
        )
        if projectors_by_tier or predicate_calibrator is not None:
            vlm_backend.set_calibration(
                projector=projectors_by_tier or None,
                predicate_calib=predicate_calibrator,
            )
    else:
        _apply_slice_calibration(slices, projectors_by_tier, predicate_calibrator)

    configured_snapshots = int(ds_cfg.get("total_indexed_snapshots", 0) or 0)
    total_indexed = configured_snapshots if configured_snapshots > 0 else len(slices)

    engine = STGenDBEngine.from_yaml(
        config_path=cfg_file,
        topology=topology,
        total_indexed_snapshots=total_indexed,
        vlm_backend=vlm_backend,
    )
    if mode_override is not None:
        mode_clean = mode_override.strip().lower()
        if mode_clean not in ("warm", "cold"):
            raise ValueError(f"Unsupported execution mode: {mode_override!r}")
        engine.execution_mode = mode_clean

    try:
        if sql_override is not None:
            results = [engine.execute_sql(sql_override, slices, query_id="SQL_CLI")]
        else:
            queries = load_queries(queries_path, target_cameras=list(topology.camera_ids))
            results: List[Any] = [engine.execute_query(q, slices) for q in queries]
        return evaluate_workload(results)
    finally:
        engine.reset_hardware_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="ST-GenDB Multi-Camera Video Query Pipeline")
    parser.add_argument(
        "--cfg_file",
        type=str,
        default="configs/default_4090.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--sql",
        type=str,
        default=None,
        help="Optional single declarative SQL query override",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["warm", "cold"],
        default=None,
        help="Optional session execution mode override ('warm' or 'cold')",
    )
    parser.add_argument(
        "--allow-vlm",
        dest="allow_vlm",
        action="store_true",
        help="Enable live 4-bit NF4 Qwen2.5-VL-7B-Instruct GPU inference",
    )
    args = parser.parse_args()
    summary = run_pipeline(
        cfg_file=args.cfg_file,
        sql_override=args.sql,
        mode_override=args.mode,
        allow_vlm=args.allow_vlm,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
