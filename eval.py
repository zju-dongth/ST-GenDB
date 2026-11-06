"""Query accuracy and hardware telemetry evaluation utilities for ST-GenDB."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence
import numpy as np

from st_gendb.pipeline import QueryExecutionResult


def evaluate_query(result: QueryExecutionResult) -> Dict[str, Any]:
    """Summarizes accuracy, latency, memory footprint, and admission telemetry for one query."""
    report = result.audit_report
    return {
        "query_id": result.query_id,
        "query_class": result.query_class,
        "precision": round(float(result.precision), 2),
        "recall": round(float(result.recall), 2),
        "f1": round(float(result.f1), 2),
        "latency_ms": round(float(result.latency_sec) * 1000.0, 2),
        "peak_vram_gb": round(float(result.peak_vram_gb), 2),
        "visual_tokens_materialized": int(result.tokens_evaluated),
        "token_savings_ratio": round(float(result.token_savings_ratio), 2),
        "buffer_hit_rate": round(float(result.cache_hit_rate), 2),
        "clause_repairs": int(report.repair_iterations) if report is not None else 0,
        "clauses_certified": bool(report.passed_all) if report is not None else True,
        "oom_occurred": bool(result.oom_occurred),
        "num_matches": len(result.matched_results),
    }


def evaluate_workload(results: Sequence[QueryExecutionResult]) -> Dict[str, Any]:
    """Aggregates per-query execution results into workload-level evaluation metrics."""
    per_query: List[Dict[str, Any]] = [evaluate_query(r) for r in results]
    if not per_query:
        return {
            "num_queries": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "peak_vram_gb": 0.0,
            "visual_tokens_materialized": 0,
            "mean_visual_tokens_materialized": 0.0,
            "token_savings_ratio": 0.0,
            "buffer_hit_rate": 0.0,
            "clause_c1_c3_repairs": 0,
            "certified_queries": 0,
            "oom_count": 0,
            "queries": [],
        }

    latencies = np.asarray([q["latency_ms"] for q in per_query], dtype=np.float64)
    total_tokens = int(sum(q["visual_tokens_materialized"] for q in per_query))
    return {
        "num_queries": len(per_query),
        "precision": round(float(np.mean([q["precision"] for q in per_query])), 2),
        "recall": round(float(np.mean([q["recall"] for q in per_query])), 2),
        "f1": round(float(np.mean([q["f1"] for q in per_query])), 2),
        "latency_ms": round(float(np.mean(latencies)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "peak_vram_gb": round(float(max(q["peak_vram_gb"] for q in per_query)), 2),
        "visual_tokens_materialized": total_tokens,
        "mean_visual_tokens_materialized": round(float(total_tokens) / float(len(per_query)), 2),
        "token_savings_ratio": round(float(np.mean([q["token_savings_ratio"] for q in per_query])), 2),
        "buffer_hit_rate": round(float(np.mean([q["buffer_hit_rate"] for q in per_query])), 2),
        "clause_c1_c3_repairs": int(sum(q["clause_repairs"] for q in per_query)),
        "certified_queries": int(sum(1 for q in per_query if q["clauses_certified"])),
        "oom_count": int(sum(1 for q in per_query if q["oom_occurred"])),
        "queries": per_query,
    }
