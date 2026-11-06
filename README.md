# ST-GenDB: Admission-Gated Plan Synthesis and Progressive Token Buffering for Multi-Camera Video

Reference implementation for the paper *"ST-GenDB: Admission-Gated Plan Synthesis and Progressive Token Buffering for Multi-Camera Video"*.

## Repository Structure

```text
.
├── main.py                     # Main query compilation and execution entrypoint
├── load_data.py                # Topology, detection slice, query, and calibration loader
├── eval.py                     # Query accuracy and hardware telemetry evaluation metrics
├── st_gendb/                   # Core optimizer, admission gate, buffer pool, and operators
│   ├── schema.py               # Camera topology G, Top-k provenance view, and plan DAG
│   ├── synthesizer.py          # SQL parser and Intent-to-Plan Optimizer (IPO)
│   ├── admission_gate.py       # Clause-Level Admission Gate (Clauses C1, C2, C3)
│   ├── buffer_pool.py          # 3-tier visual token buffer pool and provenance-heat cache
│   ├── algebra.py              # Multi-camera operators (SEM_FILTER, SEM_LINK, SEM_COMOVE_JOIN)
│   ├── calibration.py          # Validation subspace projection and log-odds calibration
│   ├── vlm_backend.py          # 4-bit NF4 Qwen2.5-VL-7B-Instruct visual-KV slot backend
│   └── pipeline.py             # End-to-end STGenDBEngine query compiler and executor
└── configs/
    └── default_4090.yaml       # Single-GPU RTX 4090 configuration template
```

## Quick Start

1. Install dependencies (Python 3.10+):
   ```bash
   pip install -r requirements.txt
   ```

2. Configure prepared dataset and model checkpoint paths in [`configs/default_4090.yaml`](configs/default_4090.yaml).

3. Run the query execution pipeline:
   ```bash
   python3 main.py --cfg_file configs/default_4090.yaml --allow-vlm
   python3 main.py --cfg_file configs/default_4090.yaml --sql "SELECT clip_id FROM camera_streams WHERE cam_id IN (1, 2) AND SEM_FILTER(clip, 'white van');"
   ```
