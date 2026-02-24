# Inference Optimization MVP

This project is a functional MVP designed to **simulate** the process of optimizing a production-grade model suite for cost-effective infrastructure. It mimics the technical challenges of maintaining performance (latency and throughput) and accuracy when migrating to budget hardware.

## Project Goal
The goal of this MVP is to demonstrate how to resolve the "performance vs. cost" trade-off through two primary levers:
1.  **Precision Engineering:** Using quantization to reduce compute overhead.
2.  **Load Distribution:** Using horizontal scaling and intelligent routing to recover throughput.

## The Simulation Context
This MVP simulates a suite of four models commonly used in production environments:
*   **Encoders:** Text embedding models for search.
*   **Multilingual Encoders:** Cross-language semantic models.
*   **Re-rankers:** High-accuracy relevance scoring models.
*   **Speech-to-Text (STT):** Audio transcription models.

By transitioning from FP32 (Full Precision) to FP16 (Optimized) across multiple workers, the project demonstrates how to achieve a "0.95x accuracy at a fraction of the cost" outcome.

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd inference-optimization-mvp
```

### 2. Environment Setup (using `uv`)
This project uses `uv` for fast and reliable Python package management.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows

# Sync dependencies
uv pip install -r requirements.txt
```

---

## Running the Simulation

You can run the full optimization benchmark using the following command:

```bash
# Simulate optimization on CPU with 2 workers
python benchmarks/benchmark.py --runs 10 --device cpu

# Simulate optimization on multiple GPUs (e.g., dual T4 setup)
python benchmarks/benchmark.py --runs 50 --devices cuda:0,cuda:1
```

### Visualizing Results
After running the benchmark, generate the comparison charts:
```bash
python benchmarks/plot_results.py
```
Charts will be saved to the `results/` directory.

---

## Sample Simulation Results

The following table illustrates the typical output of the MVP, showing how the **Optimized** configuration (2 workers + lowered precision) scales throughput (RPS) while maintaining accuracy above the 95% threshold.

| Model             | Precision | Workers | p50 ms  | p95 ms  | RPS     | Acc %  |
|-------------------|-----------|---------|---------|---------|---------|--------|
| **ENCODER**       | fp32      | 1       | 80.25   | 142.13  | 12.46   | 100.0  |
| **ENCODER**       | fp16      | 2       | 188.07  | 191.06  | 10.38   | 96.20  |
| **MULTILINGUAL**  | fp32      | 1       | 184.21  | 319.68  | 5.43    | 100.0  |
| **MULTILINGUAL**  | fp16      | 2       | 357.04  | 368.39  | 5.40    | 98.57  |
| **RERANKER**      | fp32      | 1       | 330.44  | 611.68  | 3.03    | 100.0  |
| **RERANKER**      | fp16      | 2       | 104.60  | 124.23  | 15.66   | 100.0  |
| **STT**           | fp32      | 1       | 1139.48 | 1419.64 | 0.88    | 100.0  |
| **STT**           | fp16      | 2       | 1627.59 | 2105.49 | 0.93    | 100.0  |

---

## Project Structure
- `models/`: Model loading and quantization logic.
- `load_balancer/`: Thread-safe Least-Connections routing.
- `benchmarks/`: Simulation harness for latency, RPS, and accuracy.
- `results/`: JSON logs and visualization charts.
