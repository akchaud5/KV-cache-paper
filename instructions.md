# H100 Evaluation Checklist

Follow these steps after cloning the repository onto a Vast.ai VM with an H100 GPU.

## 1. Provision the Environment
- Confirm the VM is running CUDA 12.2+ with NVIDIA drivers that detect the H100 (`nvidia-smi` should list `H100 SXM/PCIe`).
- Install system packages: `sudo apt-get update && sudo apt-get install -y build-essential git python3-pip`
- Optionally create a Python virtual environment (`python3 -m venv .venv && source .venv/bin/activate`).

## 2. Install Dependencies
- Upgrade pip: `python3 -m pip install --upgrade pip`
- Install project in editable mode with extras: `python3 -m pip install -e ".[dev]"` *(requires wheel, setuptools; already declared in `pyproject.toml`)*.
- Install CUDA-enabled PyTorch + FlashAttention (Ampere/Hopper builds):
  ```bash
  python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  python3 -m pip install flash-attn --no-build-isolation
  ```
- Install vLLM or SGLang if running live inference traces:
  ```bash
  python3 -m pip install "vllm>=0.4.0"
  ```

## 3. Data & Calibration Prep
- Capture attention statistics during prefill using your serving stack (vLLM/SGLang). Save per-layer aggregated attention tensors to `.pt` files under `artifacts/attn/`.
- (Optional) Run calibration prompts to populate mixed-precision statistics via `scripts/run_experiment.py --simulate` to verify configs before real data.

## 4. Run Compression Experiments
- Baseline (full cache): `python3 scripts/run_experiment.py --config experiments/baseline_full.yaml --simulate`
- Hybrid 8× compression: `python3 scripts/run_experiment.py --config experiments/hybrid_8x.yaml --cache-path path/to/cache.pt --attention-stats path/to/attn.pt`
- Hybrid 12× compression: same as above with `experiments/hybrid_12x.yaml`
- Each run emits a Rich table and optional JSON report (use `--report-path reports/run.json`).

## 5. Benchmark & Logging
- Use LongBench/RULER scripts (to be integrated) to evaluate quality; log metrics into `reports/`.
- Measure throughput/latency with your serving framework; update `metrics` sections in config YAMLs for reproducibility.
- Track GPU memory with `nvidia-smi --query-gpu=memory.used --format=csv -l 1` or Nsight Systems.

## 6. Ablations & Paper Artifacts
- Compare selectors (Pyramid/Snap/Hybrid) and bit-width mixes by editing YAMLs.
- Produce plots (compression vs. accuracy, TTFT/TPOT curves) using exported JSON.
- Document findings in `docs/` (create folder) and draft MLSys paper sections referencing collected data.

## 7. Housekeeping
- Keep commits reproducible: record exact driver version, CUDA version, PyTorch commit.
- Push reports/plots selectively; avoid uploading raw KV cache dumps (large files).
- When done, push results to GitHub and snapshot environment specs for paper appendix.

