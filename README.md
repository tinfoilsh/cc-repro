# Confidential-computing inference overhead — reproducible benchmark

Measures the inference overhead of running an LLM under confidential computing
(CPU TEE + GPU CC) versus a plain VM, using only stock components — `vllm serve`
+ `guidellm` + public HuggingFace weights. Run the same scripts in two
environments (one confidential, one not) and diff the results.

## What it measures

Two arms — `cc-on` = confidential stack (CPU TEE + GPU CC), `cc-off` = plain VM.
A model is served as a local in-process vLLM engine and benchmarked over
localhost with guidellm across a concurrency sweep, reporting TTFT, inter-token
latency (ITL/TPOT), and throughput.

## Requirements

- NVIDIA GPU(s) in a host that supports CC mode. Kimi preset = 8 GPUs (TP=8);
  Gemma preset = 1.
- Two environments to boot the same image in: a confidential one (CPU TEE + GPU
  CC on) and a non-confidential baseline (GPU CC off).
- vLLM 0.21.0 (the Dockerfile pins it; required for the Kimi NVFP4 MoE preset).
- A HuggingFace token with access to the model repos.

## Quickstart

```bash
# Build the image (or: pip install -r requirements.txt into a vLLM 0.21.0 env)
docker build -t cc-repro .

# Pick a preset and set your HF token
source presets/kimi-k2-6.env        # or: source presets/gemma-4-31b.env
export HF_TOKEN=...

# Run each arm in its environment: cc-off in the plain VM, cc-on in the CC VM.
# One CC VM usually holds all the GPUs, so run the two arms sequentially.
bash e2e/serve.sh                   # load weights, wait for /health
bash e2e/run_arm.sh cc-off          # -> results/cc-off/run1/   (use cc-on in the CC VM)

# Gather both arms' results/ into one tree, then compare
python3 analysis/compare_sweep.py results/cc-off/run1 results/cc-on/run1
```

## Configuration

All knobs live in the preset env files (`presets/*.env`); override per run via env:

| Var | Meaning |
|---|---|
| `MODEL`, `MODEL_REVISION` | HuggingFace repo id (+ optional commit) |
| `TP` | tensor-parallel size |
| `MAX_NUM_SEQS`, `GPU_MEM_UTIL`, `EXTRA_VLLM_ARGS` | passthrough to `vllm serve` |
| `PROMPT_TOKENS`, `OUTPUT_TOKENS`, `CONCS`, `MAX_SECONDS`, `SEED` | the guidellm sweep |
| `RESULTS_DIR`, `LABEL` | where results land (`results/<cond>/<label>/`) |

## Layout

```
e2e/         serve.sh, run_arm.sh, provenance.sh
analysis/    compare_sweep.py
presets/     kimi-k2-6.env, gemma-4-31b.env
```

## Reading the output

- `compare_sweep.py` — per concurrency: output tok/s off→on, loss%, "CC keeps"
  (on/off), TTFT and ITL overhead%. A fixed per-token CC tax shows up as a large
  loss at concurrency 1 that shrinks as batch grows.
- Each arm records `cc-mode.txt` / `provenance.txt` (mode, driver, versions, topology).

## Notes for a valid comparison

- **The two arms must differ only in CC mode.** Same image, identical serve flags,
  and the same GPU count / vCPU count / NUMA layout in both. Each arm verifies its
  mode in-band (`nvidia-smi conf-compute`).
- **n=1 by default.** Each point is one 60–75 s window; run n≥3 (vary `LABEL`) and
  read TTFT **p50**, not mean, especially at concurrency 1.
- **First-request spike.** The first request triggers CC-sensitive CUDA-graph
  capture; it lands in TTFT p99 unless you set `WARMUP_PCT` (needs a guidellm build
  with `--warmup-percent`).
- **Gemma spec decoding** is off by default (the config that runs on vLLM 0.21.0);
  enabling it needs vLLM 0.22.0 + the MTP assistant model (see the preset).
- **vLLM 0.21.0 pin** is required for the Kimi NVFP4 MoE preset (0.22.x breaks it);
  both arms of a model run the same version.
