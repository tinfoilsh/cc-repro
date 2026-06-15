#!/usr/bin/env bash
# Run ONE arm (cc-on or cc-off): capture provenance, verify the GPU CC mode
# in-band, run the concurrency sweep, write a manifest. Run AFTER serve.sh
# reports healthy, in the SAME shell (so the preset env is in scope).
#
#   Usage: bash e2e/run_arm.sh <cc-on|cc-off>
#
# Boot the SAME image twice — once in your full confidential stack (CPU TEE +
# GPU CC, e.g. an Intel TDX guest with the GPU in CC mode) labeled cc-on, and
# once in a non-confidential VM (GPU CC off) labeled cc-off — and run this in
# each. on-vs-off = full confidential-stack overhead. nvidia-smi conf-compute is
# recorded so the mode is proven, not assumed.
#
# Sweep config comes from the sourced preset: PROMPT_TOKENS, OUTPUT_TOKENS,
# CONCS, MAX_SECONDS, SEED. Output is compatible with analysis/compare_sweep.py.
set -uo pipefail
COND=${1:?usage: run_arm.sh <cc-on|cc-off>}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8001}"
SERVED_NAME="${SERVED_NAME:-model}"
MODEL="${MODEL:-unknown}"
TOKENIZER="${TOKENIZER:-$MODEL}"
TP="${TP:-1}"
LABEL="${LABEL:-run1}"
PROMPT_TOKENS="${PROMPT_TOKENS:-1024}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-256}"
CONCS="${CONCS:-1 4 16 64 128}"
MAX_SECONDS="${MAX_SECONDS:-75}"
SEED="${SEED:-7}"
# Optional warmup exclusion: set WARMUP_PCT (e.g. 0.1) to discard the first fraction
# of each point (the first request triggers CC-sensitive CUDA-graph capture). Off by
# default; needs a guidellm build that supports --warmup-percent.
WARMUP_ARG="${WARMUP_PCT:+--warmup-percent $WARMUP_PCT}"
OUT="${RESULTS_DIR:-./results}/${COND}/${LABEL}"
BASE="http://localhost:${PORT}"
mkdir -p "$OUT"

curl -sf "${BASE}/health" >/dev/null || { echo "[arm] server not healthy on ${BASE}; run serve.sh first"; exit 1; }

echo "[arm] === provenance -> $OUT ==="
bash "$HERE/provenance.sh" > "$OUT/provenance.txt" 2>&1 || true
nvidia-smi -q > "$OUT/nvidia-smi.q.txt" 2>&1 || true
cp "${LOG:-./serve.log}" "$OUT/serve.log" 2>/dev/null || true
# Verify the GPU is actually in the CC mode we think it is (in-band via nvidia-smi).
CC_DETECT=$( (nvidia-smi conf-compute -f 2>/dev/null \
              || grep -i "Confidential Compute" "$OUT/nvidia-smi.q.txt" | head -1) | tr -d '\n')
echo "${CC_DETECT:-unknown}" > "$OUT/cc-mode.txt"
echo "[arm] declared=$COND  detected=[${CC_DETECT:-unknown}]"
vllm --version > "$OUT/vllm-version.txt" 2>&1 || true
pip freeze 2>/dev/null | grep -iE "^(vllm|guidellm|flashinfer|torch)" > "$OUT/pkgs.txt" || true
curl -s "${BASE}/metrics" > "$OUT/metrics.before.txt" 2>&1 || true

echo "[arm] === sweep: in=$PROMPT_TOKENS out=$OUTPUT_TOKENS conc=[$CONCS] ==="
for C in $CONCS; do
  echo "[arm]   concurrent=$C"
  guidellm benchmark run \
    --target "$BASE" --model "$SERVED_NAME" \
    --processor "$TOKENIZER" --processor-args '{"trust_remote_code": true}' \
    --rate-type concurrent --rate "$C" \
    --data "prompt_tokens=${PROMPT_TOKENS},output_tokens=${OUTPUT_TOKENS}" \
    --max-seconds "$MAX_SECONDS" --random-seed "$SEED" --disable-progress $WARMUP_ARG \
    --output-path "$OUT/c${C}.json" > "$OUT/c${C}.log" 2>&1 \
    && echo "[arm]   ok -> c${C}.json" \
    || { echo "[arm]   WARN c=$C failed (tail):"; tail -6 "$OUT/c${C}.log"; }
done

curl -s "${BASE}/metrics" > "$OUT/metrics.after.txt" 2>&1 || true
cat > "$OUT/manifest.json" <<EOF
{
  "condition": "${COND}",
  "label": "${LABEL}",
  "model": "${MODEL}",
  "served_model_name": "${SERVED_NAME}",
  "tensor_parallel": ${TP},
  "sweep": {"prompt_tokens": ${PROMPT_TOKENS}, "output_tokens": ${OUTPUT_TOKENS}, "concurrency": "${CONCS}"},
  "served_via": "localhost in-process vLLM (no proxy/TLS)",
  "detected_cc": "$(printf '%s' "${CC_DETECT:-unknown}" | sed 's/"/\\"/g')"
}
EOF
echo "[arm] done -> $OUT"
