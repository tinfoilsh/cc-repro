#!/usr/bin/env bash
# Serve the model as a LOCAL in-process vLLM engine, so the benchmark client can
# hit it over localhost (no TLS, no reverse proxy, no network) — the CC on/off
# delta must not be confounded by any deployment/serving path. The command is
# byte-for-byte identical in both arms; only the environment (CC on vs off)
# differs. Run this, wait for "healthy", then run run_arm.sh in the same shell.
#
#   Usage:  source presets/<model>.env ; bash e2e/serve.sh
#   or:     MODEL=<hf-repo> TP=8 bash e2e/serve.sh
set -euo pipefail

: "${MODEL:?set MODEL to the HuggingFace repo id of the weights (see presets/)}"
TP="${TP:-1}"
PORT="${PORT:-8001}"
SERVED_NAME="${SERVED_NAME:-model}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-48}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
HEALTH_TIMEOUT_MIN="${HEALTH_TIMEOUT_MIN:-35}"
LOG="${LOG:-./serve.log}"

# Refuse to run with the placeholder model id still in place.
case "$MODEL" in
  *"<SET-"*) echo "[serve] ERROR: edit the preset and set the real HF repo id for MODEL"; exit 2;;
esac

if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
  echo "[serve] already healthy on :${PORT}"; exit 0
fi

echo "[serve] vLLM $(vllm --version 2>/dev/null || echo '?')  model=$MODEL  TP=$TP  port=$PORT"
echo "[serve] extra args: ${EXTRA_VLLM_ARGS:-(none)}"
# shellcheck disable=SC2086  # EXTRA_VLLM_ARGS is an intentional word-split list
nohup vllm serve "$MODEL" \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --served-model-name "$SERVED_NAME" \
  --port "$PORT" \
  --trust-remote-code \
  ${MODEL_REVISION:+--revision "$MODEL_REVISION"} \
  $EXTRA_VLLM_ARGS \
  > "$LOG" 2>&1 &
disown
echo "[serve] pid $! -> $LOG ; waiting for /health (timeout ${HEALTH_TIMEOUT_MIN}m)"

for i in $(seq 1 $((HEALTH_TIMEOUT_MIN * 12))); do      # 12 * 5s = 1 min
  if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo "[serve] healthy after ~$((i * 5))s"; exit 0
  fi
  if ! pgrep -f "vllm serve" >/dev/null; then
    echo "[serve] ERROR: vllm exited early — last 40 log lines:"; tail -40 "$LOG"; exit 1
  fi
  sleep 5
done
echo "[serve] ERROR: not healthy after ${HEALTH_TIMEOUT_MIN}m — see $LOG"; tail -40 "$LOG"; exit 1
