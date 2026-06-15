# Self-contained benchmark image: vLLM + guidellm + this repo's scripts.
# Nothing here is deployer-specific — it is a stock vLLM image plus the client
# bench tools, so it runs anywhere you can run a CUDA container (your CC VM and
# your non-CC baseline VM alike).
#
# Pinned to vLLM 0.21.0: 0.22.x ships a broken nvidia-cutlass-dsl that ICE-crashes
# CUDA-graph capture for every NVFP4 MoE kernel path, so it cannot serve the Kimi
# NVFP4 preset. Dense models (Gemma) are unaffected by the pin.
ARG VLLM_VERSION=v0.21.0-ubuntu2404
FROM vllm/vllm-openai:${VLLM_VERSION}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates jq \
    && rm -rf /var/lib/apt/lists/*

# Client-side bench deps only. We deliberately do NOT reinstall vllm[bench] —
# that would upgrade vLLM past the 0.21.0 pin. The base image already provides
# torch + the `vllm bench` subcommands.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace
COPY . /workspace

# Long-lived bench shell; exec in and run the scripts.
ENTRYPOINT []
CMD ["sleep", "infinity"]
