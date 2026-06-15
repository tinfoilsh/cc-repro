#!/bin/bash
# Config/provenance dump: GPU CC mode, driver/CUDA/NCCL versions, topology, TDX/SWIOTLB.
# Every section tolerates failure (some conf-compute flags vary by driver).
sec() { echo; echo "===== $* ====="; }

sec date; date -u
sec nvidia-smi; nvidia-smi
sec "driver version"; cat /proc/driver/nvidia/version 2>&1
sec "conf-compute help"; nvidia-smi conf-compute --help 2>&1
sec "conf-compute -f (features)"; nvidia-smi conf-compute -f 2>&1
sec "conf-compute -e (environment)"; nvidia-smi conf-compute -e 2>&1
sec "conf-compute -d (devtools mode)"; nvidia-smi conf-compute -d 2>&1
sec "conf-compute -grs (gpus ready state)"; nvidia-smi conf-compute -grs 2>&1
sec "conf-compute -srs (ready state setting)"; nvidia-smi conf-compute -srs 2>&1 | head -3
sec "conf-compute -mgm (multi-gpu mode)"; nvidia-smi conf-compute -mgm 2>&1
sec "topo -m"; nvidia-smi topo -m 2>&1
sec "nvlink -s (GPU0)"; nvidia-smi nvlink -s -i 0 2>&1
sec "nvidia-smi -q (GPU0: identity/vbios/clocks)"
nvidia-smi -q -i 0 2>&1 | grep -A2 -iE 'product name|vbios|driver version|cuda version|clocks$' | head -40
sec "cpu count + numa topology (MUST match across cc-on/cc-off arms)"
nproc 2>&1; echo "--- lscpu ---"; lscpu 2>&1 | head -25; echo "--- numa ---"; (numactl --hardware 2>&1 || echo "(numactl absent)")
sec "gpu clocks / power (drift check across arms)"
nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu --format=csv 2>&1
sec "kernel cmdline (guest kernel via /proc)"; cat /proc/cmdline 2>&1
sec "swiotlb / tdx in dmesg (may need privs)"
dmesg 2>/dev/null | grep -iE 'swiotlb|tdx|sev|conf.*comput' | head -20 || echo "(dmesg unavailable in container)"
sec "torch/cuda/nccl versions"
python3 -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('nccl', torch.cuda.nccl.version()); print('gpus', torch.cuda.device_count())" 2>&1
sec "vllm version"; python3 -c "import vllm; print(vllm.__version__)" 2>&1
