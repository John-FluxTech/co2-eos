#!/usr/bin/env bash
# Benchmark the economized EOS hot path on the OVH V100S via flux-compute.
#
#   flux-compute run --cloud flux-ovh --upload . \
#       --script bench/gpu_bench.sh --fetch "bench-out:gpu-results"
#
# flux-compute uploads this repo to ~/co2-eos.  The Ubuntu 24.04 GPU image
# ships python3 without ensurepip/venv, so install those first; install
# jax[cuda12] LAST so the CUDA jaxlib wins the resolve over co2-eos's jax dep.
set -euo pipefail
export MPLBACKEND=Agg

sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip
python3 -m venv ~/venv && source ~/venv/bin/activate
pip install -q -U pip
pip install -q -e ~/co2-eos
pip install -q -U "jax[cuda12]"

# Must see a GPU device, else the benchmark would silently time the CPU.
# JAX reports CUDA devices with platform == 'gpu'.
python -c "import jax; print('JAX devices:', jax.devices()); \
assert any(d.platform == 'gpu' for d in jax.devices()), 'no GPU device visible'"

mkdir -p ~/bench-out
cd ~/co2-eos

echo '=== v0.1-autodiff vs economized path, spec batch sizes (V100S) ==='
python -u bench/compare.py --json ~/bench-out/compare_gpu_econ.json | tee ~/bench-out/compare_gpu_econ.txt

echo '=== scaling to GPU saturation (V100S) ==='
python -u bench/compare.py --ns 64,4096,16384,65536,262144,1048576 \
    --json ~/bench-out/compare_gpu_scaling_econ.json | tee ~/bench-out/compare_gpu_scaling_econ.txt

echo '=== hot-path component profile (V100S) ==='
python -u bench/profile_hotpath.py --json ~/bench-out/profile_hotpath_gpu_econ.json | tee ~/bench-out/profile_hotpath_gpu_econ.txt

nvidia-smi --query-gpu=name,memory.total --format=csv | tee ~/bench-out/gpu_info.txt
echo "gpu_bench.sh complete"
