#!/usr/bin/env bash
# Remote (omnirun) generation of the DREGON + Matrice100 onboard ego-noise
# dataset. Generation only -- the dload commit runs locally afterwards (R2 creds
# stay off the remote box; see scripts/egonoise_generate.py --commit-from).
#
# Usage (omnirun picks the backend; only resources are requested):
#   omnirun submit --gpus 1 --time 2h --mem 16 -- bash scripts/egonoise_gpu_job.sh
set -euo pipefail

DRONES="${DRONES:-dregon matrice100}"
SEEDS="${SEEDS:-0 1 2}"
OUT="${OUT:-results/egonoise}"

# jax[cuda12] loads its CUDA from the pip nvidia wheels; prepend their lib dirs
# so XLA finds libcusparse/etc even where the base image's /usr/local/cuda is
# incomplete (the documented Kaggle gotcha; harmless on other backends).
NV="$(uv run --extra gpu python - <<'PY'
import glob, os, sysconfig
libs = sorted(glob.glob(os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "*", "lib")))
print(":".join(libs))
PY
)"
export LD_LIBRARY_PATH="${NV}:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"

# Verify the accelerator: trust CudaDevice, not runtime feel (CLAUDE.md).
uv run --extra gpu python -c "import jax; print('JAX DEVICES:', jax.devices())"

# shellcheck disable=SC2086
uv run --extra gpu python -u scripts/egonoise_generate.py \
    --drones ${DRONES} --seeds ${SEEDS} --out "${OUT}"

echo "generation done -> ${OUT}"
ls -la "${OUT}" || true
