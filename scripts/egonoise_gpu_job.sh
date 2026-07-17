#!/usr/bin/env bash
# Remote (omnirun) generation of the DREGON + Matrice100 onboard ego-noise
# dataset. Generates in-job and, when R2 creds are forwarded (COMMIT_DLOAD set
# + AWS_* env), commits straight to the dload bucket -- so nothing depends on
# the (ephemeral, e.g. colab) session outliving the job for an outputs pull.
#
# Usage (omnirun picks the backend; only resources are requested):
#   omnirun submit --gpus 1 --time 2h \
#     --env COMMIT_DLOAD=drone-egonoise \
#     --env AWS_ACCESS_KEY_ID=... --env AWS_SECRET_ACCESS_KEY=... \
#     --env AWS_DEFAULT_REGION=auto \
#     -- bash scripts/egonoise_gpu_job.sh
set -euo pipefail

DRONES="${DRONES:-dregon matrice100}"
SEEDS="${SEEDS:-0 1 2}"
OUT="${OUT:-results/egonoise}"
COMMIT_DLOAD="${COMMIT_DLOAD:-}"

# The dload commit needs the `data` extra; add it only when committing so the
# generation-only path stays lean.
EXTRAS=(--extra gpu)
COMMIT_ARGS=()
if [ -n "${COMMIT_DLOAD}" ]; then
    EXTRAS+=(--extra data)
    COMMIT_ARGS+=(--commit-dload "${COMMIT_DLOAD}")
fi

# jax[cuda12] loads its CUDA from the pip nvidia wheels; prepend their lib dirs
# so XLA finds libcusparse/etc even where the base image's /usr/local/cuda is
# incomplete (the documented Kaggle gotcha; harmless on other backends).
NV="$(uv run "${EXTRAS[@]}" python - <<'PY'
import glob, os, sysconfig
libs = sorted(glob.glob(os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "*", "lib")))
print(":".join(libs))
PY
)"
export LD_LIBRARY_PATH="${NV}:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"

# Verify the accelerator: trust CudaDevice, not runtime feel (CLAUDE.md).
uv run "${EXTRAS[@]}" python -c "import jax; print('JAX DEVICES:', jax.devices())"

# shellcheck disable=SC2086
uv run "${EXTRAS[@]}" python -u scripts/egonoise_generate.py \
    --drones ${DRONES} --seeds ${SEEDS} --out "${OUT}" "${COMMIT_ARGS[@]}"

echo "generation done -> ${OUT}"
ls -la "${OUT}" || true
