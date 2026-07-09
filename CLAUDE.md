# AuraFlow — agent instructions

Differentiable aeroacoustics in JAX. Binding design decisions live in
`docs/architecture.md`; per-reference physics digests in `docs/research/*.md`.
Read those before touching physics code — don't re-derive from the PDFs.

## RAM safety on this dev box (CRITICAL)

The box has 3.7 GiB RAM, no swap, and typically only ~1.3 GiB available
(docker + open-webui hold the rest). JAX XLA compile caches accumulate across
a process (each stage/shape adds 100–300 MB); an unbounded pytest or pipeline
run WILL OOM-kill the user's tmux. Rules:

- NEVER run the whole test suite in one process. Run pytest one file at a
  time. Never use pytest-xdist locally.
- Wrap every JAX-running command in a cgroup cap so a mistake kills only the
  command, never the box:

  ```sh
  systemd-run --user --scope -q -p MemoryMax=1100M -p MemorySwapMax=0 -- \
      uv run pytest tests/<dir>/<file>.py -q
  ```

- `tests/conftest.py` clears XLA caches after every test — keep that fixture.
- `auraflow.datasets.jasa.generate_flyover(..., low_memory=True)` clears
  caches at stage boundaries (peak ~750 MB instead of >1.1 GB). Local smoke
  runs must use it; GPU runs should not.
- Anything heavier (full 44.1 kHz generation, CFD > 32³) runs on GPU via
  omnirun, never locally.
- At most ONE JAX-running subagent at a time; a subagent process itself costs
  ~400 MB on top of its python.

## Tooling

- `uv run pytest ...` / `uv run ruff check src tests scripts` /
  `uv run basedpyright src/auraflow/<module>` (run checks one at a time).
- Extras: `viz` (matplotlib), `viz-live` (websockets), `cfd` (jaxfluids, git-pinned),
  `mesh` (trimesh), `data` (dload-ml), `gpu` (jax[cuda12] + explicit nvidia wheels).
  Base `import auraflow` must always work without any extra.
- **Kaggle GPU jobs**: jax silently falls back to CPU unless the pip nvidia lib dirs
  are PREPENDED to `LD_LIBRARY_PATH` (Kaggle's `/usr/local/cuda/lib64` lacks
  libcusparse and shadows the wheels; clearing the variable loses libcuda). Wrap the
  job command:
  `bash -c 'NV=$(uv run --extra gpu python -c "import glob,os,sysconfig;print(\":\".join(sorted(glob.glob(os.path.join(sysconfig.get_paths()[\"purelib\"],\"nvidia\",\"*\",\"lib\")))))"); export LD_LIBRARY_PATH="$NV:/usr/local/nvidia/lib64"; uv run --extra gpu ... '`
  and verify with a printed `jax.devices()` — trust only `CudaDevice`, never runtime feel.
- float64 everywhere in acoustics; tests enable x64 via `tests/conftest.py`.
- Dataset outputs are managed with dload (R2 bucket configured in
  `dload.toml`, creds in `.env`); GPU jobs via omnirun (`omnirun.toml`).
