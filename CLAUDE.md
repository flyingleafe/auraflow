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
- Extras: `viz` (matplotlib), `cfd` (jaxfluids, git-pinned), `data` (dload-ml).
  Base `import auraflow` must always work without any extra.
- float64 everywhere in acoustics; tests enable x64 via `tests/conftest.py`.
- Dataset outputs are managed with dload (R2 bucket configured in
  `dload.toml`, creds in `.env`); GPU jobs via omnirun (`omnirun.toml`).
