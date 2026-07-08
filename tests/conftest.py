import os

# Force bitwise-deterministic GPU kernels BEFORE jax/XLA initializes. The suite
# asserts bit-exact same-seed reproducibility (Griffin-Lim, flyover generation);
# XLA's default GPU reductions are order-nondeterministic, which fails those
# tests on CUDA while they pass on CPU. Costs some GPU test speed; irrelevant
# for correctness.
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true"
).strip()

import jax  # noqa: E402
import pytest  # noqa: E402

jax.config.update("jax_enable_x64", True)


@pytest.fixture(autouse=True)
def _clear_jax_caches_after_test():
    """Drop XLA compile caches after every test.

    Compiled executables otherwise accumulate across a test file (each distinct
    shape/function combination stays resident) and can exceed the small dev
    box's available RAM. Clearing costs recompiles within a file but keeps the
    per-file peak bounded by the heaviest single test.
    """
    yield
    jax.clear_caches()
