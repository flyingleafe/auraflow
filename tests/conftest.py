import jax
import pytest

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
