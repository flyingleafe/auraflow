"""Tiny end-to-end CFD -> permeable-sphere -> FW-H pulse test.

A small 32^3 Gaussian pressure pulse is run in a quiescent box, sampled on a
permeable sphere, and propagated to two observers on the x-axis. Gates are loose
(the quantitative validation lives in ``scripts/cfd_pulse_validation.py``, meant
for GPU via omnirun):

- the far-field signal is finite and non-trivial;
- the peak arrives later at the farther observer by ~ (r_far - r_near) / c0;
- the peak amplitude decays roughly as 1/r between the two observers.

Skipped cleanly when the ``cfd`` extra (jaxfluids) is not installed. This is the
heaviest test in the suite; it still fits comfortably in memory (peak RSS < 1 GB
at 32^3) but takes ~1 min. Run it on its own:
``uv run --extra cfd python -m pytest tests/cfd/test_pulse_e2e.py``.
"""

import jax.numpy as jnp
import pytest

pytest.importorskip("jaxfluids")

from auraflow.cfd.case import acoustic_box_case  # noqa: E402
from auraflow.cfd.run import propagate_to_observers, run_acoustic_case  # noqa: E402
from auraflow.cfd.sphere import PermeableSphere  # noqa: E402
from auraflow.core.medium import Medium  # noqa: E402


def test_pulse_propagates_and_decays():
    med = Medium()
    c0 = float(med.c0)
    r_sphere = 0.2
    case = acoustic_box_case(
        med,
        half_size=0.5,
        cells=(32, 32, 32),
        cfl=0.4,
        pulse=True,
        pulse_amplitude=100.0,
        pulse_width=0.06,
    )
    sphere = PermeableSphere.fibonacci(64, radius=r_sphere, center=(0.0, 0.0, 0.0))
    hist = run_acoustic_case(case, sphere, n_steps=140, sample_every=2)

    # Surface history is finite and carries a perturbation.
    assert jnp.all(jnp.isfinite(hist.p))
    assert float(jnp.max(jnp.abs(hist.p - med.p0))) > 1.0

    r_near, r_far = 1.0, 2.0
    obs = jnp.array([[r_near, 0.0, 0.0], [r_far, 0.0, 0.0]])
    p_prime, t_obs = propagate_to_observers(hist, sphere, obs, med)

    assert p_prime.shape[0] == 2
    assert jnp.all(jnp.isfinite(p_prime))

    peak = jnp.max(jnp.abs(p_prime), axis=1)
    assert float(peak[0]) > 0.0 and float(peak[1]) > 0.0

    # 1/r amplitude decay between the two observers (loose gate).
    ratio = float(peak[0] / peak[1])
    assert 1.4 < ratio < 3.0, f"amplitude ratio {ratio} not ~2.0 (1/r)"

    # Peak arrives later at the farther observer by ~ (r_far - r_near)/c0.
    idx = jnp.argmax(jnp.abs(p_prime), axis=1)
    t_peak = t_obs[jnp.arange(2), idx]
    dt_arrival = float(t_peak[1] - t_peak[0])
    expected = (r_far - r_near) / c0
    assert dt_arrival == pytest.approx(expected, rel=0.30), (
        f"arrival delay {dt_arrival} vs expected {expected}"
    )
