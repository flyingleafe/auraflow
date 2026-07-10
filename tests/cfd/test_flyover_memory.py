"""Memory-flatness guard for the lazy per-chunk flyover tiling (GH #3).

The lazy path (``tile_surface_history(..., lazy=True)`` +
``quadrotor_surface_flyover``) must never materialize a full ``[S, T]`` surface
history: only ``[chunk, T]`` slices are ever built, and the adaptive
``panel_chunk`` cap keeps ``chunk * T`` bounded, so the host peak RSS stays
~flat as the flyover DURATION (hence ``T``) grows. We verify this at reduced
scale by running the synthesis in a fresh subprocess per duration (so each
child reports its own clean ``ru_maxrss`` high-water mark, uncontaminated by
the parent's or a sibling's XLA caches) and checking the peak grows far less
than the 5x duration ratio.

Kept in its own file so the parent test process never imports JAX / initialises
a device backend (the children do), keeping parent+child memory under the
dev-box cgroup cap.
"""

import subprocess
import sys
import textwrap

# Runs one lazy (or eager) synthesis and prints its peak RSS in KB on the last
# stdout line. Small scale (<= 1280 panels, 3 mics) so it fits the RAM cap.
_CHILD = textwrap.dedent(
    """
    import resource, sys
    import numpy as np
    import jax
    jax.config.update("jax_enable_x64", True)
    from auraflow.body.mesh import TriMesh
    from auraflow.cfd.flyover import tile_surface_history, quadrotor_surface_flyover
    from auraflow.core.medium import Medium

    duration = float(sys.argv[1])
    lazy = sys.argv[2] == "lazy"

    OMEGA, N_BLADES = 70.3, 3
    T_BP = 2.0 * np.pi / (OMEGA * N_BLADES)
    mesh = TriMesh.sphere(radius=0.2, subdivisions=3)  # 1280 panels
    surf = {
        "points": np.asarray(mesh.centroids(), dtype=np.float64),
        "normals": np.asarray(mesh.normals(), dtype=np.float64),
        "area": np.asarray(mesh.areas(), dtype=np.float64),
    }
    n_s = surf["points"].shape[0]
    dtau = T_BP / 20.0
    n_in = 55  # ~2.7 blade periods of raw data
    tau = np.arange(n_in) * dtau
    f0 = OMEGA * N_BLADES / (2 * np.pi)
    s = np.sin(2 * np.pi * f0 * tau[None, :] + 0.1 * np.arange(n_s)[:, None])
    surf_u = surf["normals"][:, None, :] * (1.5 * s)[:, :, None]
    raw = {"tau": tau, "rho": 0.01 * s, "u": surf_u, "p": 2.0 * s}

    layout = (np.array([[0.0, 0.0, 0.0]]), np.array([1.0]))
    obs = np.array([[20.0, 3.0, -5.0], [-15.0, -2.0, 4.0], [0.0, 0.0, -8.0]])

    tiled = tile_surface_history(raw, OMEGA, N_BLADES, duration=duration, lazy=lazy)
    p, _ = quadrotor_surface_flyover(
        surf, tiled, layout, speed=9.0, altitude=25.0, t_pass=duration / 2,
        observers=obs, medium=Medium(), phase_offsets=[0.0],
        panel_chunk=128, obs_chunk=1,
    )
    p.block_until_ready()
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(peak_kb)
    """
)


def _peak_kb(duration: float, mode: str = "lazy") -> int:
    out = subprocess.run(
        [sys.executable, "-c", _CHILD, str(duration), mode],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.strip().splitlines()[-1])


def test_peak_rss_flat_across_duration():
    """5x the flyover duration must not 5x (or even 1.5x) the host peak RSS."""
    short = _peak_kb(0.06, "lazy")
    long = _peak_kb(0.30, "lazy")  # 5x the duration -> 5x T
    # Lazy path: per-chunk [c, T] with chunk*T held ~constant by the adaptive cap,
    # so the peak is dominated by the (duration-independent) JAX baseline. A full
    # [S, T] materialization would instead scale ~linearly with duration.
    assert long < 1.5 * short, (
        f"peak RSS grew too much with duration: {short} KB -> {long} KB "
        f"(ratio {long / short:.2f}); the lazy path may be materializing [S, T]"
    )
