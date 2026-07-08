"""BPM rotor application: spectrogram shape/finiteness, energy consistency, Doppler."""

import jax.numpy as jnp
import numpy as np

from auraflow.bemt.unsteady import SectionState
from auraflow.cona.broadband import (
    doppler_rebin,
    rotor_broadband_levels,
    rotor_broadband_spectrogram,
)
from auraflow.core.medium import Medium
from auraflow.signal.spectra import P_REF, third_octave_bands


def _hover_state(n_b=2, n_s=6, n_t=40, omega=800.0, radius=0.15):
    """A tiny analytic hovering-rotor SectionState (rigid rotation, no flight)."""
    t = jnp.linspace(0.0, 2.0 * np.pi / omega, n_t)  # ~one revolution
    r = jnp.linspace(0.03, radius, n_s)
    dr = jnp.full(n_s, float((radius - 0.03) / (n_s - 1)))
    chord = jnp.linspace(0.02, 0.014, n_s)
    offsets = 2.0 * np.pi * jnp.arange(n_b) / n_b
    psi = omega * t[None, :] + offsets[:, None]  # [B,T]

    cospsi = jnp.cos(psi)[:, None, :]
    sinpsi = jnp.sin(psi)[:, None, :]
    r_b = r[None, :, None]
    pos = jnp.stack(
        [r_b * cospsi, r_b * sinpsi, jnp.zeros_like(r_b * cospsi)], axis=-1
    )  # [B,S,T,3], hub at origin, disc in z=0
    vel = jnp.stack(
        [-omega * r_b * sinpsi, omega * r_b * cospsi, jnp.zeros_like(r_b * cospsi)], axis=-1
    )
    acc = jnp.stack(
        [-(omega**2) * r_b * cospsi, -(omega**2) * r_b * sinpsi, jnp.zeros_like(r_b * cospsi)],
        axis=-1,
    )
    w = jnp.broadcast_to(omega * r_b, (n_b, n_s, n_t))
    med = Medium()
    mach = w / med.c0
    reyn = w * chord[None, :, None] / med.nu
    alpha = jnp.full((n_b, n_s, n_t), np.radians(4.0))
    zeros = jnp.zeros((n_b, n_s, n_t))
    return t, SectionState(
        r=r,
        dr=dr,
        chord=chord,
        psi=psi,
        phi=alpha,
        alpha=alpha,
        w=w,
        reynolds=reyn,
        mach=mach,
        cl=zeros,
        cd=zeros,
        v_axial=zeros,
        v_swirl=zeros,
        lift_per_span=zeros,
        drag_per_span=zeros,
        position=pos,
        velocity=vel,
        acceleration=acc,
        force_on_fluid=jnp.zeros((n_b, n_s, n_t, 3)),
    )


class TestDopplerRebin:
    def test_static_conserves_energy(self):
        rng = np.random.default_rng(0)
        centers, _ = third_octave_bands(100.0, 10000.0)
        nb = centers.shape[0]
        msq = jnp.asarray(rng.uniform(0.1, 1.0, (4, nb)))
        out = doppler_rebin(msq, jnp.ones(4))  # D=1 -> identity
        assert np.allclose(np.asarray(out), np.asarray(msq), atol=1e-12)

    def test_shift_conserves_total(self):
        centers, _ = third_octave_bands(100.0, 10000.0)
        nb = centers.shape[0]
        # Interior-supported spectrum (tapers to ~0 at the grid edges) so a
        # small shift does not truncate energy off the band grid.
        k = np.arange(nb)
        msq = jnp.asarray(np.exp(-0.5 * ((k - nb / 2) / 3.0) ** 2))[None, :]
        for d in (0.9, 1.05, 1.2, 0.75):
            out = doppler_rebin(msq, jnp.array([d]))
            # Interior energy conserved to <0.1 dB (edge truncation aside; the
            # spectrum has support away from edges here).
            total_in = float(jnp.sum(msq))
            total_out = float(jnp.sum(out))
            assert abs(10 * np.log10(total_out / total_in)) < 0.1


class TestSpectrogram:
    def test_shape_and_finite(self):
        t, state = _hover_state()
        med = Medium()
        obs = jnp.array([[1.0, 0.0, -1.5], [0.0, 1.2, -1.0]])
        centers, spec, ftimes = rotor_broadband_spectrogram(
            state,
            obs,
            med,
            t,
            fmin=200.0,
            fmax=8000.0,
            n_frames=8,
            include_tip=True,
            include_bluntness=True,
            h=5e-4,
        )
        assert spec.shape == (2, 8, centers.shape[0])
        assert ftimes.shape == (8,)
        assert np.all(np.isfinite(np.asarray(spec)))
        # Some audible content.
        assert np.max(np.asarray(spec)) > 0.0

    def test_timevarying_average_matches_revaveraged(self):
        t, state = _hover_state(n_t=60)
        med = Medium()
        obs = jnp.array([[1.0, 0.5, -1.2]])
        centers, spec, _ = rotor_broadband_spectrogram(
            state,
            obs,
            med,
            t,
            fmin=200.0,
            fmax=8000.0,
            n_frames=20,
        )
        _, levels = rotor_broadband_levels(state, obs, med, t, fmin=200.0, fmax=8000.0)
        # Energy-average the spectrogram over frames.
        frame_msq = 10.0 ** (np.asarray(spec[0]) / 10.0) * P_REF**2
        avg_msq = np.mean(frame_msq, axis=0)
        avg_db = 10.0 * np.log10(avg_msq / P_REF**2)
        lvl = np.asarray(levels[0])
        # Compare occupied bands within a couple dB.
        occ = lvl > (lvl.max() - 30.0)
        diff = np.abs(avg_db[occ] - lvl[occ])
        assert np.median(diff) < 2.0


class TestGradient:
    def test_levels_finite_grad(self):
        import jax

        t, state = _hover_state(n_t=20)
        med = Medium()
        obs = jnp.array([[1.0, 0.0, -1.0]])

        def loss(scale):
            s = SectionState(
                r=state.r,
                dr=state.dr,
                chord=state.chord * scale,
                psi=state.psi,
                phi=state.phi,
                alpha=state.alpha,
                w=state.w,
                reynolds=state.reynolds,
                mach=state.mach,
                cl=state.cl,
                cd=state.cd,
                v_axial=state.v_axial,
                v_swirl=state.v_swirl,
                lift_per_span=state.lift_per_span,
                drag_per_span=state.drag_per_span,
                position=state.position,
                velocity=state.velocity,
                acceleration=state.acceleration,
                force_on_fluid=state.force_on_fluid,
            )
            _, lv = rotor_broadband_levels(s, obs, med, t, fmin=200.0, fmax=8000.0)
            return jnp.sum(lv)

        g = jax.grad(loss)(1.0)
        assert np.isfinite(float(g))
