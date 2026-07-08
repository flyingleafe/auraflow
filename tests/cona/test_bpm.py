"""BPM isolated-airfoil self-noise: RP-1218 anchor gates + mechanism sanity."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from auraflow.cona.bpm import (
    a_shape,
    b_shape,
    boundary_layer_thickness,
    bpm_third_octave,
    directivity_high,
    directivity_low,
    k1_amplitude,
    st1_peak,
    tbl_te_noise,
)
from auraflow.core.medium import Medium
from auraflow.signal.spectra import third_octave_bands


def _bands():
    centers, _ = third_octave_bands(100.0, 20000.0)
    return centers


class TestShapeFunctions:
    def test_a_peak_zero(self):
        # A(0) ~= 0 at St = St_peak (a = 0); digest's rounded constants give
        # sqrt(67.552) - 8.219 ~= 2e-6, not exactly 0.
        assert abs(float(a_shape(1.0, 1.0e6))) < 1e-3

    def test_a_reaches_minus20_at_a0(self):
        # By construction A(a0) = -20; a0 = 1.13 for Rc > 8.57e5.
        val = float(a_shape(10.0**1.13, 3.0e6))
        assert abs(val + 20.0) < 1e-6

    def test_b_peak_zero(self):
        # B(0) = sqrt(16.888) - 4.109 ~= 5e-4 (digest's rounded constants).
        assert abs(float(b_shape(1.0, 1.0e6))) < 1e-3

    def test_b_reaches_minus20_at_b0(self):
        # b0 = 0.56 for Rc > 8.57e5.
        val = float(b_shape(10.0**0.56, 3.0e6))
        assert abs(val + 20.0) < 1e-6

    def test_a_is_even_symmetric(self):
        # a = |log10(ratio)| -> A(r) == A(1/r).
        assert np.isclose(float(a_shape(2.0, 1e6)), float(a_shape(0.5, 1e6)))


class TestAmplitudes:
    def test_k1_high_re_plateau(self):
        assert abs(float(k1_amplitude(1.0e6)) - 128.5) < 1e-9
        assert abs(float(k1_amplitude(2.0e6)) - 128.5) < 1e-9

    def test_k1_low_re_branch(self):
        assert abs(float(k1_amplitude(1.0e5)) - (-4.31 * 5.0 + 156.3)) < 1e-6

    def test_k1_mid_branch(self):
        rc = 5.0e5
        assert abs(float(k1_amplitude(rc)) - (-9.0 * np.log10(rc) + 181.6)) < 1e-6

    def test_st1_scaling(self):
        m = 0.2
        assert np.isclose(float(st1_peak(m)), 0.02 * m**-0.6)


class TestBoundaryLayer:
    def test_zero_alpha_symmetric(self):
        bl = boundary_layer_thickness(1.5e6, 0.0, 0.3048, tripped=True)
        assert np.isclose(float(bl.dstar_p), float(bl.dstar_s))
        assert np.isclose(float(bl.delta_p), float(bl.delta_s))

    def test_tripped_dstar_anchor(self):
        # Rc = 1.5e6, tripped, alpha=0: dstar0/c = 10^[3.411-1.5397 L+0.1059 L^2].
        rc, c = 1.5e6, 0.3048
        bl = boundary_layer_thickness(rc, 0.0, c, tripped=True)
        logr = np.log10(rc)
        expect = 10.0 ** (3.411 - 1.5397 * logr + 0.1059 * logr**2) * c
        assert np.isclose(float(bl.dstar_p), expect, rtol=1e-6)

    def test_alpha_grows_suction_shrinks_pressure(self):
        bl0 = boundary_layer_thickness(1.5e6, 0.0, 0.3, tripped=True)
        bl8 = boundary_layer_thickness(1.5e6, 8.0, 0.3, tripped=True)
        assert float(bl8.dstar_s) > float(bl0.dstar_s)
        assert float(bl8.dstar_p) < float(bl0.dstar_p)


class TestDirectivity:
    def test_dh_normalized_at_90_90(self):
        d = float(directivity_high(np.pi / 2, np.pi / 2, 0.2, 0.16))
        assert abs(d - 1.0) < 1e-9

    def test_dl_null_in_te_plane(self):
        # D_bar_l ~ sin^2(Theta_e): null at Theta_e = 0 (streamwise / TE plane).
        assert float(directivity_low(0.0, np.pi / 2, 0.2)) < 1e-12
        assert float(directivity_low(np.pi / 2, np.pi / 2, 0.2)) > 0.5

    def test_dh_zero_at_forward_arc(self):
        # 2 sin^2(Theta/2) -> 0 as Theta -> 0.
        assert float(directivity_high(0.0, np.pi / 2, 0.2, 0.16)) < 1e-12


class TestTBLTESpectrum:
    def test_canonical_peak_band(self):
        # U=71.3 m/s, c=0.3048 m, tripped, alpha=0 -> peak near 1-1.6 kHz.
        med = Medium(c0=340.46, nu=1.4529e-5)
        U, c = 71.3, 0.3048
        m = U / med.c0
        rc = U * c / med.nu
        bands = _bands()
        bl = boundary_layer_thickness(rc, 0.0, c, tripped=True)
        spl = tbl_te_noise(
            bands,
            U,
            m,
            rc,
            bl.dstar_p,
            bl.dstar_s,
            1.0,
            1.0,
            directivity_high(np.pi / 2, np.pi / 2, m, 0.8 * m),
            directivity_low(np.pi / 2, np.pi / 2, m),
            0.0,
            med.nu,
        )
        peak_f = float(bands[int(np.argmax(np.asarray(spl)))])
        assert 800.0 <= peak_f <= 2000.0

    def test_spl_increases_with_speed(self):
        med = Medium()
        c = 0.1
        bands = _bands()

        def oaspl(U):
            m = U / med.c0
            rc = U * c / med.nu
            bl = boundary_layer_thickness(rc, 3.0, c, tripped=True)
            spl = tbl_te_noise(
                bands,
                U,
                m,
                rc,
                bl.dstar_p,
                bl.dstar_s,
                1.0,
                1.0,
                directivity_high(np.pi / 2, np.pi / 2, m, 0.8 * m),
                directivity_low(np.pi / 2, np.pi / 2, m),
                3.0,
                med.nu,
            )
            return 10.0 * np.log10(np.sum(10.0 ** (np.asarray(spl) / 10.0)))

        # ~5th power scaling: 50 dB/decade nominal; allow a loose window.
        rise = oaspl(100.0) - oaspl(10.0)
        assert 35.0 < rise < 65.0


class TestAssembly:
    def test_total_is_energy_sum(self):
        med = Medium()
        bands = _bands()
        out = bpm_third_octave(
            bands,
            60.0,
            0.1,
            0.05,
            60.0 * 0.1 / med.nu,
            60.0 / med.c0,
            med,
            alpha_deg=4.0,
            include_lbl_vs=True,
            include_bluntness=True,
            h=1e-3,
            tripped=False,
        )
        recon = 10.0 * np.log10(
            10.0 ** (np.asarray(out.tbl_te) / 10.0)
            + 10.0 ** (np.asarray(out.lbl_vs) / 10.0)
            + 10.0 ** (np.asarray(out.tip) / 10.0)
            + 10.0 ** (np.asarray(out.bluntness) / 10.0)
        )
        assert np.allclose(recon, np.asarray(out.total), atol=1e-6)

    def test_bluntness_vanishes_as_h_to_zero(self):
        med = Medium()
        bands = _bands()
        args: dict[str, Any] = dict(alpha_deg=2.0, include_tbl_te=False, include_bluntness=True)
        big = bpm_third_octave(
            bands, 60.0, 0.1, 0.05, 60.0 * 0.1 / med.nu, 60.0 / med.c0, med, h=2e-3, **args
        )
        tiny = bpm_third_octave(
            bands, 60.0, 0.1, 0.05, 60.0 * 0.1 / med.nu, 60.0 / med.c0, med, h=0.0, **args
        )
        assert np.max(np.asarray(big.bluntness)) > 0.0  # audible
        assert np.max(np.asarray(tiny.bluntness)) < -200.0  # floored

    def test_lbl_adds_energy(self):
        med = Medium()
        bands = _bands()
        base: dict[str, Any] = dict(alpha_deg=1.0, tripped=False)
        no_lbl = bpm_third_octave(
            bands,
            30.0,
            0.05,
            0.02,
            30.0 * 0.05 / med.nu,
            30.0 / med.c0,
            med,
            include_lbl_vs=False,
            **base,
        )
        with_lbl = bpm_third_octave(
            bands,
            30.0,
            0.05,
            0.02,
            30.0 * 0.05 / med.nu,
            30.0 / med.c0,
            med,
            include_lbl_vs=True,
            **base,
        )
        assert np.max(np.asarray(with_lbl.total)) >= np.max(np.asarray(no_lbl.total)) - 1e-6
        assert np.max(np.asarray(with_lbl.lbl_vs)) > 0.0


class TestVmapAndGrad:
    def test_vmap_over_sections(self):
        med = Medium()
        bands = _bands()
        n = 5
        U = jnp.linspace(30.0, 90.0, n)
        c = jnp.linspace(0.05, 0.02, n)

        def one(u, cc):
            return bpm_third_octave(
                bands, u, cc, cc, u * cc / med.nu, u / med.c0, med, alpha_deg=3.0
            ).total

        out = jax.vmap(one)(U, c)
        assert out.shape == (n, bands.shape[0])
        assert np.all(np.isfinite(np.asarray(out)))

    def test_grad_through_bpm(self):
        med = Medium()
        bands = _bands()

        def loss(chord):
            out = bpm_third_octave(
                bands,
                60.0,
                chord,
                chord,
                60.0 * chord / med.nu,
                60.0 / med.c0,
                med,
                alpha_deg=5.0,
                include_lbl_vs=True,
                include_bluntness=True,
                h=1e-3,
                include_tip=True,
                tripped=False,
            )
            return jnp.sum(out.total)

        g = jax.grad(loss)(0.08)
        assert np.isfinite(float(g))
