# Applying BPM per blade section on a rotating rotor

Canonical recipe = Brooks & Burley BARC (AIAA 2001-2210, NTRS 20010050555), reproduced by
ANOPP2 ASNIFM/ABART, PSU-WOPWOP broadband, UCD-QuietFly, OpenFAST.

## Segmentation & section state

- Disk discretized ~14 radial × 18 azimuth stations (Δψ = 20°); each segment = quasi-steady
  isolated airfoil with local TE-fixed frame at quarter-chord mid-span.
- V_tot = V_rotation − V_flight − V_induced (induced velocity from BEMT/inflow model!).
  In TE coords: U = sqrt(V_x² + V_z²); α* = asin(V_z/U), referenced to zero-lift line.
- BL thicknesses from BPM correlations (or XFOIL) at local Re_c, α*.

## Directivity & Doppler (Brooks–Burley variant)

- Observer transformed to segment's **retarded** TE coordinates; angles
  Θ_er = acos(x_er/r_er), Φ_er = acos(y_er/sqrt(y_er²+z_er²)).
- B&B use (1 − M_tot cos ξ_r)^−4 convective amplification for ALL self-noise sources
  (stronger than RP-1218's D̄h denominators; OpenFAST keeps original RP-1218 forms —
  choose one, document; we default to RP-1218 D̄h with retarded geometry).
- Glauert–Prandtl compressibility: G_TBL-TE /= (1 − M²).
- Doppler factor f0/f = 1/(1 − M_tot cos ξ_r); band energy conserved by re-binning with
  bandwidth stretch ∝ f0/f.
- Source nulling (B&B): LBL-VS zeroed unless inflow non-uniformity < 1% U and skew < 15°;
  bluntness only for segment M < 0.5; tip noise uses local flow AoA.

## Energy summation (Eq. 26)

G_BB(f) = Σ_mn N_b (Δψ/360°) (f/f0)_mn [G_Self(f)]_mn
- energy (PSD) sum, never amplitudes; (Δψ/360°) = azimuth dwell fraction;
  (f/f0) = inverse-Doppler dwell correction; one blade's revolution average × blade count;
  then energy-sum over rotors.
- Continuous form (UCD-QuietFly Eq. 3.6): S_pp = (N_B/2π)∫ (ω/ω_d)² S̄_pp dφ.
- Time-varying mode (PSU-WOPWOP/QuietFly): skip azimuth averaging; assign each azimuth
  station's spectrum to observer time τ + R_s/c0 → 1/3-octave spectrogram over the rev
  (this is what CONA/JASA needs for the BPM spectrogram fed to Griffin–Lim).

## Fallback: Pegg TM-80200 whole-rotor empirical model

f_p = −240 log T + 2.448 V_T + 942 [Hz, SI]; SPL_1/3 = 20 log(V_T/c0)³ +
10 log[A_b/r² (cos²θ1 + 0.1)] + S_1/3 + 10 log(C̄_L/0.4) + 130.
S_1/3 table (f/f_p → dB): 1/32:−29.0, 1/16:−24.5, 1/8:−19.5, 1/4:−15.3, 1/2:−11.7,
1:−7.5, 2:−11.5, 4:−12.1, 8:−16.5, 16:−17.0, 32:−21.8, 64:−26.4, 128:−30.0.

## Implementation plan for auraflow.cona.broadband

Per rotor rev: for each (radial, azimuth) segment get (U, α*, Re, M) from the BEMT/inflow
state already computed for tonal noise → BPM 1/3-octave SPL (all mechanisms, α*-switch)
→ retarded TE-frame D̄h/D̄ℓ per observer → Doppler band shift → time-varying assignment to
observer-time frames → [n_frames, n_bands] spectrogram per rotor (independent per rotor,
summed energetically for levels; kept separate for per-rotor Griffin–Lim synthesis).
