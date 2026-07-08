# Test reference digest: JASA GP drone-noise paper (data generation)

Source: Lee, Ko, Seshadri, Rauleder — *Bayesian machine learning framework for time-domain
prediction of multirotor vehicle noise*, JASA 159(4):3418–3435 (Apr 2026),
DOI 10.1121/10.0043469.

GP regression predicts full acoustic pressure time series of a quadrotor UAM vehicle;
all train/test data are **synthetic CONA auralization signals**. Our job: reproduce this data
generation with the CONA backend, then regenerate the same cases with CFD+FW-H and compare.

## Data-generation pipeline (what our CONA backend must produce)

1. Trim/flight control: RPM trim along mission profile (front vs aft rotors differ in fwd flight).
2. Inflow: parameterized Beddoes prescribed wake.
3. Blade loads: BET with that inflow.
4. **Tonal noise: time-domain convective FW-H** (pressure time history directly).
5. **Broadband noise: BPM model** as 1/3-octave-band spectrogram (no turbulence-ingestion/BWI).
6. **Auralization: Griffin–Lim** phase reconstruction of the broadband spectrogram → time signal;
   sum with tonal signal → final 44.1 kHz auralization.

NOTE: the CONA paper itself (Ko 2022) is tonal-only; steps 5–6 are the extension used here
(Ko et al. 2023). Our backend must implement both.

## Case setup (reproduce exactly)

- Vehicle: **NASA 1-Pax UAM concept quadrotor** (Silva et al. 2018, AIAA 2018-3847),
  3-bladed rotors, mass 583.85 kg. Hover trim 671 RPM → BPF = 33.55 Hz. BPFs within 10–50 Hz
  across cases. Rotor geometry NOT in this paper — take from Silva 2018 / Ko 2022 (R = 2.0 m
  per CONA paper's UAM case; 662 RPM there vs 671 here).
- Trajectory: level edgewise flight along +x at 30 m altitude, passes over origin at t = 0.5 s;
  V∞ ∈ {1..10} m/s; each simulation 1 s; no wind stated.
- Mics: 256 at ground level, 10 m grid, x ∈ [−150, 160], y ∈ [0, 70] (32×8; y ≥ 0 by symmetry).
- Signals: 44 100 Hz × 1 s per mic per case → 112.9M samples total.

## Down-selection & preprocessing used by the GP (for exact replication of the study)

- Velocities kept: 6–10 m/s; aft region only. Printed x-range "[−30, 90] m" is internally
  inconsistent (aft = x<0; 96 mics = 12×8 columns); most consistent reading: 12 x-columns
  aft-side, e.g. x ∈ [−140, −30], 5 velocities → 480 signals.
- Tonal/broadband split: 4-level db4 DWT; approximation coeffs = tonal (44100 → 1378 pts,
  training targets); detail coeffs = broadband → per-mic GP noise σ_b.
- De-Dopplerization: v_D = −v∞·(r_mic − r_src)/|r_mic − r_src| with r_src = [0,0,30];
  a = c/(c + v_D); resample t ∈ [0,1] → [0,a]; inverse map on predictions.
- Phase alignment: circular shift until first fore-rotor BPF phase matches baseline
  (mic (−30,0), V=1 m/s) within 0.001 rad.
- Split: train V = {6,8,10}, test V = {7,9} (extrapolation demos at 4,5). 12 of 96 mics chosen
  by active learning (max posterior variance, LHS start).

## Evaluation metrics (use for our CONA vs CFD+FW-H comparison too)

- Time-series overlays, FFT magnitude (energy < ~400 Hz).
- DTW distance (Python `dtw` package).
- Loudness: ISO 532-1:2017 Zwicker; Psychoacoustic annoyance: Zwicker & Fastl (SQAT toolbox).
- GP model (context): SVGP/GPyTorch, kernel = Matérn5/2(x)⊙Matérn5/2(y)⊙Fourier(t; 10 BPF
  harmonics per rotor)⊙Matérn5/2(V), 1000 inducing pts.

## Known gaps/ambiguities

1. Rotor geometry/airfoils: from Silva et al. 2018 + Ko et al. 2022/2023.
2. Aft-region x-extent inconsistent as printed.
3. Harmonic count stated as both 20 and 24.
4. Griffin–Lim iterations and FFT/window settings unspecified.
5. Trim controller settings only via CONA references.
