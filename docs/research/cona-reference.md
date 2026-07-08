# Reference #3 digest: CONA framework

Source: Ko, Jeong, Cho, Lee (Seoul National Univ.) — *Real-Time Prediction Framework for
Frequency-Modulated Multirotor Noise*, Physics of Fluids, DOI 10.1063/5.0081103 (2022).
CONA = COmprehensive multirotor Noise Assessment framework.

Caveats: CONA is **tonal-noise-only** — no broadband (BPM), no atmospheric absorption, no
ground reflection. Doppler/interference emerge naturally from the time-domain moving-source
formulation. Several sub-models are deferred to references (see "external formulations" below).

## Pipeline (5 modules)

1. **Flight control** — 6-DOF quadrotor sim (Davoudi et al.), backstepping trajectory-tracking
   controller (Zuo 2010), Dryden turbulence model for wind gusts. Output: per-rotor RPM time
   histories + vehicle states (position, velocity, Euler angles, body rates).
2. **Aerodynamics** — HBEM (BEMT + linear inflow) with airfoil lookup tables
   (Cl, Cd, Cm)(α, M, Re); optional Beddoes prescribed wake; trim coupling with flight control.
3. **Time reconstruction** — resample control-rate outputs onto 1°-azimuth time steps (based on
   mean RPM) by linear interpolation; apply unsteady aerodynamic corrections here.
4. **Noise prediction** — time-domain tonal noise via convective FW-H **Formulation 1C**
   (Najafi-Yazdi, Brès & Mongeau 2011), source-time-dominant algorithm (Casalino 2003;
   Brès et al. 2004). Output: pressure time series per observer (auralization-ready).
5. **TFA** — IMSST time-frequency analysis of the synthesized signal (Yu, JSV 2021).

## Aerodynamics

### HBEM + linear inflow (Pitt–Peters static)

λ(r,ψ) = λ0 (1 + kx r cos ψ + ky r sin ψ),  kx = (15π/32) tan(χ/2), ky = 0
χ = atan(V∞ cos αp / (V∞ sin αp + λi)),  μ = V∞ cos αp / Vtip

Drees' model explicitly rejected for fixed-pitch rotors. Fuselage drag from empirical copter
drag coefficient vs pitch (Russell et al. 2016). H-force and rotor moments from HBEM.

### Beddoes prescribed wake (parameterized)

Tip vortex trajectory vs wake age Δψv (normalized by R):

x_v = r_v cos ψv + μx Δψv,  y_v = r_v sin ψv
z_v = −μz Δψv + piecewise (three branches by x_v / cos ψv sign — see paper Eq. 3):
  branch 1 (x_v < −r_v cos ψv): −λi [w0 − ws μx y_v + wc χ (cos ψv + μx Δψv/(2 r_v) − |y_v³|)] Δψv
  branch 2 (cos ψv > 0):        −2λi (w0 − ws μx y_v − wc χ |y_v³|) Δψv
  branch 3 (otherwise):         −2λi (w0 − ws μx y_v − wc χ |y_v³|)/μx

Original params (w0, ws, wc) = (1.0, 0.0, 0.5). Tuned: eHANG fwd flight (0.4, −1.0, 0.7);
DJI F450 quad (0.4, −1.0, 0.77); NASA UAM quad (0.9, −1.5, 0.8).
Circulation: Lamb–Oseen vortex, core radius grows with wake age (Greenwood 2011 thesis).
Induced velocities via Biot–Savart at rotor collocation points.
Known limitation: overestimates self-wake unsteady loading noise for close-spaced UAV quads.

### Unsteady corrections (variable RPM), applied after time reconstruction

L = L_NC + L_C, with ḣ = α̇ = 0 simplifications (pure variable-velocity):
L_NC ≈ ½ ρ0 CLα c² v̇(τ) α(τ)   (apparent mass)
L_C  ≈ ½ ρ0 CLα c v(τ) {v(s)α − X(s) − Y(s)}   (Duhamel w/ Wagner function)
Wagner via Jones approx: φ(s) = 1 − A1 e^{−b1 s} − A2 e^{−b2 s},
A1=0.165, b1=0.0455, A2=0.335, b2=0.3 (paper prints positive exponents — typo, use negative).
X, Y = recursive deficiency functions (van der Wall & Leishman 1992) — lax.scan-friendly.

## Acoustics

- Convective FW-H **Formulation 1C**, impermeable, thickness + loading only (no quadrupole).
  Convective-stress-tensor term in loading source not needed for impermeable surfaces.
  Uniform mean flow (wind) handled exactly; Doppler implicit.
- **Thickness noise: full blade surface mesh** (each airfoil-surface panel a monopole source) —
  compact assumption deliberately NOT used. Requires actual airfoil section shapes.
- **Loading noise: chordwise-compact** — sectional lift/drag as point dipoles at the section
  pitch axis.
- Source-time-dominant algorithm: loop over emission times, project to arrival times,
  interpolate onto uniform observer grid (scatter-add — differentiable in JAX).
- Multi-rotor: independent RPM histories per rotor (trim: rear rotors ~160 RPM faster at
  10 m/s; gusts ±80 RPM); coherent time-domain summation — interference/AM emerges.

## Validation cases (reuse as our test cases)

1. DJI 9450 blade, R=0.121 m, hover 5400 RPM; polars from CFD; mics at 10R, elevations every
   22.5° from 45° below disk plane; matches NASA SALT experiment & OVERFLOW2+PSU-WOPWOP
   to 2nd harmonic.
2. eHANG Ghost 3.0 blade, R=0.103 m, fwd flight 15 m/s, pitch 15°, Selig polars; 16 mics on
   1.5 m circle 1.25 m below disk; first-BPF error < 3 dB.
3. DJI F450 quad (450 mm diagonal spacing), 5250 RPM, μ=0.15/0.2; OASPL directivity vs MultiPA.
4. DJI Phantom 2 flyover: 6.10 m/s, 5.49 m altitude, 3.66 m/s wind, ground mic — spectrogram
   FM/Doppler reproduced.
5. NASA UAM quad eVTOL: R=2.0 m, 662 RPM, 21.34 m/s; observers at 10R.

## External formulations needed for exact reimplementation

- Formulation 1C integrals: Najafi-Yazdi, Brès & Mongeau, Proc. R. Soc. A 467:144–165 (2011).
- Source-time-dominant algorithm details: Casalino JSV 261:583–612 (2003); Brès et al.
  JSV 275:719–738 (2004).
- Vortex circulation/core growth: Greenwood, FRAME thesis, U. Maryland (2011).
- Deficiency function recursions: van der Wall & Leishman, 18th ERF (1992).
- HBEM details: Davoudi & Duraisamy AIAA 2019-2823; flight sim Davoudi et al. AIAA J. 58 (2020);
  controller Zuo, IET CTA 4:2343–2355 (2010).
- IMSST: Yu, JSV 492:115813 (2021).

## Differentiability notes for JAX port

RPM/state trajectories → HBEM (closed form) → Beddoes wake (closed-form geometry + Biot–Savart)
→ unsteady correction (recursive linear filters, lax.scan) → Formulation 1C with source-time
projection (scatter-add w/ interpolation weights). Non-smooth pieces: polar table lookups
(differentiable interpolation) and Beddoes piecewise branches (jnp.where).
