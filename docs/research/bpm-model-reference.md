# BPM airfoil self-noise model — complete implementation reference

Primary source: Brooks, Pope, Marcolini, *Airfoil Self-Noise and Prediction*, NASA RP-1218
(1989), https://ntrs.nasa.gov/citations/19890016302. Equation numbers are the report's.
Every piecewise constant verified against the NTRS scan and cross-checked 1:1 against
OpenFAST `AeroAcoustics.f90` (all match; note OpenFAST *docs page* misprints Eq. 61's
M_max^3 as M_max^2 — report and code use M² M_max³).

## 0. Conventions

c chord [m]; L wetted span of segment [m]; U section inflow speed [m/s]; M = U/c0;
Mc ≈ 0.8 M; Rc = U c/ν; R_δp* = U δp*/ν; α* = effective AoA in DEGREES (from zero-lift
line for cambered sections); δ, δ*, θ at TE [m]; subscripts p/s/0 = pressure/suction/zero-α;
f = 1/3-octave center [Hz]; re = TE→observer distance [m]; h TE bluntness [m]; Ψ TE solid
angle [deg] (0° flat plate, ≈14° NACA0012); SPL dB re 20 µPa; log = log10.
Report's code values: c0 = 340.46 m/s, ν = 1.4529e-5 m²/s. δ*, L, re in consistent meters.
Calibration range: NACA 0012, Rc ≲ 3e6, M ≲ 0.21, α* ≤ 25.2°.

## 1. Boundary-layer correlations (NACA 0012), thicknesses normalized by c

### Zero-α, tripped (Eqs. 2–4)
δ0/c  = 10^[1.892 − 0.9045 log Rc + 0.0596 (log Rc)²]
δ0*/c = 0.0601 Rc^−0.114                          (Rc ≤ 3e5)
        10^[3.411 − 1.5397 log Rc + 0.1059 (log Rc)²]   (Rc > 3e5)
θ0/c  = 0.0723 Rc^−0.1765                         (Rc ≤ 3e5)
        10^[0.5578 − 0.7079 log Rc + 0.0404 (log Rc)²]  (Rc > 3e5)

### Zero-α, untripped (Eqs. 5–7)
δ0/c  = 10^[1.6569 − 0.9045 log Rc + 0.0596 (log Rc)²]
δ0*/c = 10^[3.0187 − 1.5397 log Rc + 0.1059 (log Rc)²]
θ0/c  = 10^[0.2021 − 0.7079 log Rc + 0.0404 (log Rc)²]
Light-trip code variant: tripped δ0* × 0.6; δ0 from untripped Eq. 5.

### Pressure side vs α* (both trips, Eqs. 8–10)
δp/δ0   = 10^[−0.04175 α* + 0.00106 α*²]
δp*/δ0* = 10^[−0.0432 α* + 0.00113 α*²]
θp/θ0   = 10^[−0.04508 α* + 0.000873 α*²]

### Suction side, tripped (Eqs. 11–13)
δs/δ0:   10^{0.0311 α*} (0–5°); 0.3468·10^{0.1231 α*} (5–12.5°); 5.718·10^{0.0258 α*} (12.5–25°)
δs*/δ0*: 10^{0.0679 α*} (0–5°); 0.381·10^{0.1516 α*} (5–12.5°); 14.296·10^{0.0258 α*} (12.5–25°)
θs/θ0:   10^{0.0559 α*} (0–5°); 0.6984·10^{0.0869 α*} (5–12.5°); 4.0846·10^{0.0258 α*} (12.5–25°)

### Suction side, untripped (Eqs. 14–16)
δs/δ0:   10^{0.03114 α*} (0–7.5°); 0.0303·10^{0.2336 α*} (7.5–12.5°); 12·10^{0.0258 α*} (12.5–25°)
δs*/δ0*: 10^{0.0679 α*} (0–7.5°); 0.0162·10^{0.3066 α*} (7.5–12.5°); 52.42·10^{0.0258 α*} (12.5–25°)
θs/θ0:   10^{0.0559 α*} (0–7.5°); 0.0633·10^{0.2157 α*} (7.5–12.5°); 14.977·10^{0.0258 α*} (12.5–25°)

(Light trip uses untripped suction-side scaling on the ×0.6 δ0*. For non-NACA0012 airfoils,
optionally replace correlations with XFOIL BL output — OpenFAST practice.)

## 2. TBL-TE + separation noise

SPL_TOT = 10 log(10^{SPLα/10} + 10^{SPLs/10} + 10^{SPLp/10})   (24)

Attached (α* ≤ (α*)0 = min(γ0, 12.5°)):
SPLp = 10 log(δp* M⁵ L D̄h / re²) + A(Stp/St1) + (K1 − 3) + ΔK1    (25)
SPLs = 10 log(δs* M⁵ L D̄h / re²) + A(Sts/S̄t1) + (K1 − 3)          (26)
SPLα = 10 log(δs* M⁵ L D̄h / re²) + B(Sts/St2) + K2                (27)

Stalled (α* > (α*)0): SPLp = SPLs = −∞;
SPLα = 10 log(δs* M⁵ L D̄ℓ / re²) + A'(Sts/St2) + K2               (30)
where A' = A-curve with Rc → 3Rc in a0, and low-freq directivity D̄ℓ.

Strouhal: Stp = f δp*/U; Sts = f δs*/U; St1 = 0.02 M^−0.6; S̄t1 = (St1+St2)/2;
St2 = St1 · {1 (α*<1.33°); 10^{0.0054(α*−1.33)²} (1.33–12.5°); 4.72 (>12.5°)}.

### A-shape (a = |log(St/St_peak)|)
A_min: sqrt(67.552 − 886.788 a²) − 8.219 (a<0.204); −32.665 a + 3.981 (0.204–0.244);
       −142.795 a³ + 103.656 a² − 57.757 a + 6.006 (a>0.244)
A_max: sqrt(67.552 − 886.788 a²) − 8.219 (a<0.13); −15.901 a + 1.098 (0.13–0.321);
       −4.669 a³ + 3.491 a² − 16.699 a + 1.149 (a>0.321)
a0(Rc): 0.57 (Rc<9.52e4); −9.57e−13 (Rc−8.57e5)² + 1.13 (≤8.57e5); 1.13 (>8.57e5)
AR = (−20 − A_min(a0))/(A_max(a0) − A_min(a0));  A(a) = A_min + AR (A_max − A_min)

### B-shape (b = |log(Sts/St2)|)
B_min: sqrt(16.888 − 886.788 b²) − 4.109 (b<0.13); −83.607 b + 8.138 (0.13–0.145);
       −817.810 b³ + 355.210 b² − 135.024 b + 10.619 (b>0.145)
B_max: sqrt(16.888 − 886.788 b²) − 4.109 (b<0.10); −31.330 b + 1.854 (0.10–0.187);
       −80.541 b³ + 44.174 b² − 39.381 b + 2.344 (b>0.187)
b0(Rc): 0.30 (Rc<9.52e4); −4.48e−13 (Rc−8.57e5)² + 0.56 (≤8.57e5); 0.56 (>8.57e5)
BR = (−20 − B_min(b0))/(B_max(b0) − B_min(b0));  B(b) = B_min + BR (B_max − B_min)

### Amplitudes
K1: −4.31 log Rc + 156.3 (Rc<2.47e5); −9.0 log Rc + 181.6 (2.47e5–8e5); 128.5 (>8e5)
ΔK1: α*[1.43 log(R_δp*) − 5.29] (R_δp* ≤ 5000); 0 otherwise
K2 = K1 + {−1000 (α* < γ0−γ); sqrt(β² − (β/γ)²(α*−γ0)²) + β0 (|α*−γ0| ≤ γ); −12 (α* > γ0+γ)}
γ = 27.094 M + 3.31; γ0 = 23.43 M + 4.651; β = 72.65 M + 10.74; β0 = −34.19 M − 13.82

## 3. LBL-VS noise (untripped/lightly tripped only; user switch)

SPL = 10 log(δp M⁵ L D̄h / re²) + G1(St'/St'_peak) + G2(Rc/(Rc)0) + G3(α*)   (53)
NOTE: uses δp (BL thickness, not δ*). St' = f δp/U.
St'1: 0.18 (Rc≤1.3e5); 0.001756 Rc^0.3931 (1.3e5–4e5); 0.28 (>4e5)
St'_peak = St'1 · 10^{−0.04 α*}
G1(e), e = St'/St'_peak: 39.8 log e − 11.12 (e≤0.5974); 98.409 log e + 2.0 (≤0.8545);
  −5.076 + sqrt(2.484 − 506.25 (log e)²) (≤1.17); −98.409 log e + 2.0 (≤1.674);
  −39.8 log e − 11.12 (>1.674)
G2(d), d = Rc/(Rc)0: 77.852 log d + 15.328 (d≤0.3237); 65.188 log d + 9.125 (≤0.5689);
  −114.052 (log d)² (≤1.7579); −65.188 log d + 9.125 (≤3.0889); −77.852 log d + 15.328 (>)
(Rc)0 = 10^{0.215 α* + 4.978} (α*≤3°); 10^{0.120 α* + 5.263} (α*>3°)
G3 = 171.04 − 3.03 α*

## 4. Tip vortex noise (tip segment only; user switch)

SPL_TIP = 10 log(M² M_max³ ℓ² D̄h / re²) − 30.5 (log St'' + 0.3)² + 126   (61)
St'' = f ℓ / U_max; M_max/M ≈ 1 + 0.036 α'_TIP; U_max = c0 M_max
ℓ/c: rounded tip 0.008 α'_TIP; flat tip: 0.0230 + 0.0169 α' (0–2°); 0.0378 + 0.0095 α' (>2°)
α'_TIP: correct geometric tip AoA by sectional lift-slope ratio near tip (Eq. 66) unless
large-AR untwisted.

## 5. TE bluntness noise (h > 0)

SPL = 10 log(h M^5.5 L D̄h / re²) + G4(h/δ*avg, Ψ) + G5(h/δ*avg, Ψ, St'''/St'''_peak)  (70)
St''' = f h/U; δ*avg = (δp* + δs*)/2
St'''_peak = (0.212 − 0.0045 Ψ)/(1 + 0.235 (h/δ*avg)^−1 − 0.0132 (h/δ*avg)^−2)  (h/δ*avg ≥ 0.2)
           = 0.1 (h/δ*avg) + 0.095 − 0.00243 Ψ                                  (< 0.2)
G4 = 17.5 log(h/δ*avg) + 157.5 − 1.114 Ψ (h/δ*avg ≤ 5); 169.7 − 1.114 Ψ (> 5)
G5 = (G5)_{Ψ=0} + 0.0714 Ψ [(G5)_{Ψ=14} − (G5)_{Ψ=0}]
(G5)_{Ψ=14}, η = log(St'''/St'''_peak):
  m η + k (η<η0); 2.5 sqrt(1 − (η/μ)²) − 2.5 (η0≤η<0);
  sqrt(1.5625 − 1194.99 η²) − 1.25 (0≤η<0.03616); −155.543 η + 4.375 (≥0.03616)
μ: 0.1221 (<0.25); −0.2175 x + 0.1755 (0.25–0.62); −0.0308 x + 0.0596 (0.62–1.15);
   0.0242 (≥1.15)   [x = h/δ*avg]
m: 0 (x≤0.02); 68.724 x − 1.35 (≤0.5); 308.475 x − 121.23 (≤0.62); 224.811 x − 69.35 (≤1.15);
   1583.28 x − 1631.59 (<1.2); 268.344 (≥1.2)
η0 = −sqrt(m²μ⁴/(6.25 + m²μ²));  k = 2.5 sqrt(1 − (η0/μ)²) − 2.5 − m η0
(G5)_{Ψ=0}: same machinery with x → x' = 6.724 x² − 4.019 x + 1.107   (82)
Code caps (report Appendix D + OpenFAST): G5 ← min(G5, 0); G5 ← min(G5, (G5)@x=0.25).
Valid h/δ*avg ≈ 0.2–10.

## 6. Directivity (Appendix B)

TE-local frame: x_e downstream along chordline into wake, y_e spanwise, z_e normal.
Θe = angle from x_e to observer vector; Φe: sin²Φe = z_e²/(y_e²+z_e²). Θe=Φe=90° → D̄h=1.
D̄h = 2 sin²(Θe/2) sin²Φe / [(1 + M cos Θe)(1 + (M − Mc) cos Θe)²]    (B1)
D̄ℓ = sin²Θe sin²Φe / (1 + M cos Θe)⁴                                  (B2)
D̄h for all mechanisms except stalled TBL-TE (uses D̄ℓ). Evaluate per blade segment in its
local TE frame (retarded coordinates if Doppler matters).

## 7. Assembly

Per segment per band: SPL_total = 10 log(Σ_mech 10^{SPL/10}); energy-sum over segments;
A-weight if desired. Mechanism switches: TBL-TE always; separation replaces it above
(α*)0 = min(γ0, 12.5°); LBL-VS untripped only; bluntness if h>0; tip on tip segment only.
Ignore the report code's undocumented `ITRIP==3 → DSTRP×1.48` oddity.
