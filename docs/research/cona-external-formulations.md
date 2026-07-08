# CONA external formulations (recovered from open literature)

Sources verified against open primary/near-primary documents; see source index at bottom.

## 1. FW-H Formulation 1C (convective; Najafi-Yazdi, Brès & Mongeau 2011)

Recovered verbatim from Najafi-Yazdi's McGill dissertation ch. 3 (eqs. 3.15–3.51):
https://www.mcgill.ca/acousticlab/files/acousticlab/dissertation_alireza.pdf

Uniform mean flow U0 along +x1 (rotate frame otherwise), β² = 1 − M0².

Convective wave operator: □̄² = ∂tt − c0²∇² + 2 U0j ∂t∂j + U0i U0j ∂i∂j, acting on H(f)ρ'.

Source tensors:
- Q_j = ρ(u_j + U0j − v_j) + ρ0(v_j − U0j)
- L_ij = ρ u_i (u_j + U0j − v_j) + P_ij,  P_ij = (p−p0)δij − σij
- Impermeable surface (u_n = v_n): Q_j n_j = ρ0 v_n + ρ' U0n;  L_ij n_j = ρ u_i U0n + P_ij n_j.
  (U0 → 0 recovers F1A sources ρ0 v_n and l_i = P_ij n_j.)

Convected Green's fn: G = δ(τ − t + R/c0)/(4π R*), with phase distance R and amplitude
distance R*:
  R  = [−M0 (x1−y1) + R*]/β²;   R* = sqrt((x1−y1)² + β²((x2−y2)² + (x3−y3)²))

Radiation vectors (NOT unit vectors): R̃_i = ∂R/∂x_i, R̃*_i = ∂R*/∂x_i:
  R̃1 = (−M0 + R̃*1)/β², R̃2 = (x2−y2)/R*, R̃3 = (x3−y3)/R*
  R̃*1 = (x1−y1)/R*, R̃*2 = β²(x2−y2)/R*, R̃*3 = β²(x3−y3)/R*

Doppler factor: 1 − M_R with M_R = v_i R̃_i / c0. Emission time τ_e = t − [R/c0]_{τe}.

### Thickness noise (final source-time form, eq. 3.45)

4π p'_T = ∫[ (Q̇_i n_i + Q_i ṅ_i)/(R*(1−M_R)²) ] dη
        + ∫[ (−∂τR*) Q_i n_i /(R*²(1−M_R)²) ] dη
        + ∫[ Q_i n_i (∂τM_R)/(R*(1−M_R)³) ] dη
        − M0 ∫[ (R̃̇1 Q_i n_i + R̃1 Q̇_i n_i + R̃1 Q_i ṅ_i)/(R*(1−M_R)²) ] dη
        + M0 ∫[ (∂τR*) R̃1 Q_i n_i /(R*²(1−M_R)²) ] dη
        − M0 ∫[ (∂τM_R) R̃1 Q_i n_i /(R*(1−M_R)³) ] dη
        − U0 ∫[ R̃*1 Q_i n_i /(R*²(1−M_R)) ] dη            (all at τ_e)

### Loading noise (final source-time form, eq. 3.49)

4π p'_L = (1/c0)∫[ (L̇_ij n_j R̃_i + L_ij ṅ_j R̃_i + L_ij n_j R̃̇_i)/(R*(1−M_R)²) ] dη
        − (1/c0)∫[ (∂τR*) L_ij n_j R̃_i /(R*²(1−M_R)²) ] dη
        + (1/c0)∫[ (∂τM_R) L_ij n_j R̃_i /(R*(1−M_R)³) ] dη
        + ∫[ L_ij n_j R̃*_i /(R*²(1−M_R)) ] dη              (all at τ_e)

### Auxiliary source-time derivatives (only y(η,τ) moves)

∂τR* = −v_i R̃*_i;  ∂τR = −v_i R̃_i = −c0 M_R;  ∂τM_R = (v̇_i R̃_i + v_i R̃̇_i)/c0;
R̃̇_i by differentiating the R̃ formulas (in JAX: jax.jvp/analytic).

### Special cases

- Static source+observer (wind tunnel): R, R*, R̃, n const, M_R = 0:
  4π p'_T = ∫[(1 − M0 R̃1) Q̇_i n_i/R* − U0 R̃*1 Q_i n_i/R*²] dη
  4π p'_L = ∫[(1/c0) L̇_ij n_j R̃_i/R* + L_ij n_j R̃*_i/R*²] dη
- U0 = 0: collapses term-by-term to Farassat 1A (verified in the dissertation) — use this
  as a cross-implementation test against our F1A kernel.

## 2. Deficiency-function recursion (variable freestream; van der Wall & Leishman)

Wagner/Jones: φ(s) = 1 − A1 e^{−b1 s} − A2 e^{−b2 s}; A1=0.165, A2=0.335, b1=0.0455, b2=0.3.
s in SEMICHORDS: Δs_n = (V_n + V_{n−1}) Δt / c.

KEY: march the 3/4-chord DOWNWASH w(s) = V(s) α_qs(s), not α alone.
w_3/4 = V α + ḣ + (c/4) α̇ (quarter-chord pivot; adjust arm otherwise).
Effective downwash w_E = w − X − Y; circulatory lift:
  L_C = ½ ρ V(s) c CLα [w(s) − X(s) − Y(s)]

Mid-point recurrence (2nd-order; Leishman "Algorithm D"; lax.scan carry):
  X_n = X_{n−1} e^{−b1 Δs_n} + A1 Δw_n e^{−b1 Δs_n/2}
  Y_n = Y_{n−1} e^{−b2 Δs_n} + A2 Δw_n e^{−b2 Δs_n/2}
Compressible: exponents b_i β² Δs (β² = 1 − M²). X0 = Y0 = 0.
Continuous equivalent cross-checked vs OpenFAST UA docs. Non-circulatory apparent-mass
terms added separately (CONA Eq. 9). NOTE: discrete recurrences are standard-form
reconstructions (primaries paywalled) — verify vs Wagner step response in tests.

## 3. Lamb–Oseen tip vortex + core growth (Greenwood FRAME thesis, open)

Swirl: V_θ(r) = Γv/(2πr) (1 − e^{−α r²/r_c²}), α = 1.25643 (peak at r = r_c).
(Greenwood actually uses Vatistas n=2: v_θ = Γv r/(2π(r_c^{2n}+r^{2n})^{1/n}); choice barely
matters for noise.)

Circulation from loading: Γ̄0 = 2π C_T / b → Γv = 2π C_T Ω R²/b (b = blade count), with
azimuthal modulation Γ̄ = Γ̄0(γ0 + γ1S sin ψv + γ1C cos ψv), γ0 = 1 nominal.

Core growth (Squire/Bhagwat–Leishman):
  r_c(Δψ) = sqrt(r_c0² + 4 α_L δ ν Δψ/Ω),  δ = 1 + a1 Re_v,  Re_v = Γv/ν
  α_L = 1.25643;  a1 ≈ 2e−4 (literature range 6.5e−5..2e−4 — tunable);
  δ ~ 3–20 for model rotors; r_c0 ≈ 0.05c–0.25c (identify empirically).
Greenwood nondim form: r̄_c(ζ) = sqrt(r̄0² + 4 C_v ζ), C_v absorbs α_L δ ν/(Ω R²).

## 4. Dryden turbulence (MIL-F-8785C low-altitude form)

Spectra: Φu = 2σu²Lu/(πV) · 1/(1+(Luω/V)²);
Φv,w = σ²L/(πV) · (1+3(Lω/V)²)/(1+(Lω/V)²)².
Forming filters: Hu(s) = σu sqrt(2Lu/πV)/(1+(Lu/V)s);
Hv,w(s) = σ sqrt(2L/πV) (1+√3(L/V)s)/(1+(L/V)s)².

Low altitude (h < 1000 ft, h in FEET): Lw = h; Lu = Lv = h/(0.177+0.000823h)^1.2;
σw = 0.1 W20; σu = σv = σw/(0.177+0.000823h)^0.4.
W20 = mean wind at 20 ft: light 15 kt, moderate 30 kt, severe 45 kt.

Discrete (Euler, timestep T, unit Gaussians η):
  u_g^n = (1 − VT/Lu) u_g^{n−1} + sqrt(2VT/Lu) σu η
  (same for v, w with their L, σ) — exact for u; ZOH the 2-state Hv,w for exactness.
Requires T << L/V.

## 5. Trajectory-tracking controller

Zuo 2010 (CONA's controller) is paywalled — structure only recovered. SUBSTITUTE (used by
auraflow, documented as substitute): geometric SE(3) controller, Lee/Leok/McClamroch,
arXiv:1003.2005 (open):

Dynamics (e3 down, thrust f along −Re3): ẋ=v; m v̇ = mg e3 − f Re3; Ṙ = R Ω̂;
J Ω̇ + Ω×JΩ = M. Mixing: f = Σfi; M = [d(f4−f2), d(f1−f3), c_τf(−f1+f2−f3+f4)].

Errors: e_x = x−x_d; e_v = v−ẋ_d; e_R = ½(Rc'R − R'Rc)∨; e_Ω = Ω − R'Rc Ωc.
Control: f = (k_x e_x + k_v e_v + mg e3 − m ẍ_d)·Re3
M = −k_R e_R − k_Ω e_Ω + Ω×JΩ − J(Ω̂ R'Rc Ωc − R'Rc Ω̇c)
Desired attitude: T_c = −k_x e_x − k_v e_v − mg e3 + m ẍ_d; b3c = −T_c/|T_c|;
b2c = b3c×b1d/|b3c×b1d|; Rc = [b2c×b3c, b2c, b3c]; Ω̂c = Rc'Ṙc.
Exponentially stable for Ψ(R(0),Rc(0)) < 1; almost-global attractivity Ψ < 2.
Smooth, singularity-free, JAX-differentiable.

## Source index

- 1C: McGill dissertation (open) + antares FWH docs; original RSPA doi:10.1098/rspa.2010.0172.
- Deficiency: Aeroelasticity.jl Wagner docs; OpenFAST UA theory; arXiv:2104.15122.
- Vortex: Greenwood FRAME thesis (DRUM, open); Ramasamy NASA PDF; Fei NTRS dissertation.
- Dryden: MathWorks continuous/discrete Dryden pages (reproduce MIL specs).
- Controller: arXiv:1003.2005.
Reconstruction flags: §2 discrete recurrences and §4 discrete noise gain are standard-form
reconstructions; verify in tests (Wagner step response; gust variance).
