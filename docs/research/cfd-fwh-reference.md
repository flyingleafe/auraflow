# Reference #2 digest: CFD + permeable-surface FW-H (OpenCFD-FWH)

Source: Zhang, Yu, Liu, Li — *An MPI-OpenMP mixing parallel open source FW-H code for
aeroacoustics calculation*, arXiv:2312.16263 (Dec 2023). Code: https://github.com/Z-K-L/OpenCFD-FWH
(Fortran 90). Companion CFD solver: OpenCFD-EC (compressible FV, multiblock structured).

Scope caveat: the paper's *implemented* formulation targets wind-tunnel configurations —
source geometry stationary, uniform inflow at Mach M0 with AoA, fixed observers. It states the
general moving-surface permeable Farassat 1A equations but then simplifies away the surface-motion
terms. For drone rotors AuraFlow must implement the full moving-surface form (or keep the
stationary-sphere simplification, which is exactly our use case: a *static* permeable sphere
around the drone with the blades moving inside it — then the surface is stationary and the
simplified algorithm applies with M0 = 0 or the convective form for forward flight).

## CFD setup used for validation (30P30N high-lift airfoil)

- Compressible finite-volume, multiblock structured; IDDES-SA (init from RANS-SA).
- M∞ = 0.17, AoA 5.5°, Re = 1.71e6; domain 50c forward/vertical, 75c downstream; span c/9 periodic.
- >43M cells, y+ < 1; Roe flux + 3rd-order MUSCL; dual-time LU-SGS, Δt = 2e-7 s, 5 subiters.
- Non-reflecting treatment: sponge layer at all outer boundaries with viscosity ×100.
- ~0.1 s physical time; first transient discarded, 0.06534 s retained for acoustics.

## Permeable FW-H equation

□²[c²(ρ−ρ0)] = ∂t[Qn δ(f)] − ∂xi[Li δ(f)] + ∂²xi xj[Tij H(f)]

- Qn = [ρ0 vi + ρ(ui − vi)] n̂i
- Li = [Pij + ρ ui (uj − vj)] n̂j,  Pij = (p−p0)δij (viscous stress neglected)
- Tij = ρ ui uj + Pij − c²(ρ−ρ0)δij — volume quadrupole integral DROPPED entirely
  (permeable surface enclosing the nonlinear region already carries interior quadrupoles).

## General moving-surface Farassat 1A (permeable), quadrupole dropped

4π p'_T(x,t) = ∫[ (Q̇i n̂i + Qi n̂̇i) / (r(1−Mr)²) ]_ret dS
             + ∫[ Qn (r Ṁr + c0(Mr − M²)) / (r²(1−Mr)³) ]_ret dS

4π p'_L(x,t) = (1/c0)∫[ L̇r / (r(1−Mr)²) ]_ret dS
             + ∫[ (Lr − LM) / (r²(1−Mr)²) ]_ret dS
             + (1/c0)∫[ Lr (r Ṁr + c0(Mr − M²)) / (r²(1−Mr)³) ]_ret dS

p' = p'_T + p'_L;  Mi = vi/c0, Mr = Mi r̂i, Lr = Li r̂i, LM = Li Mi;
overdot = source-time derivative; τ_ret = t − r_ret/c0; n̂ outward; di = xi − yi.

## Wind-tunnel simplification (Garrick triangle)

For stationary surface in uniform flow (speed U0, Mach M0, AoA α), retarded-time geometry is
closed-form and time-invariant; n̂̇ = 0, ṀR = 0:

- β² = 1 − M0², R* = sqrt(d1'² + β²(d2'² + d3²)) with d' the AoA-rotated separation,
  R = (−M1 d1 − M2 d2 + R*)/β², M1 = M0 cos α, M2 = M0 sin α.
- R̂1' = (−M0 R* + d1')/(β² R), R̂2' = d2'/R, R̂3 = d3/R, rotate back by α.
- R and R̂ are constants per (panel, observer) — precompute once.
- Wind-tunnel source terms: Qn = [−ρ0 U0i + ρ ui] n̂i, Li = [Pij + ρ(ui − U0i) uj] n̂j; M = M0.

Simplified integrals (Eqs. 32–33): same as 1A above with r→R, r̂→R̂, and the n̂̇/Ṁr terms removed.

Nondimensionalization used by the code (Eqs. 41–45): ρ*=ρ/ρ0, u*=u/U0, p*=p/(ρ0U0²),
c0*=1/M0, t*=t U0/Lref — introduces extra M0 factors; see the paper if bit-matching is needed.

## Time algorithm (source-time marching / advanced time)

1. No retarded-time root-finding: contribution from source time τk arrives at t = τk + R/c0.
2. Q̇, L̇ via 2nd-order central FD in source time (one-sided at ends).
3. Common valid observer window: t_start = max_panels(R)/c0 …
   t_end = τ_max + min_panels(R)/c0 (nondim: t* = R* M0 forms).
4. Per-panel arrival series → cubic-spline resample onto the shared observer grid → sum.
5. Surface integration: one-point quadrature (panel-center value × panel area).
6. Observers processed independently (trivially vmappable in JAX).

## Surface placement & sampling

- Analytic validation: sphere r=2 m, 18×36 = 648 panels.
- 30P30N: surface 1 chord off the body, extended 5c downstream, NO downstream end-cap
  (open outflow face avoids spurious noise as wake eddies cross the surface).
- Exchanged fields per frame: (ρ, u1, u2, u3, p) at panel centers.
- Sampling: CFD Δt 2e-7 s, FW-H sampled every 50 steps (1e-5 s → 100 kHz), 6535 frames.

## Validation

- Stationary monopole (M0=0.6, α=45°, f=5 Hz) and dipole (M0=0.5, α=10°, f=7.5 Hz) in uniform
  flow vs analytics: directivity/time signals practically identical. Good first test cases —
  synthesize surface data from analytic potential φ = A/(4πR*) exp[iω(t−R/c0)],
  u'=∇φ, p' = −ρ0(∂t + U0i ∂i)φ, ρ' = p'/c0².
- 30P30N: CL within 2.3% of workshop; far-field PSD matches JAXA experiment below 10 kHz.
  Spanwise correction: PSD_corr = PSD + 10 log10(b_exp/b_m).
- Spectra: Welch, 50% overlap, Hanning, dB/Hz.

## Cost

18,252 panels × 6,535 frames × 40 observers ≈ 5 CPU-hours serial, 62.6 GB RAM in their code —
trivial for a JAX vectorized implementation on GPU.

## Implications for AuraFlow

- Implement permeable FW-H with the *stationary-surface* fast path (static sphere around drone,
  optional uniform mean flow / AoA via Garrick triangle) + the general moving-surface 1A path
  shared with the BEMT/impermeable solver.
- Drop quadrupole volume term; ensure sphere encloses tip vortices.
- Use source-time-marching + interpolation to a common observer grid (JAX: vmap over panels
  and observers; interpolation via jnp.interp or cubic).
- Analytic monopole/dipole-in-flow tests are the correctness gate for the implementation.
- CFD side needs: sponge layers, low-dissipation schemes, surface sampling of (ρ, u, p) at
  panel centers every N steps, transient discard.
