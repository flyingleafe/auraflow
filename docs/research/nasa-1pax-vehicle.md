# NASA 1-Pax UAM quadrotor — implementation spec

The JASA paper's vehicle (583.85 kg) is the **rotor-speed-controlled 1-Pax variant** of
Malpica & Withrow-Maser, VFS TVF 2020, Table 1 (1287.17 lb), a resize of the original
Johnson/Silva/Solis 2018 concept (NTRS 20180003381). NOT in AIAA 2018-3847 (that's 6-pax).

## Adopted numbers (RPM-control variant unless noted)

- Design gross weight 583.85 kg; payload 113.4 kg (250 lb).
- 4 rotors × 3 blades; rotor radius **6.4 ft = 1.951 m**; disk loading 2.5 lb/ft² (119.7 N/m²).
- Thrust-weighted solidity **0.065**; hover tip speed **450 ft/s = 137.16 m/s**;
  Ω = 70.3 rad/s = **671 RPM** (collective-control sibling: R=6.5 ft, 662 RPM — matches the
  CONA paper's UAM case).
- Flap frequency 1.030/rev, 4% hinge offset, δ3 = 45°; Lock number 3.66;
  rotor inertia 12.11 slug·ft² = 16.42 kg·m²; aircraft inertias Ixx/Iyy/Izz =
  735.8/803.2/971.9 slug·ft² = 997.7/1089.1/1317.8 kg·m²; motor 23.5 hp each.
- Rotor design thrust 338.2 lb = 1504 N; power 20.8 hp each (hover).
- Hover FM: rotor 0.77, aircraft 0.71. CT/σ ≈ 0.104 front and rear.

## Layout

- X-arrangement: hubs at lateral AND longitudinal stations ±1.35R from array midpoint
  (adjacent-disk separation 35%R; arm 1.91R from centroid). For R=1.951 m: (±2.63, ±2.63) m.
- **Rear rotors 0.35R (0.683 m) above front rotors.**
- CG 0.9 ft (0.274 m) forward of rotor-array midpoint (flapping variant).
- Rotation (read from 6-pax AIAA 2018-3847 Fig. 3 arrows, standard torque balance):
  front-right CCW, front-left CW, rear-right CW, rear-left CCW (viewed from above).

## Blade geometry — published vs reconstructed

Published: linear twist **−12°** root-to-tip, taper **0.75** (tip/root chord), solidity 0.065.
NOT published: chord/twist tables, airfoil designations ("modern airfoils"), tip shape,
hub height. No CAMRAD II/OpenVSP/NDARC files for the 1-pax (6-pax only at
https://www.nasa.gov/reference/uam-refs/).

Reconstruction (document in code as assumptions):
- Thrust-weighted mean chord c̄ = σπR/Nb ≈ 0.065·π·1.951/3 = 0.1329 m; with taper 0.75
  linear chord c(r) = c_root·(1 − (1−0.75)·(r−r_hub)/(R−r_hub)), c_root chosen so the
  thrust-weighted solidity (weighted by r²) equals 0.065.
- Linear twist θ(r): −12° total, root-to-tip; absolute collective set by trim.
- Airfoil: use a representative low-Re-capable rotorcraft section (e.g. NACA 23012 or
  Boeing VR-12 class); CONA's UAM case used XFOIL-generated polars — we do the same and
  note sensitivity.
- Root cutout: assume 0.15R (typical; not published).

## Mission context (Johnson 2018 Table 1)

Hover 2 min OGE at 5000 ft ISA+20; cruise 50 nm at 70 kt; reserve 20 min.
JASA cases: level flight 1–10 m/s at 30 m altitude (landing-approach speeds).
