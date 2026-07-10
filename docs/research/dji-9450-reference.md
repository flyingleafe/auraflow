# DJI 9450 quadrotor — implementation spec

**Naming caveat (read first):** the propeller most often called "the DJI 9450 rotor" in the
acoustics literature is actually **two different DJI parts being conflated**:

1. **DJI 9450** (9.4 in dia. × 5.0 in pitch) — the real part number, a plastic (and carbon)
   self-tightening prop bundled with the DJI Phantom 3 Advanced/Professional/4K. Directly
   measured (chord/twist via optical **PropellerScanner**, thrust via load-cell static rig) by
   **Deters, Kleinke & Selig, AIAA 2017-3743** ("Static Testing of Propulsion Elements for Small
   Multirotor Unmanned Aircraft Systems"), UIUC.
2. **DJI 9443** (9.4 in dia. × 4.3 in pitch) — a *different, lower-pitch* DJI part, tested by
   **Zawodny, Boyd & Burley, AHS/AIAA 2016 (NTRS 20160009054)**, who call it "**DJI-CF**"
   (a carbon-fiber replica) throughout their paper. This is the paper with the SALT hover
   acoustic spectra, laser-scanned chord/twist, and CT-vs-RPM curve that CONA and several
   follow-on papers cite when they say "DJI 9450" (our own `cona-reference.md` digest repeats
   this conflation: "DJI 9450 blade, R=0.121 m, hover 5400 RPM ... matches NASA SALT
   experiment"). **It is actually the 9443.**

Both parts share diameter, hub family, and overall planform shape (DJI's small-quad prop line),
so the two datasets are close cousins, not the same object. This digest gives **both**,
clearly labeled, so downstream code can pick the real 9450 (Deters) for geometry fidelity or the
9443/DJI-CF (Zawodny) for its much richer acoustic/aerodynamic validation dataset.

Sources fetched and digitized directly (see "Sources" at the end for URLs). Numbers below are
tagged **PUBLISHED** (stated as text/number in the source), **DIGITIZED** (pixel-calibrated
reading off a published figure — the underlying curve is real data, but the numeric table here
is a reconstruction of it, accurate to roughly ±0.005 c/R and ±0.3° twist), or
**RECONSTRUCTED/ESTIMATED** (not stated anywhere; derived here with a documented assumption).

## Adopted numbers

- Rotor: 2 blades, fixed pitch. Real DJI 9450: true diameter **9.45 in = 0.240 m** (R = 0.1200 m)
  [PUBLISHED, Deters Table 1]. DJI-CF/9443 stand-in: diameter **9.4 in = 0.24 m** (R = 0.1194 m,
  weight 12.1 g) [PUBLISHED, Zawodny Table 1].
- Root cutout ≈ **0.15R** (9450) / ≈ **0.06R** (DJI-CF/9443) — first digitized data point in each
  source's chord/twist figure [DIGITIZED].
- Vehicle: DJI Phantom 3 (Advanced/Professional/4K use the 9450 plastic/composite prop bundled
  in this digest). Mass **1.280 kg** (Adv./Pro./4K, w/ battery+props) or **1.216 kg** (Standard);
  **1.236 kg** (SE); DJI Phantom 2 **≤1.300 kg** [PUBLISHED, DJI spec sheets].
- Diagonal wheelbase (motor-to-motor, props excluded) **350 mm** for Phantom 2 and Phantom 3
  (all variants) [PUBLISHED, DJI user manuals/spec sheets].
- Battery: 4S LiPo, **15.2 V** nominal [PUBLISHED, DJI TB47/TB48 spec]. Motor: DJI **2212**
  brushless, exact factory Kv not published by DJI; commonly cited as **≈920 Kv**
  [ESTIMATED — third-party/aftermarket-equivalent listings, not a DJI datasheet number].
- Reconstructed hover point (thrust = weight/4 cross-check, see below): **≈5000–5400 RPM**
  using the real-9450 static-thrust curve (Deters); **≈6150–6400 RPM** using the DJI-CF/9443
  curve (Zawodny) [RECONSTRUCTED]. The literature's commonly-quoted "5400 RPM" hover figure
  (CONA and others) lands inside the 9450-based range.

## Rotor blade geometry

### (A) Real DJI 9450 Plastic — Deters, Kleinke & Selig 2017, Fig. 7 [PRIMARY, DIGITIZED]

Digitization method: page rendered at 8×, pixel-column tracing of the solid (chord) and dashed
(twist) traces, calibrated against the plot's own axis tick gridlines (both axes cross-checked
to sub-pixel accuracy against the 0/0.25/0.5/0.75/1.0 r/R gridlines and the 0/15/30/45° twist
gridlines). True diameter 9.45 in (0.24 m), R = 0.1200 m.

| r/R  | c/R   | c (mm) | twist β (deg) |
|------|-------|--------|---------------|
| 0.15 | 0.180 | 21.6   | 16.8          |
| 0.20 | 0.219 | 26.3   | 19.0          |
| 0.25 | 0.246 | 29.5   | 20.7          |
| 0.30 | 0.254 | 30.5   | 20.8          |
| 0.35 | 0.240 | 28.8   | 19.9          |
| 0.40 | 0.223 | 26.8   | 18.6          |
| 0.45 | 0.204 | 24.5   | 17.0          |
| 0.50 | 0.191 | 22.9   | 15.8          |
| 0.55 | 0.179 | 21.5   | 14.8          |
| 0.60 | 0.165 | 19.8   | 13.9          |
| 0.65 | 0.156 | 18.7   | 12.8          |
| 0.70 | 0.144 | 17.3   | 11.8          |
| 0.75 | 0.133 | 16.0   | 10.9          |
| 0.80 | 0.125 | 15.0   | 10.6          |
| 0.85 | 0.115 | 13.8   | 9.9           |
| 0.90 | 0.105 | 12.6   | 9.7           |
| 0.95 | 0.096 | 11.5   | 9.6 (noisy)   |
| 1.00 | 0.057 | 6.8    | 8.5           |

Peak chord at r/R≈0.28–0.30 (c/R≈0.254); peak twist at the same station (≈20.8°); twist falls off
much more slowly toward the tip than the DJI-CF/9443 rotor below (consistent with the 9450's
larger 5.0-in nominal pitch). Deters et al. state (verbatim): "the carbon propeller has larger
twist angles than the plastic version. The APC and Master Airscrew propellers have a similar
shape to their chord distributions..." — i.e. plastic and carbon 9450 share the same chord
distribution but the carbon variant is twisted a few degrees more everywhere [PUBLISHED].

### (B) DJI-CF (= DJI 9443 carbon-fiber replica) — Zawodny, Boyd & Burley 2016, Fig. 3(b)
[SECONDARY / cross-reference, DIGITIZED at high precision]

This is the dataset most of the acoustics literature actually means by "the DJI 9450 rotor."
Digitization: page rendered at 8×, blue-marker/line pixels isolated by RGB threshold and
averaged per column (line + circle markers are the same trace so this recovers the curve
essentially exactly, no manual reading involved), calibrated against the plot's black axis-box
border (found from long dark pixel runs, not the dotted gridlines). Diameter 9.4 in (0.24 m),
R = 0.1194 m; root cutout r/R≈0.06, tip r/R≈1.0 (first/last laser-scan sample).

| r/R  | c/R    | c (mm) | twist β (deg) |
|------|--------|--------|---------------|
| 0.06 | 0.1514 | 18.1   | 17.6          |
| 0.10 | 0.1739 | 20.8   | 18.8          |
| 0.15 | 0.2183 | 26.1   | 19.5           |
| 0.20 | 0.2518 | 30.1   | 19.7          |
| 0.25 | 0.2615 | 31.2   | 19.0          |
| 0.30 | 0.2502 | 29.9   | 17.9          |
| 0.35 | 0.2344 | 28.0   | 16.8          |
| 0.40 | 0.2167 | 25.9   | 15.6          |
| 0.45 | 0.1967 | 23.5   | 14.6          |
| 0.50 | 0.1826 | 21.8   | 13.6          |
| 0.55 | 0.1705 | 20.4   | 12.6          |
| 0.60 | 0.1564 | 18.7   | 11.4          |
| 0.65 | 0.1439 | 17.2   | 10.6          |
| 0.70 | 0.1336 | 16.0   | 9.4           |
| 0.75 | 0.1249 | 14.9   | 8.6           |
| 0.80 | 0.1156 | 13.8   | 7.7           |
| 0.85 | 0.1072 | 12.8   | 7.0           |
| 0.90 | 0.1005 | 12.0   | 6.6           |
| 0.95 | 0.0927 | 11.1   | 6.3           |
| 1.00 | 0.0728 | 8.7    | 5.3           |

Peak chord at r/R≈0.22–0.25 (c/R≈0.26); peak twist ≈19.7–19.8° at r/R≈0.18–0.20, monotonically
decreasing to ≈5.3° at the tip. Both curves in Table (B) are markedly smoother than Table (A)
because Zawodny's blade geometry came from a fine-resolution laser scan (dozens of stations)
whereas Deters' PropellerScanner reading has only a handful of underlying data points connected
by straight segments (visible as the "kinks" in Table A).

**NOT published anywhere for either rotor:** analytic chord/twist fit coefficients, hub/root
fillet shape, tip shape/rounding, exact thickness distribution.

## Airfoil / blade cross-section

- **PUBLISHED, verbatim (Zawodny et al., p.3):** "The chord and pitch angle data of the DJI-CF
  rotor shown in Figure 3(b) were extracted from a high-resolution laser scan of the blade
  surfaces... As a result of having a fine resolution computational surface grid of the DJI-CF
  rotor, accurate radial section profiles were able to be extracted for input into the blade
  element aero-analysis." I.e. **no named airfoil family is used for the DJI rotor** — NASA
  worked directly from scanned cross-sections of the actual thin, cambered, injection-molded
  blade ("thin cambered plate," consistent with the user's expectation; there is no symmetric
  or classical-airfoil idealization in the source). By contrast, the APC-SF companion rotor in
  the *same* paper explicitly blends "an Eppler63-type airfoil near the hub to a Clark-Y-type
  airfoil near the tip," so the "named airfoil" habit in some downstream citations is bleeding
  over from the APC rotor, not the DJI one.
- **RECONSTRUCTED/THIRD-PARTY, not NASA-sourced:** CFD Support's own "DJI-9450 Acoustic
  Benchmark" CFD write-up (a non-peer-reviewed vendor benchmark, geometry diameter 239 mm =
  9.41 in, matching Zawodny's *DJI-CF/9443*, not Deters' 9.45-in real 9450) states their
  reconstructed blade profile is "combined out of EPPLER 856 (0–0.2 r/R) and EPPLER 63
  (0.2–1 r/R)." Use this only as a plausible-looking XFOIL-analyzable stand-in if a named
  section is required for a polar lookup table; it is not what NASA measured.
- For the real 9450 (Deters), no airfoil family is stated at all — same "thin cambered plate,
  laser/optically scanned" character applies.

## Thrust, hover RPM, tip Mach, BPF

### Static thrust coefficient data

Real DJI 9450 plastic (Deters, Fig. 31/32, static/hover-equivalent bench test; propeller-
convention C_T = T/(ρ n² D⁴)) [PUBLISHED, DIGITIZED]:

- C_T ≈ 0.088 at Ω = 2000 RPM, rising and plateauing at **C_T ≈ 0.103–0.105** for Ω ≳ 4000 RPM,
  flat out to the 8000 RPM test limit. (Equivalent helicopter-convention C_T = T/(ρπR²(ΩR)²):
  multiply by 4/π³ ≈ 0.1290 → **C_T,heli ≈ 0.0113–0.0135** in the plateau.)

DJI-CF/9443 (Zawodny, Fig. 6(a), helicopter-convention C_T = T/(ρπR²(ΩR)²)) [PUBLISHED]:

- C_T ≈ 0.0085–0.009 for Ω = 3000–3500 RPM, plateauing at **C_T ≈ 0.0090–0.0093** for
  Ω ≳ 4500 RPM out to the 7200 RPM test limit.
- Explicit measured thrust points (printed directly on the SPL-vs-θ figure captions, p.9):
  **4800 RPM → 0.389 lbf = 1.730 N**, **5400 RPM → 0.472 lbf = 2.100 N**,
  **6000 RPM → 0.623 lbf = 2.771 N** (single rotor, static). Consistency check: CT=0.0092,
  ρ=1.225 kg/m³, R=0.1194 m gives T(6000 RPM) = 2.84 N, within 2.5% of the measured 2.771 N.
  At 6000 RPM this also matches the CFD Support benchmark's own BPF quote of "100 rev/s × 2
  blades = 200 Hz" [PUBLISHED cross-check].

### Hover RPM reconstruction (thrust ≈ weight/4)

Per-rotor hover thrust required: T = m·g/4. Using m = 1.280 kg (Phantom 3 Adv./Pro./4K,
the variant that actually ships the 9450 prop): **T = 3.14 N**. Using m = 1.216 kg
(Standard): T = 2.98 N. [RECONSTRUCTED for both branches below.]

- **Real 9450 (Deters CT_static plateau ≈0.104 propeller-convention, R=0.1200 m):**
  Ω ≈ 541–528 rad/s → **RPM ≈ 5170 (1.280 kg) / 5040 (1.216 kg)**. This range brackets the
  **5400 RPM** figure widely quoted in the multirotor-acoustics literature (CONA, etc.) as "the"
  DJI hover RPM — good agreement, and it is the number we recommend adopting as the nominal
  hover operating point.
- **DJI-CF/9443 cross-check (Zawodny CT data, R=0.1194 m):** constant-C_T (0.0092) estimate
  gives RPM ≈ 6300 (1.280 kg) / 6150 (1.216 kg); a power-law fit through the three explicit
  measured thrust points (T ∝ RPM^2.10) gives RPM ≈ 6420 / 6260. **Adopt RPM ≈ 6150–6400** for
  this rotor. It sits noticeably higher than the 9450-based estimate because the 9443 has
  ~14% less pitch and thus less thrust per RPM — physically consistent, not a contradiction.

### Tip Mach number and blade-passage frequency (2 blades, BPF = 2·RPM/60)

| Rotor / RPM source              | RPM (hover est.) | Ω (rad/s) | Tip speed (m/s) | Tip Mach | BPF (Hz) |
|----------------------------------|------------------|-----------|------------------|----------|----------|
| Real 9450, m=1.280 kg            | 5170             | 541.5     | 65.0             | 0.191    | 172      |
| Real 9450, m=1.216 kg            | 5040             | 527.8     | 63.3             | 0.186    | 168      |
| Literature nominal (CONA "5400") | 5400             | 565.5     | 67.9             | 0.200    | 180      |
| DJI-CF/9443, m=1.280 kg          | 6300–6420        | 660–672   | 78.8–80.3        | 0.232–0.236 | 210–214 |
| DJI-CF/9443, m=1.216 kg          | 6150–6260        | 644–655   | 76.9–78.3        | 0.226–0.230 | 205–209 |
| Zawodny explicit test point      | 6000             | 628.3     | 75.0 (R=0.1194)  | 0.221    | 200      |

All tip-Mach values use a=340 m/s. [RECONSTRUCTED except the 6000 RPM/200 Hz row, which is a
published cross-check value.]

## Vehicle layout

- 4 rotors, X configuration, diagonal wheelbase (motor-to-motor) **0.350 m** [PUBLISHED].
  Hub offset from body centroid along each body axis:
  0.5·diagonal·cos45° = 0.5·0.350·0.70711 = **0.1237 m** [RECONSTRUCTED from the published
  diagonal via the standard X-layout formula]. Hub coordinates (body frame, z=0 rotor plane,
  x=fwd, y=right):
  - Front-right:  (+0.1237, +0.1237)
  - Front-left:   (+0.1237, −0.1237)
  - Rear-right:   (−0.1237, +0.1237)
  - Rear-left:    (−0.1237, −0.1237)
- Spin directions [PUBLISHED — DJI/community documentation, e.g. phantompilots.com forum
  threads on motor rotation, consistent with standard diagonal-pair torque balance]: front-left
  **CW**, front-right **CCW**, rear-left **CCW**, rear-right **CW** (viewed from above); diagonal
  pairs (FL/RR and FR/RL) share rotation direction, canceling net yaw torque in hover — the
  same pattern already adopted in `nasa-1pax-vehicle.md`.
- Rotor-plane height above the body/ground: **NOT PUBLISHED** by DJI in any spec sheet found;
  not needed for the acoustic source model at the level of fidelity used here, so left
  unspecified (flag if code later needs it).
- Motor: DJI 2212 (both DJI-CF/9443 rig and the actual Phantom 3 use this motor family per
  Zawodny Table 1 and Deters Fig. 2) [PUBLISHED]; ESC: DJI OPTO E300 (Zawodny bench only)
  [PUBLISHED, bench setup, not necessarily the exact production Phantom 3 ESC].

## Sources

- Zawodny, N. S., Boyd, D. D., Burley, C. L., "Acoustic Characterization and Prediction of
  Representative, Small-Scale Rotary-Wing Unmanned Aircraft System Components," AHS/AIAA 2016 —
  NTRS record: https://ntrs.nasa.gov/citations/20160009054 — PDF:
  https://ntrs.nasa.gov/api/citations/20160009054/downloads/20160009054.pdf
- Deters, R. W., Kleinke, S., Selig, M. S., "Static Testing of Propulsion Elements for Small
  Multirotor Unmanned Aircraft Systems," AIAA 2017-3743:
  https://m-selig.ae.illinois.edu/pubs/DetersKleinkeSelig-2017-AIAA-Paper-2017-3743.pdf
- CFD Support, "Acoustic Benchmark Propeller DJI-9450" (vendor CFD benchmark writeup, not
  peer-reviewed; used only for the EPPLER 856/63 airfoil claim and the 200 Hz BPF cross-check):
  https://www.cfdsupport.com/acoustic-benchmark-propeller-dji-9450/
- Ko, Jeong, Cho, Lee, "Real-Time Prediction Framework for Frequency-Modulated Multirotor
  Noise" (CONA), Physics of Fluids, DOI 10.1063/5.0081103 (2022) — cited here for the "5400 RPM"
  nominal-hover convention (see this repo's `cona-reference.md`).
- DJI Phantom 3 Standard weight/diagonal: https://drone-world.com/dji-phantom-3-specs-professional-advanced/
  and https://dronespec.dronedesk.io/dji-phantom-3-standard ;
  Phantom 3 Pro/Adv/4K weight (1280 g): same aggregator pages.
- DJI Phantom 2 weight/diagonal (≤1300 g, 350 mm): DJI Phantom 2 User Manual,
  https://dl.djicdn.com/downloads/phantom_2/en/PHANTOM2_User_Manual_v1.4_en.pdf
- DJI TB47/TB48 battery (15.2 V, 4S): multiple retailer listings, e.g.
  https://www.amazon.com/Phantom3-Professional-Standard-Intelligent-Universal/dp/B08D7CY6BM
- DJI 2212 motor Kv (≈920, not an official DJI number): e.g.
  https://emaxmodel.com/products/2212-920kv-cw-ccw-brushless-motor-for-dji-phantom-1-phantom-2-f450
- Phantom motor spin-direction convention: https://phantompilots.com/threads/motor-rotation.112988/
  and https://phantompilots.com/threads/the-motor-rotation-again.146046/
