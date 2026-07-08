# Auralization digest: Griffin–Lim + 1/3-octave band-to-audio practice

## Griffin–Lim (GLA)

- Griffin & Lim 1984 (DOI 10.1109/TASSP.1984.1164317): alternating projections between
  consistent spectrograms (ISTFT→STFT) and correct-magnitude set; LSE inverse = windowed
  overlap-add / sum of squared windows (standard istft). Monotonic error decrease, sublinear,
  local minima.
- Fast GLA (Perraudin et al., WASPAA 2013): FISTA-style momentum, α≈0.99 de facto standard.
- librosa.griffinlim defaults (practical recipe): n_iter=32 (32–100 typical), Hann,
  hop = n_fft/4, momentum 0.99, random phase init; update includes
  `angles -= momentum/(1+momentum) * tprev` then re-impose magnitude.

## Band-level → STFT-bin magnitude conversion (NASA practice)

- Rizzi & Sullivan AIAA 2005-2983: divide total power in each 1/3-octave band uniformly
  across the narrowband bins falling in the band; i.i.d. uniform random phase per bin;
  overlap-add buffer-wise synthesis (~512-sample buffers @ 44.1 kHz). ≥8192-tap filters
  reproduce spectra accurately.
- NASA UAM rotor broadband (Krishnamurthy/Rizzi AIAA 2021, NTRS 20205010694): ANOPP2 ASNIFM
  gives 1/3-octave SPL time histories per rev (bands k=10..49, f_c,k = 10^(k/10) Hz);
  per-band band-limited unit noise (uniform magnitude √(N/fs), random phase, Hermitian
  symmetric) amplitude-modulated by upsampled sqrt(band mean-square) envelope:
  p_k[n] = s_k[n]·w_k[n]/w_RMS,k. Separate stochastic signal per rotor (avoid coherence).
  Tonal noise synthesized separately (additive sines / F1A) and summed.
- NASA does NOT use Griffin–Lim; Ko et al. 2023 (JASA 154:3004, DOI 10.1121/10.0022352,
  paywalled) DOES: assemble magnitude spectrogram (tonal + broadband), reconstruct with GLA;
  justification: broadband inter-rotor phase is perceptually unimportant. Their exact GLA
  hyperparameters unpublished — use librosa-style defaults.

## Band definitions

- Base-2: f_c = 1000·2^((n−30)/3), edges f_c·2^(±1/6).
- Base-10 (IEC 61260-1:2014 preferred, and what NASA band numbers use):
  f_c = 10^(k/10) Hz, ratio 10^(1/10), edges f_c·10^(±1/20). Band 30 = 1 kHz.

## Scaling caveat

Bin magnitudes must respect Parseval for the chosen STFT convention (two-sided factor 2,
FFT length, window power Σw²). Robust practice: synthesize → re-analyze into 1/3-octave
levels → verify round-trip (NASA does this; our tests do the same).

## Implication for auraflow

`signal.synthesis.third_octave_to_stft_mag` uses uniform power spreading (NASA rule);
`cona.auralize` = band spectrogram → spread → (fast) GLA (momentum 0.99, Hann, hop n_fft/4)
→ add tonal time series. Alternative validated path (NASA-style modulated noise, no GLA)
can be added later as an option.
