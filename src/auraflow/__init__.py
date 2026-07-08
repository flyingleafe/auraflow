"""AuraFlow: differentiable aeroacoustics simulations in JAX.

Backends, from fastest/most approximate to slowest/most precise:

- ``auraflow.bemt``: blade-element momentum theory loading + compact-chordwise
  Farassat 1A tonal noise (ported from ``fwh_rotor_sim``).
- ``auraflow.cona``: CONA-style end-to-end framework — BEMT/analytic loading,
  tonal + broadband (BPM) noise, atmospheric propagation.
- ``auraflow.cfd``: full compressible CFD in a near-field region coupled to a
  permeable-surface FW-H solver for far-field propagation.

Shared infrastructure lives in ``auraflow.core`` (geometry, frames, airfoils),
``auraflow.fwh`` (FW-H formulations), and ``auraflow.signal`` (spectra, SPL).
"""

__version__ = "0.1.0"
