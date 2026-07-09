# Loudspeaker model (`auraflow.body.speaker`)

A `Speaker` is the degenerate-motion FW-H body: a **static** rigid enclosure
whose selected *membrane* faces carry a prescribed normal-velocity signal
`u_n(t)`. The membrane radiates as thickness (monopole) sources through the
mesh → FW-H path (`auraflow.body.mesh_pressure`). Rigid-enclosure scattering is
neglected — exact in the free-field and baffled limits used by the analytic
gates (validated against the Rayleigh baffled circular piston).

## Build a piston or import a cabinet

```python
import jax.numpy as jnp
from auraflow.core.medium import Medium
from auraflow.body import Speaker, circular_piston, load_mesh, select_faces

medium = Medium()  # ISA sea level

# (a) the canonical validation object: a baffled circular piston (disk membrane).
piston = circular_piston(radius=0.1, n=8, baffled=True)

# (b) an imported cabinet STL/OBJ/PLY; pick the membrane faces by a predicate on
#     face centroids (here: every face whose centroid is on the +x cabinet wall).
cabinet = load_mesh("cabinet.stl")           # needs the `mesh` extra (trimesh)
speaker = Speaker.from_mesh(
    cabinet,
    lambda c: c[:, 0] > c[:, 0].max() - 1e-3,  # +x face -> the driver cone
    baffled=False,
)
# or pass explicit ids: Speaker.from_mesh(cabinet, select_faces(cabinet, pred))
```

`circular_piston(radius, n)` builds a concentric-ring disk (`n` radial rings,
`4n` angular sectors) so each panel stays ≪ wavelength — the piston
directivity/Rayleigh gates depend on the phase variation across the membrane.
`baffled=True` models an infinite rigid baffle in the membrane plane by an image
source (the radiated pressure is doubled); it is valid for listeners on the
source side of the baffle plane.

## Play an audio file / drive with a velocity signal

```python
# Decode a WAV to a mono waveform (any loader; here a placeholder array).
import numpy as np
fs = 44_100.0
audio = np.sin(2 * np.pi * 1000.0 * np.arange(fs // 20) / fs)  # 50 ms, 1 kHz

listeners = jnp.array([[0.0, 0.0, 1.0], [0.5, 0.0, 1.0]])  # two mics [O, 3]

# `play` maps the waveform straight onto membrane velocity u_n = gain * audio on
# the speaker's own sample grid tau = arange(len(audio)) / fs.
p, t_obs = piston.play(audio, fs, listeners, medium, gain=1.0)
# p: radiated pressure [Pa], shape [O, T_obs]; t_obs: observer-time grid [s].

# Lower-level: supply your own u_n(t) on an explicit tau grid.
tau = jnp.arange(len(audio)) / fs
u_n = jnp.asarray(audio)  # membrane normal velocity [m/s]
p, t_obs = piston.radiate(u_n, tau, listeners, medium)
```

**Idealization.** `play` treats the waveform *as the cone velocity* (units m/s
after `gain`); there is no electroacoustic transducer or enclosure-compliance
model between the electrical signal and the cone. `u_n` may be a single `[T]`
signal shared by every membrane face, or per-face `[Fm, T]`.

Everything is differentiable: e.g. `jax.grad` of the listener OASPL with respect
to `gain`, the enclosure vertices, or the medium properties all work directly.
