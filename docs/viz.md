# Live in-browser 3-D visualization

`auraflow.viz` streams a running simulation to a self-contained three.js page in
the browser — **while it computes** — over a WebSocket. It has two halves:

- a **streaming hub + `VizStreamer`** (`auraflow.viz.server`) the simulation loop
  feeds, which also serves the frontend over HTTP on the same port;
- a compact **binary wire protocol** (`auraflow.viz.stream`, pure NumPy/stdlib)
  and the packaged **frontend** (`auraflow/viz/static/{index.html,app.js}`).

`websockets` is an optional dependency (the `viz-live` extra); it is imported
lazily, so `import auraflow` and `import auraflow.viz.stream` work without it.
three.js is pinned (r0.160.0) via a CDN import map — no build step, no npm.

## Install

```sh
uv sync --extra viz-live            # + --extra cfd for the CFD demo
# or: pip install 'auraflow[viz-live]'
```

## Run the demos

Both scripts start the hub in-process, print the URL, and run until Ctrl-C.
They are sized for the low-RAM dev box (tiny grids / short flights).

```sh
# Live CFD: a Gaussian pressure pulse expanding onto the permeable sphere.
uv run --extra cfd --extra viz-live python scripts/viz_demo_cfd.py \
    --cells 64 64 1 --steps 400

# CONA flyover replay: quadrotor over a ground mic line, rotor disks spinning.
uv run --extra viz-live python scripts/viz_demo_flyover.py
```

Open the printed `http://localhost:8000` in a browser. The page shows the domain
box, the permeable-sphere point cloud coloured by p′, the field slice as a
colour-mapped plane, the vehicle (rotor disks + arms), the mic array, and a
strip chart of selected pressure traces. Controls: orbit (drag/scroll),
Live / Pause, and a scrubber over the replay buffer; the HUD shows connection
status, time, frame index, and buffered-frame count.

A standalone hub (serves the page + accepts remote producers on `/produce`):

```sh
python -m auraflow.viz.server --port 8000
```

## Instrumenting your own loop

```python
from auraflow.viz import VizStreamer

with VizStreamer(port=8000) as viz:          # enabled=False -> zero-overhead no-op
    viz.init_scene(box_min=..., box_max=..., sphere_points=..., mics=..., rotors=...)
    for step in range(n):
        ...                                   # advance the simulation
        if viz.active:                        # skip device_get when nobody watches
            viz.push_frame(t=t, step=step, field_slice=sl, sphere_p=p)
```

`push_frame` is non-blocking and best-effort: it schedules a send on the hub's
event loop and returns immediately; slow clients get old frames dropped from
their bounded queues rather than stalling the sim. Downsample fields on the sim
side first with `downsample_slice` (≤64²) / `downsample_brick` (≤32³).

Backend hooks:

- **CFD**: `auraflow.cfd.run.run_acoustic_case(..., viz=streamer)` pushes a
  downsampled mid-plane slice + sphere overpressure every sample step
  (`auraflow.viz.cfd` builds the payloads).
- **CONA flyover**: `auraflow.viz.flyover.stream_flyover(streamer, vehicle,
  flight, mics=, mic_signals=)` replays a finished `FlightHistory` as an
  animation (the CONA stages are batch, so it is post-hoc replay).

## Message protocol (v1)

Every message is one binary WebSocket frame:

| bytes | contents |
|-------|----------|
| `0:4` | uint32 **big-endian** header length `H` |
| `4:4+H` | UTF-8 **JSON header** |
| `4+H:` | **binary payload** — arrays concatenated back-to-back |

The JSON header always has `"v"` (protocol version, `1`), `"type"`
(`"scene"` \| `"frame"`), and `"arrays"`: a list of
`{name, dtype, shape, offset, nbytes}` locating each array in the payload.
Payload dtypes are compact (`float32` fields/positions/pressures, `int32`,
`uint8`); float64 is downcast to float32 on encode.

### `scene` (sent once per client on connect; cached by the hub)

| field | where | meaning |
|-------|-------|---------|
| `box_min`, `box_max` | header | domain box corners `[3]` [m] |
| `rotors` | header | list of `{hub[3], radius, n_blades, axis[3], spin}` (world, m) |
| `slice_plane` | header | `{axis, coord, u_axis, v_axis, u_range, v_range}` for the field plane |
| `fields`, `title`, `dt` | header | field names shown, page title, nominal frame `dt` |
| `sphere_points` | payload | permeable-surface points `[S,3]` [m] |
| `mics` | payload | microphone/observer positions `[M,3]` [m] |

### `frame` (one per pushed step)

| field | where | meaning |
|-------|-------|---------|
| `t`, `step` | header | physical time [s], integer index |
| `slice_range` | header | `[lo, hi]` colour scale for the slice/brick |
| `vehicle_pos`, `vehicle_R` | header | pose: position `[3]`, attitude flat `[9]` (world←body, row-major) |
| `rotor_azimuths` | header | reference-blade azimuth per rotor `[Nr]` [rad] |
| `field_slice` | payload | 2-D scalar-field slice `[H,W]` (already downsampled) |
| `brick` | payload | 3-D scalar brick `[nx,ny,nz]` (already downsampled) |
| `sphere_p` | payload | p′ at the sphere points `[S]` [Pa] |
| `mic_p` | payload | instantaneous per-mic pressure `[M]` [Pa] |
| `mic_ring` | payload | rolling per-mic pressure window `[M,L]` [Pa] for the strip chart |

All arrays are optional; a backend sends only what it has.

## Tests

`tests/viz/` (no browser, no GPU/JAX): `test_stream.py` — protocol round-trip +
downsampler correctness; `test_server.py` — hub smoke (Python WS client receives
scene + frames) and the disabled/no-consumer no-op paths; `test_static.py` — the
frontend ships and is internally linked. Run one file at a time, memory-capped,
per the repo's RAM rules.
