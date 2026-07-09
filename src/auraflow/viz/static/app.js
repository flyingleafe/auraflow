// AuraFlow live-visualization frontend.
//
// Connects to the streaming hub over WebSocket, decodes the binary protocol
// (see auraflow/viz/stream.py), and renders the simulation in three.js: a
// domain-box wireframe, the permeable-sphere point cloud coloured by p', a
// scalar-field slice as a colour-mapped textured plane, the vehicle as
// rotor-disk + arm primitives animated by pose/azimuth, the microphone array as
// ground points, and a 2-D strip chart of selected pressure traces.
//
// One file, no build step. three.js is pinned via the import map in index.html.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const PROTOCOL_VERSION = 2;
const MAX_BUFFER = 3000; // frames kept for replay/scrub

// --------------------------------------------------------------------------
// Protocol decode (mirror of auraflow.viz.stream.encode_message)
// --------------------------------------------------------------------------
const DTYPE_CTOR = {
  float32: Float32Array, int32: Int32Array, uint32: Uint32Array, uint8: Uint8Array,
};

function decodeMessage(buf) {
  const dv = new DataView(buf);
  const headerLen = dv.getUint32(0, false); // big-endian
  const headerBytes = new Uint8Array(buf, 4, headerLen);
  const header = JSON.parse(new TextDecoder("utf-8").decode(headerBytes));
  if (header.v !== PROTOCOL_VERSION) throw new Error("bad protocol version " + header.v);
  const payloadOffset = 4 + headerLen;
  const arrays = {};
  for (const spec of header.arrays || []) {
    const Ctor = DTYPE_CTOR[spec.dtype];
    const count = spec.nbytes / Ctor.BYTES_PER_ELEMENT;
    // Copy out (payload offset may be unaligned for the typed-array ctor).
    const slice = buf.slice(payloadOffset + spec.offset, payloadOffset + spec.offset + spec.nbytes);
    arrays[spec.name] = { data: new Ctor(slice), shape: spec.shape };
  }
  return { header, arrays };
}

// --------------------------------------------------------------------------
// Colormap: diverging blue-white-red, t in [0,1] -> [r,g,b] in [0,1]
// --------------------------------------------------------------------------
function diverging(t) {
  t = Math.min(1, Math.max(0, t));
  const x = 2 * t - 1; // [-1,1]
  if (x < 0) {
    const a = 1 + x; // 0..1 toward white
    return [a, a, 1];
  }
  const a = 1 - x;
  return [1, a, a];
}

// --------------------------------------------------------------------------
// three.js scene scaffolding
// --------------------------------------------------------------------------
const canvas = document.getElementById("view");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0e14);
const camera = new THREE.PerspectiveCamera(55, 1, 0.001, 1e6);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const dir = new THREE.DirectionalLight(0xffffff, 0.8);
dir.position.set(1, 1, 2);
scene.add(dir);

function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

// Scene objects (rebuilt on each scene message).
let group = new THREE.Group();
scene.add(group);
let sphereCloud = null;   // THREE.Points, colored by p'
let slicePlane = null;    // textured plane
let sliceTexture = null;
let vehicleGroup = null;   // group with per-rotor disks
let rotorDisks = [];       // [{mesh, spin}]
let bodyMeshes = [];       // THREE.Mesh, one per scene mesh (pose-animated)
let sceneMeta = null;

function disposeGroup(g) {
  g.traverse((o) => {
    if (o.geometry) o.geometry.dispose();
    if (o.material) {
      const m = o.material;
      if (m.map) m.map.dispose();
      m.dispose();
    }
  });
}

// AXIS unit vector for a named axis.
function axisVec(name) {
  return { x: new THREE.Vector3(1, 0, 0), y: new THREE.Vector3(0, 1, 0), z: new THREE.Vector3(0, 0, 1) }[name];
}

function buildScene(header, arrays) {
  sceneMeta = header;
  scene.remove(group);
  disposeGroup(group);
  group = new THREE.Group();
  // World is z-up in AuraFlow; make three.js agree.
  group.up = new THREE.Vector3(0, 0, 1);
  scene.add(group);
  sphereCloud = null; slicePlane = null; vehicleGroup = null; rotorDisks = []; bodyMeshes = [];

  const bmin = header.box_min, bmax = header.box_max;
  const cx = (bmin[0] + bmax[0]) / 2, cy = (bmin[1] + bmax[1]) / 2, cz = (bmin[2] + bmax[2]) / 2;
  const dx = bmax[0] - bmin[0], dy = bmax[1] - bmin[1], dz = bmax[2] - bmin[2];
  const diag = Math.hypot(dx, dy, dz) || 1;

  // Domain box wireframe.
  const box = new THREE.Box3(
    new THREE.Vector3(bmin[0], bmin[1], bmin[2]),
    new THREE.Vector3(bmax[0], bmax[1], bmax[2])
  );
  const boxHelper = new THREE.Box3Helper(box, new THREE.Color(0x3b4252));
  group.add(boxHelper);

  // Ground grid at z = min (helps read the mic plane).
  const grid = new THREE.GridHelper(Math.max(dx, dy) * 1.2, 20, 0x30363d, 0x1c2128);
  grid.rotation.x = Math.PI / 2; // GridHelper is XZ by default -> lay in XY
  grid.position.set(cx, cy, Math.min(bmin[2], 0));
  group.add(grid);

  // Permeable-sphere point cloud.
  if (arrays.sphere_points) {
    const pts = arrays.sphere_points.data;
    const n = arrays.sphere_points.shape[0];
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pts, 3));
    const col = new Float32Array(n * 3).fill(0.6);
    geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
    const mat = new THREE.PointsMaterial({ size: diag * 0.012, vertexColors: true });
    sphereCloud = new THREE.Points(geo, mat);
    group.add(sphereCloud);
  }

  // Microphone array (ground points).
  if (arrays.mics) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(arrays.mics.data, 3));
    const mat = new THREE.PointsMaterial({ color: 0xe3b341, size: diag * 0.01 });
    group.add(new THREE.Points(geo, mat));
  }

  // Field-slice textured plane.
  if (header.slice_plane) {
    const sp = header.slice_plane;
    const uLen = sp.u_range[1] - sp.u_range[0];
    const vLen = sp.v_range[1] - sp.v_range[0];
    const geo = new THREE.PlaneGeometry(uLen, vLen);
    sliceTexture = new THREE.DataTexture(new Uint8Array(4), 1, 1, THREE.RGBAFormat);
    const mat = new THREE.MeshBasicMaterial({
      map: sliceTexture, transparent: true, opacity: 0.85, side: THREE.DoubleSide,
    });
    slicePlane = new THREE.Mesh(geo, mat);
    orientPlane(slicePlane, sp);
    group.add(slicePlane);
  }

  // Vehicle: rotor disks + arms.
  if (header.rotors && header.rotors.length) {
    vehicleGroup = new THREE.Group();
    for (const r of header.rotors) {
      const disk = new THREE.Group();
      const ring = new THREE.Mesh(
        new THREE.RingGeometry(r.radius * 0.92, r.radius, 40),
        new THREE.MeshBasicMaterial({ color: 0x58a6ff, side: THREE.DoubleSide,
          transparent: true, opacity: 0.5 })
      );
      disk.add(ring);
      // Blades as thin boxes.
      for (let b = 0; b < r.n_blades; b++) {
        const blade = new THREE.Mesh(
          new THREE.BoxGeometry(r.radius, r.radius * 0.06, r.radius * 0.02),
          new THREE.MeshStandardMaterial({ color: 0xc9d1d9 })
        );
        blade.position.set(r.radius / 2, 0, 0);
        const holder = new THREE.Group();
        holder.rotation.z = (2 * Math.PI * b) / r.n_blades;
        holder.add(blade);
        disk.add(holder);
      }
      // Orient the disk so its local +z points along the rotor thrust axis.
      const axis = new THREE.Vector3(r.axis[0], r.axis[1], r.axis[2]).normalize();
      disk.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), axis);
      disk.position.set(r.hub[0], r.hub[1], r.hub[2]);
      // Arm from body origin to hub.
      const arm = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(0, 0, 0), new THREE.Vector3(r.hub[0], r.hub[1], r.hub[2])]),
        new THREE.LineBasicMaterial({ color: 0x6e7681 })
      );
      vehicleGroup.add(arm);
      vehicleGroup.add(disk);
      rotorDisks.push({ group: disk, spin: r.spin || 1, axis });
    }
    group.add(vehicleGroup);
  }

  // Imported / parametric body meshes (flat-shaded, optionally translucent).
  if (header.meshes && header.meshes.length) {
    for (let i = 0; i < header.meshes.length; i++) {
      const meta = header.meshes[i];
      const verts = arrays["mesh" + i + "_vertices"];
      const faces = arrays["mesh" + i + "_faces"];
      if (!verts || !faces) continue;
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(verts.data, 3));
      geo.setIndex(new THREE.BufferAttribute(faces.data, 1));
      const cols = arrays["mesh" + i + "_colors"];
      if (cols) geo.setAttribute("color", new THREE.BufferAttribute(cols.data, 3));
      geo.computeVertexNormals();
      const baseColor = meta.color
        ? new THREE.Color(meta.color[0], meta.color[1], meta.color[2])
        : new THREE.Color(0x9db4d0);
      const opacity = meta.opacity != null ? meta.opacity : 1.0;
      const mat = new THREE.MeshStandardMaterial({
        color: baseColor,
        vertexColors: !!cols,
        flatShading: true,
        metalness: 0.1,
        roughness: 0.75,
        side: THREE.DoubleSide,
        transparent: opacity < 1.0,
        opacity,
      });
      const mesh = new THREE.Mesh(geo, mat);
      group.add(mesh);
      bodyMeshes.push(mesh);
    }
  }

  // Camera framing.
  camera.position.set(cx + diag, cy - diag, cz + diag * 0.7);
  controls.target.set(cx, cy, cz);
  controls.update();

  document.getElementById("title").textContent = header.title || "AuraFlow";
  document.getElementById("meta").textContent =
    (header.fields && header.fields.length ? "field: " + header.fields.join(",") + "  " : "") +
    (header.rotors && header.rotors.length ? header.rotors.length + " rotor(s)  " : "") +
    (arrays.sphere_points ? arrays.sphere_points.shape[0] + " sphere pts" : "");
}

function orientPlane(mesh, sp) {
  // PlaneGeometry lies in local XY (u along x, v along y); rotate so its normal
  // points along the slice axis, then translate to the slice coordinate.
  const u = new THREE.Vector3(), v = new THREE.Vector3();
  const uc = (sp.u_range[0] + sp.u_range[1]) / 2;
  const vc = (sp.v_range[0] + sp.v_range[1]) / 2;
  const set = { x: {}, y: {}, z: {} };
  if (sp.axis === "z") { mesh.position.set(uc, vc, sp.coord); }
  else if (sp.axis === "y") { mesh.rotation.x = Math.PI / 2; mesh.position.set(uc, sp.coord, vc); }
  else { mesh.rotation.y = Math.PI / 2; mesh.rotation.z = Math.PI / 2; mesh.position.set(sp.coord, uc, vc); }
  void u; void v; void set;
}

function updateSliceTexture(arr, range) {
  if (!slicePlane || !arr) return;
  const [h, w] = arr.shape; // [nu, nv]
  const data = arr.data;
  const lo = range ? range[0] : Math.min(...data);
  const hi = range ? range[1] : Math.max(...data);
  const span = hi - lo || 1;
  const rgba = new Uint8Array(w * h * 4);
  for (let i = 0; i < h; i++) {
    for (let j = 0; j < w; j++) {
      const val = data[i * w + j];
      const [r, g, b] = diverging((val - lo) / span);
      const k = (i * w + j) * 4;
      rgba[k] = r * 255; rgba[k + 1] = g * 255; rgba[k + 2] = b * 255; rgba[k + 3] = 235;
    }
  }
  if (!sliceTexture || sliceTexture.image.width !== w || sliceTexture.image.height !== h) {
    sliceTexture = new THREE.DataTexture(rgba, w, h, THREE.RGBAFormat);
    slicePlane.material.map = sliceTexture;
  } else {
    sliceTexture.image.data = rgba;
  }
  sliceTexture.needsUpdate = true;
  slicePlane.material.needsUpdate = true;
}

function updateSpherePressure(arr) {
  if (!sphereCloud || !arr) return;
  const p = arr.data;
  let amax = 0;
  for (let i = 0; i < p.length; i++) amax = Math.max(amax, Math.abs(p[i]));
  amax = amax || 1;
  const col = sphereCloud.geometry.getAttribute("color");
  for (let i = 0; i < p.length; i++) {
    const [r, g, b] = diverging(0.5 + 0.5 * (p[i] / amax));
    col.setXYZ(i, r, g, b);
  }
  col.needsUpdate = true;
}

function updateVehicle(header) {
  if (!vehicleGroup) return;
  if (header.vehicle_pos) vehicleGroup.position.fromArray(header.vehicle_pos);
  if (header.vehicle_R) {
    const R = header.vehicle_R; // row-major 3x3 (world<-body)
    const m = new THREE.Matrix4().set(
      R[0], R[1], R[2], 0,
      R[3], R[4], R[5], 0,
      R[6], R[7], R[8], 0,
      0, 0, 0, 1
    );
    vehicleGroup.quaternion.setFromRotationMatrix(m);
  }
  if (header.rotor_azimuths) {
    for (let i = 0; i < rotorDisks.length; i++) {
      const az = header.rotor_azimuths[i] || 0;
      // Spin about the disk's local +z (its thrust axis after orientation).
      const base = rotorDisks[i].baseQuat || (rotorDisks[i].baseQuat = rotorDisks[i].group.quaternion.clone());
      const spin = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), az);
      rotorDisks[i].group.quaternion.copy(base).multiply(spin);
    }
  }
}

function updateMeshes(header) {
  if (!bodyMeshes.length || !header.mesh_poses) return;
  for (let i = 0; i < bodyMeshes.length && i < header.mesh_poses.length; i++) {
    const p = header.mesh_poses[i]; // [px,py,pz, r0..r8] (row-major world<-body)
    bodyMeshes[i].position.set(p[0], p[1], p[2]);
    const m = new THREE.Matrix4().set(
      p[3], p[4], p[5], 0,
      p[6], p[7], p[8], 0,
      p[9], p[10], p[11], 0,
      0, 0, 0, 1
    );
    bodyMeshes[i].quaternion.setFromRotationMatrix(m);
  }
}

// --------------------------------------------------------------------------
// Strip chart (selected pressure traces)
// --------------------------------------------------------------------------
const chart = document.getElementById("chart");
const cctx = chart.getContext("2d");
const traces = [[], []]; // rolling scalar histories (fallback mode)
const TRACE_LEN = 400;
let selMics = [0, 0];

function pushTrace(vals) {
  for (let k = 0; k < 2; k++) {
    traces[k].push(vals[k]);
    if (traces[k].length > TRACE_LEN) traces[k].shift();
  }
}

function drawChartFromRing(arr) {
  // arr: [M, L] recent per-mic pressure; draw two selected mics over the window.
  const [m, L] = arr.shape;
  const rows = [selMics[0] % m, selMics[1] % m];
  drawChart(rows.map((r) => Array.from(arr.data.subarray(r * L, r * L + L))));
}

function drawChart(series) {
  const w = chart.width, h = chart.height;
  cctx.clearRect(0, 0, w, h);
  cctx.strokeStyle = "#30363d";
  cctx.beginPath(); cctx.moveTo(0, h / 2); cctx.lineTo(w, h / 2); cctx.stroke();
  let amax = 1e-9;
  for (const s of series) for (const v of s) amax = Math.max(amax, Math.abs(v));
  const colors = ["#58a6ff", "#f0883e"];
  series.forEach((s, idx) => {
    if (!s.length) return;
    cctx.strokeStyle = colors[idx];
    cctx.lineWidth = 1.5;
    cctx.beginPath();
    for (let i = 0; i < s.length; i++) {
      const x = (i / (s.length - 1 || 1)) * w;
      const y = h / 2 - (s[i] / amax) * (h / 2 - 6);
      i === 0 ? cctx.moveTo(x, y) : cctx.lineTo(x, y);
    }
    cctx.stroke();
  });
  cctx.fillStyle = "#8b949e";
  cctx.font = "11px monospace";
  cctx.fillText("pressure  ±" + amax.toExponential(1) + " Pa", 6, 14);
}

function updateChart(header, arrays) {
  if (arrays.mic_ring) { drawChartFromRing(arrays.mic_ring); return; }
  if (arrays.mic_p) {
    const p = arrays.mic_p.data;
    const m = p.length;
    selMics = [0, Math.floor(m / 2)];
    pushTrace([p[selMics[0]] || 0, p[selMics[1]] || 0]);
    drawChart(traces);
    return;
  }
  if (arrays.sphere_p) {
    let rms = 0;
    for (const v of arrays.sphere_p.data) rms += v * v;
    rms = Math.sqrt(rms / arrays.sphere_p.data.length);
    pushTrace([rms, 0]);
    drawChart([traces[0]]);
  }
}

// --------------------------------------------------------------------------
// Frame application + replay buffer
// --------------------------------------------------------------------------
const frames = []; // {header, arrays}
let playhead = -1;
let live = true;

function applyFrame(f) {
  const { header, arrays } = f;
  updateSliceTexture(arrays.field_slice, header.slice_range);
  updateSpherePressure(arrays.sphere_p);
  updateVehicle(header);
  updateMeshes(header);
  updateChart(header, arrays);
  document.getElementById("t").textContent = (header.t ?? 0).toFixed(4);
  document.getElementById("frame").textContent = header.step ?? 0;
}

function onFrame(f) {
  frames.push(f);
  if (frames.length > MAX_BUFFER) frames.shift();
  const scrub = document.getElementById("scrub");
  scrub.max = frames.length - 1;
  document.getElementById("buf").textContent = frames.length;
  if (live) {
    playhead = frames.length - 1;
    scrub.value = playhead;
    applyFrame(f);
  }
}

// --------------------------------------------------------------------------
// UI controls
// --------------------------------------------------------------------------
const scrub = document.getElementById("scrub");
const scrublabel = document.getElementById("scrublabel");
document.getElementById("playpause").addEventListener("click", (e) => {
  live = false;
  playhead = Math.min(playhead + 1, frames.length - 1); // nudge; acts as step
  e.target.textContent = "⏸ Paused";
  applyAt(playhead);
});
document.getElementById("live").addEventListener("click", () => {
  live = true;
  scrublabel.textContent = "live";
  if (frames.length) { playhead = frames.length - 1; scrub.value = playhead; applyFrame(frames[playhead]); }
});
scrub.addEventListener("input", () => {
  live = false;
  applyAt(parseInt(scrub.value, 10));
});
function applyAt(i) {
  if (i < 0 || i >= frames.length) return;
  playhead = i;
  scrub.value = i;
  scrublabel.textContent = "frame " + i + " / " + (frames.length - 1);
  applyFrame(frames[i]);
}

// --------------------------------------------------------------------------
// WebSocket transport (with auto-reconnect)
// --------------------------------------------------------------------------
const dot = document.getElementById("dot");
const status = document.getElementById("status");
function setStatus(text, on) { status.textContent = text; dot.classList.toggle("on", !!on); }

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => setStatus("connected", true);
  ws.onclose = () => { setStatus("disconnected — retrying…", false); setTimeout(connect, 1500); };
  ws.onerror = () => ws.close();
  ws.onmessage = (ev) => {
    let msg;
    try { msg = decodeMessage(ev.data); } catch (err) { console.error(err); return; }
    if (msg.header.type === "scene") buildScene(msg.header, msg.arrays);
    else if (msg.header.type === "frame") onFrame(msg);
  };
}
connect();

// --------------------------------------------------------------------------
// Render loop
// --------------------------------------------------------------------------
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();
