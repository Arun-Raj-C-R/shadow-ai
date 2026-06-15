"""
Three.js WebGL Renderer — HTML Export
========================================
Generates standalone HTML files with interactive 3D CAD visualization.

Features:
    - PBR-style materials (metallic, roughness)
    - Orbit camera controls
    - Wireframe mode
    - Shaded mode with dynamic lighting
    - Edge highlighting
    - Grid and axes helpers
    - Viewport navigation
    - Responsive layout
"""

import json
import numpy as np
from .vector_math import Vec3


def render_to_html(mesh_data, title="CAD Model", subtitle="", stats=None, mode="shaded"):
    """
    Generate a self-contained HTML file using Three.js for 3D rendering.

    Parameters:
        mesh_data: dict with x, y, z, i, j, k arrays
        title: display title
        subtitle: secondary info line
        stats: dict of stat_label: stat_value pairs
        mode: 'shaded', 'wireframe', or 'both'

    Returns:
        HTML string
    """
    x = json.dumps(mesh_data.get('x', []))
    y = json.dumps(mesh_data.get('y', []))
    z = json.dumps(mesh_data.get('z', []))
    i_arr = json.dumps(mesh_data.get('i', []))
    j_arr = json.dumps(mesh_data.get('j', []))
    k_arr = json.dumps(mesh_data.get('k', []))
    n_verts = mesh_data.get('num_vertices', len(mesh_data.get('x', [])))
    n_faces = mesh_data.get('num_faces', len(mesh_data.get('i', [])))

    stats_html = ""
    if stats:
        for label, value in stats.items():
            stats_html += f'<div class="stat"><div class="stat-label">{label}</div><div class="stat-value">{value}</div></div>'
    else:
        stats_html = f'''
        <div class="stat"><div class="stat-label">Vertices</div><div class="stat-value">{n_verts}</div></div>
        <div class="stat"><div class="stat-label">Faces</div><div class="stat-value">{n_faces}</div></div>
        '''

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} | Shadow AI CAD Engine</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0a0a12;color:#e0e0e0;font-family:'Segoe UI',system-ui,-apple-system,sans-serif;overflow:hidden}}
#header{{position:fixed;top:0;left:0;right:0;z-index:100;padding:16px 24px;
  background:linear-gradient(180deg,rgba(8,8,20,0.95),rgba(8,8,20,0.7),transparent);
  pointer-events:none}}
#header>*{{pointer-events:auto}}
h1{{font-size:22px;font-weight:300;letter-spacing:2.5px;
  background:linear-gradient(90deg,#00d4ff,#00ff88,#7b2ff7);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.subtitle{{color:#666;font-size:12px;margin-top:3px;letter-spacing:1px}}
.stats{{display:flex;gap:12px;margin-top:10px;flex-wrap:wrap}}
.stat{{background:rgba(15,25,40,0.85);border:1px solid rgba(0,150,255,0.15);
  border-radius:8px;padding:8px 14px;min-width:110px;backdrop-filter:blur(10px)}}
.stat-label{{color:#556;font-size:10px;text-transform:uppercase;letter-spacing:1.2px}}
.stat-value{{color:#00d4ff;font-size:16px;font-weight:600;margin-top:2px}}
#controls{{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);z-index:100;
  display:flex;gap:8px}}
.btn{{background:rgba(15,25,40,0.9);border:1px solid rgba(0,150,255,0.25);
  border-radius:8px;padding:8px 16px;color:#aac;font-size:12px;cursor:pointer;
  transition:all 0.2s;letter-spacing:0.5px;backdrop-filter:blur(10px)}}
.btn:hover{{background:rgba(0,150,255,0.15);border-color:rgba(0,150,255,0.5);color:#fff}}
.btn.active{{background:rgba(0,150,255,0.25);border-color:#00d4ff;color:#00d4ff}}
#info{{position:fixed;bottom:20px;right:20px;color:#445;font-size:10px;z-index:100}}
canvas{{display:block}}
</style>
</head>
<body>
<div id="header">
  <h1>⚙ {title}</h1>
  <div class="subtitle">{subtitle or 'Shadow AI CAD Engine — First-Principles Computational Design'}</div>
  <div class="stats">{stats_html}</div>
</div>
<div id="controls">
  <button class="btn active" onclick="setMode('shaded')">Shaded</button>
  <button class="btn" onclick="setMode('wireframe')">Wireframe</button>
  <button class="btn" onclick="setMode('both')">Both</button>
  <button class="btn" onclick="toggleEdges()">Edges</button>
  <button class="btn" onclick="resetCamera()">Reset View</button>
  <button class="btn" onclick="toggleGrid()">Grid</button>
  <span id="live-dot" style="display:inline-flex;align-items:center;gap:6px;margin-left:16px;color:#0f0;font-size:11px;letter-spacing:1px;">
    <span style="width:8px;height:8px;border-radius:50%;background:#0f0;animation:pulse 1.5s infinite;display:inline-block"></span>LIVE
  </span>
</div>
<div id="info">Drag: rotate &middot; Scroll: zoom &middot; Right-drag: pan &middot; Auto-refreshes on edit</div>

<style>
@keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:0.3}} }}
</style>

<!-- Auto-refresh: polls file modification time every 2s -->
<script>
(function() {{
  let lastLen = document.documentElement.outerHTML.length;
  let lastCheck = Date.now();
  // Use fetch to check if file changed (same-origin file://)
  async function checkReload() {{
    try {{
      const resp = await fetch(window.location.href, {{cache: 'no-store'}});
      const text = await resp.text();
      if (text.length !== lastLen && Date.now() - lastCheck > 3000) {{
        lastLen = text.length;
        lastCheck = Date.now();
        window.location.reload();
      }}
    }} catch(e) {{}}
  }}
  setInterval(checkReload, 2000);
}})();
</script>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const X={x}, Y={y}, Z={z};
const I={i_arr}, J={j_arr}, K={k_arr};

let scene, camera, renderer, controls;
let solidMesh, wireMesh, edgeLines, gridHelper;
let showEdges = true, showGrid = true;

function init() {{
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0a12);
  scene.fog = new THREE.FogExp2(0x0a0a12, 0.002);

  // Camera
  camera = new THREE.PerspectiveCamera(50, window.innerWidth/window.innerHeight, 0.1, 10000);
  camera.position.set(50, 40, 50);

  // Renderer
  renderer = new THREE.WebGLRenderer({{antialias: true, alpha: true}});
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.shadowMap.enabled = true;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.2;
  document.body.appendChild(renderer.domElement);

  // Controls
  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.rotateSpeed = 0.6;

  // Lighting
  const ambient = new THREE.AmbientLight(0x334466, 0.6);
  scene.add(ambient);
  const dir1 = new THREE.DirectionalLight(0xffffff, 0.9);
  dir1.position.set(50, 80, 50);
  dir1.castShadow = true;
  scene.add(dir1);
  const dir2 = new THREE.DirectionalLight(0x4488cc, 0.4);
  dir2.position.set(-30, 40, -20);
  scene.add(dir2);
  const hemi = new THREE.HemisphereLight(0x446688, 0x112233, 0.3);
  scene.add(hemi);

  // Build geometry
  const geom = new THREE.BufferGeometry();
  const vertices = new Float32Array(X.length * 3);
  for (let v = 0; v < X.length; v++) {{
    vertices[v*3] = X[v]; vertices[v*3+1] = Z[v]; vertices[v*3+2] = Y[v];
  }}
  geom.setAttribute('position', new THREE.BufferAttribute(vertices, 3));

  const indices = [];
  for (let f = 0; f < I.length; f++) {{
    indices.push(I[f], J[f], K[f]);
  }}
  geom.setIndex(indices);
  geom.computeVertexNormals();

  // Solid material (PBR-style)
  const solidMat = new THREE.MeshStandardMaterial({{
    color: 0x5588bb, metalness: 0.3, roughness: 0.55,
    flatShading: false, side: THREE.DoubleSide
  }});
  solidMesh = new THREE.Mesh(geom, solidMat);
  solidMesh.castShadow = true;
  solidMesh.receiveShadow = true;
  scene.add(solidMesh);

  // Wireframe
  const wireMat = new THREE.MeshBasicMaterial({{
    color: 0x00ff88, wireframe: true, transparent: true, opacity: 0.3
  }});
  wireMesh = new THREE.Mesh(geom.clone(), wireMat);
  wireMesh.visible = false;
  scene.add(wireMesh);

  // Edges
  const edgeGeom = new THREE.EdgesGeometry(geom, 15);
  const edgeMat = new THREE.LineBasicMaterial({{color: 0x224466, linewidth: 1}});
  edgeLines = new THREE.LineSegments(edgeGeom, edgeMat);
  scene.add(edgeLines);

  // Grid
  gridHelper = new THREE.GridHelper(200, 40, 0x1a1a3a, 0x111128);
  scene.add(gridHelper);

  // Axes
  const axes = new THREE.AxesHelper(30);
  scene.add(axes);

  // Center camera on object
  geom.computeBoundingSphere();
  const center = geom.boundingSphere.center;
  const radius = geom.boundingSphere.radius;
  controls.target.copy(center);
  camera.position.set(center.x + radius*2, center.y + radius*1.5, center.z + radius*2);
  controls.update();

  window.addEventListener('resize', () => {{
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  }});

  animate();
}}

function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}

function setMode(mode) {{
  document.querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  solidMesh.visible = (mode === 'shaded' || mode === 'both');
  wireMesh.visible = (mode === 'wireframe' || mode === 'both');
  edgeLines.visible = showEdges && solidMesh.visible;
}}

function toggleEdges() {{
  showEdges = !showEdges;
  edgeLines.visible = showEdges && solidMesh.visible;
  event.target.classList.toggle('active');
}}

function toggleGrid() {{
  showGrid = !showGrid;
  gridHelper.visible = showGrid;
  event.target.classList.toggle('active');
}}

function resetCamera() {{
  const geom = solidMesh.geometry;
  geom.computeBoundingSphere();
  const c = geom.boundingSphere.center;
  const r = geom.boundingSphere.radius;
  camera.position.set(c.x + r*2, c.y + r*1.5, c.z + r*2);
  controls.target.copy(c);
  controls.update();
}}

init();
</script>
</body>
</html>'''
