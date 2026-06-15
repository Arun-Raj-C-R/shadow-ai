# crystal_viewer_tool.py
"""
Interactive 3D Crystal Structure Viewer for SHADOW/Shadow AI
============================================================
Fetches structures from Materials Project, generates interactive
3D HTML viewers using py3Dmol. Rotate, zoom, click atoms.

Dependencies: pip install py3Dmol mp-api pymatgen
"""

import os
import json
import webbrowser
import logging
import tempfile
from typing import Optional
from datetime import datetime

from dotenv import load_dotenv
config_env = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", ".env")
if os.path.exists(config_env):
    load_dotenv(config_env, override=True)
else:
    load_dotenv()

logger = logging.getLogger("crystal_viewer")

MP_API_KEY = os.environ.get("MP_API_KEY")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "crystal_views")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# â”€â”€ Element colors (CPK convention) â”€â”€
ELEMENT_COLORS = {
    "H": "#FFFFFF", "He": "#D9FFFF", "Li": "#CC80FF", "Be": "#C2FF00",
    "B": "#FFB5B5", "C": "#909090", "N": "#3050F8", "O": "#FF0D0D",
    "F": "#90E050", "Ne": "#B3E3F5", "Na": "#AB5CF2", "Mg": "#8AFF00",
    "Al": "#BFA6A6", "Si": "#F0C8A0", "P": "#FF8000", "S": "#FFFF30",
    "Cl": "#1FF01F", "Ar": "#80D1E3", "K": "#8F40D4", "Ca": "#3DFF00",
    "Ti": "#BFC2C7", "V": "#A6A6AB", "Cr": "#8A99C7", "Mn": "#9C7AC7",
    "Fe": "#E06633", "Co": "#F090A0", "Ni": "#50D050", "Cu": "#C88033",
    "Zn": "#7D80B0", "Ga": "#C28F8F", "Ge": "#668F8F", "As": "#BD80E3",
    "Se": "#FFA100", "Br": "#A62929", "Sr": "#00FF00", "Y": "#94FFFF",
    "Zr": "#94E0E0", "Nb": "#73C2C9", "Mo": "#54B5B5", "Ru": "#248F8F",
    "Rh": "#0A7D8C", "Pd": "#006985", "Ag": "#C0C0C0", "Cd": "#FFD98F",
    "In": "#A67573", "Sn": "#668080", "Sb": "#9E63B5", "Te": "#D47A00",
    "I": "#940094", "Cs": "#57178F", "Ba": "#00C900", "La": "#70D4FF",
    "Ce": "#FFFFC7", "Pb": "#575961", "Bi": "#9E4FB5", "Au": "#FFD123",
    "Pt": "#D0D0E0", "W": "#2194D6", "Ta": "#4DA6FF", "Hf": "#4DC2FF",
}

ELEMENT_RADII = {
    "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76,
    "N": 0.71, "O": 0.66, "F": 0.57, "Na": 1.66, "Mg": 1.41, "Al": 1.21,
    "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "K": 2.03, "Ca": 1.76,
    "Ti": 1.60, "V": 1.53, "Cr": 1.39, "Mn": 1.39, "Fe": 1.32, "Co": 1.26,
    "Ni": 1.24, "Cu": 1.32, "Zn": 1.22, "Ga": 1.22, "Ge": 1.20, "As": 1.19,
    "Se": 1.20, "Br": 1.20, "Sr": 2.15, "Zr": 1.75, "Mo": 1.54, "Ag": 1.45,
    "Sn": 1.39, "Ba": 2.22, "Au": 1.36, "Pb": 1.46, "Bi": 1.48,
}


def _get_structure(formula: str = None, material_id: str = None):
    """Fetch pymatgen Structure from Materials Project."""
    from mp_api.client import MPRester
    if not MP_API_KEY:
        raise ValueError("MP_API_KEY not set in .env")

    try:
        with MPRester(MP_API_KEY) as mpr:
            if material_id:
                raw_struct = mpr.get_structure_by_material_id(material_id)
                mid = material_id
            elif formula:
                docs = mpr.materials.summary.search(
                    formula=formula,
                    fields=["material_id", "energy_above_hull", "formula_pretty"],
                    energy_above_hull=(0, 0.05),
                )
                if not docs:
                    docs = mpr.materials.summary.search(
                        formula=formula,
                        fields=["material_id", "energy_above_hull", "formula_pretty"],
                    )
                if not docs:
                    raise ValueError(f"No materials found for '{formula}'")
                docs.sort(key=lambda d: d.energy_above_hull or 999)
                mid = str(docs[0].material_id)
                raw_struct = mpr.get_structure_by_material_id(mid)
            else:
                raise ValueError("Provide formula or material_id")
                
            from pymatgen.core import Structure
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            
            if isinstance(raw_struct, dict):
                struct = Structure.from_dict(raw_struct)
            else:
                struct = raw_struct
                
            try:
                sga = SpacegroupAnalyzer(struct)
                struct = sga.get_conventional_standard_structure()
            except Exception as e:
                logger.warning(f"Failed to convert to conventional cell: {e}")

        return struct, mid
    except Exception as e:
        err_msg = str(e)
        if "401" in err_msg or "authentication" in err_msg.lower() or "credential" in err_msg.lower():
            raise ValueError("Materials Project API key (MP_API_KEY) is invalid or has expired. Please go to https://materialsproject.org/api to get a new API key and update it in your .env file.") from e
        raise e


def _get_supercell(struct, min_atoms=20, max_atoms=200):
    """Make supercell for better visualization."""
    n = struct.num_sites
    if n >= min_atoms:
        return struct.copy()
    factor = max(1, int((min_atoms / n) ** (1/3)) + 1)
    factor = min(factor, 4)
    sc = struct.copy()
    sc.make_supercell([factor, factor, factor])
    if sc.num_sites > max_atoms:
        sc = struct.copy()
        f2 = max(1, factor - 1)
        sc.make_supercell([f2, f2, f2])
    return sc


def _build_html(struct, material_id: str, formula: str, info: dict = None) -> str:
    """Generate full interactive HTML with Three.js-style 3D viewer using py3Dmol."""
    cif_str = struct.to(fmt="cif")

    # Build info panel HTML
    info_html = ""
    if info:
        rows = "".join(
            f'<tr><td style="color:#888;padding:2px 8px 2px 0">{k}</td>'
            f'<td style="color:#fff">{v}</td></tr>'
            for k, v in info.items()
        )
        info_html = f'<table style="font-size:13px">{rows}</table>'

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>{formula} â€” {material_id} | SHADOW Crystal Viewer</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    background: #0a0a1a;
    font-family: 'Segoe UI', system-ui, sans-serif;
    color: #e0e0e0;
    overflow: hidden;
  }}
  #header {{
    position: fixed; top: 0; left: 0; right: 0; z-index: 100;
    background: linear-gradient(180deg, rgba(10,10,26,0.95) 0%, rgba(10,10,26,0.7) 80%, transparent 100%);
    padding: 16px 24px 32px;
  }}
  #header h1 {{
    font-size: 22px; font-weight: 300; letter-spacing: 2px;
    background: linear-gradient(90deg, #00d4ff, #7b2ff7, #ff6ec7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }}
  #header .subtitle {{ font-size: 13px; color: #666; margin-top: 4px; }}
  #info-panel {{
    position: fixed; top: 80px; right: 20px; z-index: 100;
    background: rgba(20,20,40,0.85); border: 1px solid rgba(100,100,255,0.15);
    border-radius: 12px; padding: 14px 18px; backdrop-filter: blur(10px);
    max-width: 320px;
  }}
  #viewer {{ width: 100vw; height: 100vh; }}
  #controls {{
    position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
    z-index: 100; display: flex; gap: 8px;
  }}
  #controls button {{
    background: rgba(30,30,60,0.8); color: #aaa; border: 1px solid rgba(100,100,255,0.2);
    padding: 8px 16px; border-radius: 8px; cursor: pointer; font-size: 12px;
    backdrop-filter: blur(5px); transition: all 0.2s;
  }}
  #controls button:hover {{ background: rgba(60,60,120,0.9); color: #fff; border-color: #7b2ff7; }}
  .glow {{ animation: pulse 3s ease-in-out infinite; }}
  @keyframes pulse {{ 0%,100% {{ opacity:0.7 }} 50% {{ opacity:1 }} }}
</style>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
</head>
<body>
<div id="header">
  <h1 class="glow">â¬¡ {formula} â€” {material_id}</h1>
  <div class="subtitle">SHADOW AI Crystal Viewer Â· Drag to rotate Â· Scroll to zoom Â· Click atoms</div>
</div>
<div id="info-panel">{info_html}</div>
<div id="viewer"></div>
<div id="controls">
  <button onclick="viewer.setStyle({{}}, {{stick:{{radius:0.15,colorscheme:'Jmol'}},sphere:{{scale:0.3,colorscheme:'Jmol'}}}});viewer.render()">Ball & Stick</button>
  <button onclick="viewer.setStyle({{}}, {{sphere:{{scale:0.6,colorscheme:'Jmol'}}}});viewer.render()">Space Fill</button>
  <button onclick="viewer.setStyle({{}}, {{stick:{{radius:0.12,colorscheme:'Jmol'}}}});viewer.render()">Wireframe</button>
  <button onclick="viewer.setStyle({{}}, {{cartoon:{{color:'spectrum'}}}});viewer.render()">Polyhedra</button>
  <button onclick="viewer.spin('y',1)">Spin</button>
  <button onclick="viewer.spin(false)">Stop</button>
  <button onclick="viewer.zoomTo();viewer.render()">Reset View</button>
</div>
<script>
var viewer = $3Dmol.createViewer("viewer", {{
  backgroundColor: "0x0a0a1a",
  antialias: true,
  id: "viewer"
}});

var cifData = `{cif_str}`;

viewer.addModel(cifData, "cif", {{doAssembly:true, normalizeAssembly:true,
  duplicateAssemblyAtoms:true}});
viewer.setStyle({{}}, {{
  stick: {{radius: 0.15, colorscheme: "Jmol"}},
  sphere: {{scale: 0.3, colorscheme: "Jmol"}}
}});
viewer.addUnitCell({{
  box: {{color: "white", opacity: 0.3}},
  alabel: "a", blabel: "b", clabel: "c",
  alabelstyle: {{fontColor: "white", fontSize: 14}},
  blabelstyle: {{fontColor: "white", fontSize: 14}},
  clabelstyle: {{fontColor: "white", fontSize: 14}}
}});
viewer.zoomTo();
viewer.spin("y", 0.5);
viewer.render();

// Click atom to show info
viewer.setClickable({{}}, true, function(atom) {{
  var label = atom.elem + " (" + atom.x.toFixed(2) + ", " + atom.y.toFixed(2) + ", " + atom.z.toFixed(2) + ")";
  viewer.removeAllLabels();
  viewer.addLabel(label, {{
    position: atom,
    backgroundColor: "rgba(20,20,50,0.8)",
    fontColor: "#00d4ff",
    fontSize: 14,
    borderRadius: 8
  }});
  viewer.render();
}});

// Dynamic Element Legend
setTimeout(function() {{
    var legend = document.createElement("div");
    legend.style.position = "fixed";
    legend.style.right = "20px";
    legend.style.bottom = "80px";
    legend.style.background = "rgba(20,20,40,0.85)";
    legend.style.padding = "10px 15px";
    legend.style.borderRadius = "10px";
    legend.style.border = "1px solid rgba(100,100,255,0.2)";
    legend.style.backdropFilter = "blur(10px)";
    legend.style.zIndex = "100";
    legend.style.maxHeight = "300px";
    legend.style.overflowY = "auto";
    
    var atoms = viewer.getModel().selectedAtoms({{}});
    var uniqueElements = [];
    atoms.forEach(function(a) {{ if(uniqueElements.indexOf(a.elem) === -1) uniqueElements.push(a.elem); }});
    uniqueElements.sort();
    
    var html = "<h4 style='margin:0 0 10px 0; color:#fff; border-bottom:1px solid #555; padding-bottom:5px; font-weight:normal;'>Elements</h4>";
    uniqueElements.forEach(function(elem) {{
        var color = $3Dmol.ElementColors[elem];
        var hex = color ? "#" + color.toString(16).padStart(6, '0') : "#ffffff";
        html += `<div style="display:flex; align-items:center; margin-bottom:6px;">
                   <div style="width:14px; height:14px; border-radius:50%; background:${{hex}}; margin-right:10px; border:1px solid #777;"></div>
                   <span style="font-size:14px; color:#e0e0e0; font-family:sans-serif;">${{elem}}</span>
                 </div>`;
    }});
    legend.innerHTML = html;
    document.body.appendChild(legend);
}}, 500);

</script>
</body></html>"""
    return html


def view_crystal_3d(formula: str = None, material_id: str = None,
                    supercell: bool = True) -> str:
    """
    Fetch crystal structure and open interactive 3D viewer in browser.
    Tony Stark style â€” rotate, zoom, click atoms, change view modes.
    """
    try:
        struct, mid = _get_structure(formula=formula, material_id=material_id)
        pretty = struct.composition.reduced_formula

        # Get properties for info panel
        info = {
            "Formula": pretty,
            "Material ID": mid,
            "Space Group": struct.get_space_group_info()[0] if hasattr(struct, 'get_space_group_info') else "N/A",
            "Crystal System": str(struct.get_crystal_system()) if hasattr(struct, 'get_crystal_system') else "N/A",
            "Atoms": struct.num_sites,
            "a": f"{struct.lattice.a:.3f} Ã…",
            "b": f"{struct.lattice.b:.3f} Ã…",
            "c": f"{struct.lattice.c:.3f} Ã…",
            "Î±": f"{struct.lattice.alpha:.1f}Â°",
            "Î²": f"{struct.lattice.beta:.1f}Â°",
            "Î³": f"{struct.lattice.gamma:.1f}Â°",
            "Volume": f"{struct.lattice.volume:.2f} Å³",
            "Density": f"{struct.density:.3f} g/cmÂ³",
        }

        # Optional supercell
        view_struct = _get_supercell(struct) if supercell else struct
        if supercell and view_struct.num_sites != struct.num_sites:
            info["Display"] = f"Supercell ({view_struct.num_sites} atoms)"

        # Generate HTML
        html = _build_html(view_struct, mid, pretty, info)

        # Save and open
        fname = f"{pretty}_{mid}_{datetime.now().strftime('%H%M%S')}.html"
        fpath = os.path.join(OUTPUT_DIR, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(html)

        webbrowser.open(f"file:///{fpath.replace(os.sep, '/')}")

        # Also save CIF
        cif_path = os.path.join(OUTPUT_DIR, f"{pretty}_{mid}.cif")
        struct.to(filename=cif_path)

        return json.dumps({
            "status": "success",
            "action": "view_crystal_3d",
            "formula": pretty,
            "material_id": mid,
            "html_path": fpath,
            "cif_path": cif_path,
            "space_group": info.get("Space Group"),
            "crystal_system": info.get("Crystal System"),
            "atoms": struct.num_sites,
            "lattice": {"a": struct.lattice.a, "b": struct.lattice.b, "c": struct.lattice.c},
            "density": struct.density,
            "message": f"3D viewer opened for {pretty} ({mid}). Rotate with mouse, scroll to zoom, click atoms for coordinates."
        })

    except Exception as e:
        logger.error(f"view_crystal_3d failed: {e}")
        return json.dumps({"error": str(e)})


def crystal_viewer_tool(action: str, **kwargs) -> str:
    """Unified entry point for crystal viewer."""
    if action == "view_3d":
        return view_crystal_3d(
            formula=kwargs.get("formula"),
            material_id=kwargs.get("material_id"),
            supercell=kwargs.get("supercell", True),
        )
    elif action == "get_cif":
        try:
            struct, mid = _get_structure(
                formula=kwargs.get("formula"),
                material_id=kwargs.get("material_id")
            )
            pretty = struct.composition.reduced_formula
            cif_path = os.path.join(OUTPUT_DIR, f"{pretty}_{mid}.cif")
            struct.to(filename=cif_path)
            return json.dumps({"status": "success", "cif_path": cif_path,
                              "formula": pretty, "material_id": mid})
        except Exception as e:
            return json.dumps({"error": str(e)})
    else:
        return json.dumps({"error": f"Unknown action: {action}",
                          "available": ["view_3d", "get_cif"]})
