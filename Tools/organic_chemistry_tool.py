import os
import json
import requests
import webbrowser
from datetime import datetime

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "organic_molecules")
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

def fetch_organic_molecule(molecule_name: str) -> str:
    """
    Fetches the 3D SDF structure of an organic molecule from PubChem.
    Uses RDKit for molecular intelligence (descriptors, SMILES, properties).
    Visualizes it using a 3D HTML viewer.
    """
    try:
        print(f"🧬 Querying PubChem for organic molecule: {molecule_name}...")
        
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{molecule_name}/cids/JSON"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return json.dumps({"error": f"Could not find '{molecule_name}' in PubChem. (Status: {r.status_code})"})
        
        cid = r.json()['IdentifierList']['CID'][0]
        print(f"   Found CID: {cid}")
        
        sdf_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/record/SDF/?record_type=3d"
        r_sdf = requests.get(sdf_url, timeout=10)
        
        if r_sdf.status_code != 200:
            sdf_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/record/SDF/?record_type=2d"
            r_sdf = requests.get(sdf_url, timeout=10)
            if r_sdf.status_code != 200:
                return json.dumps({"error": f"Failed to retrieve SDF structural data for {molecule_name}."})
            print("   Note: 3D structure unavailable. Using 2D structure.")
        
        sdf_data = r_sdf.text
        timestamp = datetime.now().strftime('%H%M%S')
        sdf_path = os.path.join(OUTPUT_DIR, f"{molecule_name}_{cid}_{timestamp}.sdf")
        with open(sdf_path, "w", encoding="utf-8") as f:
            f.write(sdf_data)
            
        # RDKit Molecular Intelligence
        rdkit_data = {}
        rdkit_html = ""
        if RDKIT_AVAILABLE:
            try:
                print("🧠 [RDKit] Analyzing molecular intelligence...")
                suppl = Chem.SDMolSupplier(sdf_path)
                mol = next(suppl)
                if mol is not None:
                    rdkit_data = {
                        "smiles": Chem.MolToSmiles(mol),
                        "formula": Chem.rdMolDescriptors.CalcMolFormula(mol),
                        "mol_weight": round(Descriptors.ExactMolWt(mol), 3),
                        "logP": round(Descriptors.MolLogP(mol), 3),
                        "tpsa": round(Descriptors.TPSA(mol), 3),
                        "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
                        "num_h_donors": Descriptors.NumHDonors(mol),
                        "num_h_acceptors": Descriptors.NumHAcceptors(mol),
                    }
                    rdkit_html = f"""
                    <div style="background:rgba(20,20,40,0.85); padding:15px; border-radius:10px; margin-top:20px; border:1px solid rgba(100,255,100,0.3);">
                        <h3 style="margin:0 0 10px 0; color:#00ffcc;">RDKit Intelligence</h3>
                        <p style="margin:5px 0;"><strong>Formula:</strong> {rdkit_data['formula']}</p>
                        <p style="margin:5px 0;"><strong>SMILES:</strong> {rdkit_data['smiles']}</p>
                        <p style="margin:5px 0;"><strong>Molecular Weight:</strong> {rdkit_data['mol_weight']} g/mol</p>
                        <p style="margin:5px 0;"><strong>LogP (Lipophilicity):</strong> {rdkit_data['logP']}</p>
                        <p style="margin:5px 0;"><strong>TPSA:</strong> {rdkit_data['tpsa']} Å²</p>
                    </div>
                    """
            except Exception as e:
                print(f"RDKit Error: {e}")
                
        html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>{molecule_name} - Organic Viewer</title>
<style>
  body {{ margin:0; overflow:hidden; background:#050510; color:#fff; font-family:sans-serif; }}
  #header {{ position:fixed; top:20px; left:20px; z-index:100; max-width:400px; }}
  #header h1 {{ margin:0; color:#00ffcc; text-transform: capitalize; font-size:32px; }}
</style>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
</head>
<body>
<div id="header">
  <h1>{molecule_name}</h1>
  <p style="color:#aaa;">PubChem CID: {cid} | Interactive 3D Viewer</p>
  {rdkit_html}
</div>
<div id="viewer" style="width: 100vw; height: 100vh;"></div>
<script>
  let viewer = $3Dmol.createViewer("viewer", {{backgroundColor: "0x050510"}});
  let sdfData = `{sdf_data}`;
  viewer.addModel(sdfData, "sdf");
  viewer.setStyle({{}}, {{stick: {{radius: 0.15}}, sphere: {{scale: 0.3}}}});
  viewer.zoomTo();
  viewer.spin("y", 0.5);
  viewer.render();
</script>
</body></html>"""

        html_path = os.path.join(OUTPUT_DIR, f"{molecule_name}_{cid}_{timestamp}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
            
        webbrowser.open(f"file:///{html_path.replace(os.sep, '/')}")
        
        return json.dumps({
            "status": "success",
            "molecule": molecule_name,
            "cid": cid,
            "rdkit_intelligence": rdkit_data,
            "message": "Successfully retrieved molecule and analyzed with RDKit."
        })
        
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch organic molecule: {str(e)}"})
