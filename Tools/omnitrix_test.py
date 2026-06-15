"""
OMNITRIX BATTLE SIMULATION - Full System Diagnostic
Tests every tool, every API key, every workflow in the SHADOW/Shadow AI ecosystem.
"""
import os, sys, json, time, traceback
from pathlib import Path

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from dotenv import load_dotenv

# Load env
ENV_PATH = Path(__file__).parent.parent / "config" / ".env"
load_dotenv(ENV_PATH, override=True)

# Add Tools to path
TOOLS_DIR = Path(__file__).parent
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(TOOLS_DIR.parent / "core"))

results = []

def test(name, category, fn):
    """Run a test and record result."""
    start = time.time()
    try:
        result = fn()
        elapsed = round(time.time() - start, 2)
        status = "PASS" if result.get("ok") else "FAIL"
        results.append({
            "name": name, "category": category, "status": status,
            "time_s": elapsed, "detail": result.get("detail", ""),
            "error": result.get("error", ""),
        })
        icon = "âœ…" if status == "PASS" else "âŒ"
        print(f"  {icon} {name} ({elapsed}s) â€” {result.get('detail','')[:100]}")
    except Exception as e:
        elapsed = round(time.time() - start, 2)
        results.append({
            "name": name, "category": category, "status": "ERROR",
            "time_s": elapsed, "detail": "", "error": str(e)[:200],
        })
        print(f"  ðŸ’€ {name} ({elapsed}s) â€” EXCEPTION: {str(e)[:100]}")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 1. API KEY CHECKS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
print("\n" + "="*60)
print("ðŸ”‘ PHASE 1: API KEY VERIFICATION")
print("="*60)

REQUIRED_KEYS = {
    "GEMINI_API_KEY": "Core AI model",
    "GOOGLE_API_KEY": "Google services",
    "GOOGLE_API_KEY_LIVE": "Live session",
    "GOOGLE_API_KEY_DECISION": "Decision agent",
    "GOOGLE_API_KEY_MEMORY": "Memory agent",
    "MEMORY_API_KEY": "Memory system",
    "MP_API_KEY": "Materials Project",
    "MATERIALS_PROJECT_API_KEY": "Materials Project (alias)",
    "WOLFRAM_APP_ID": "Wolfram Alpha",
    "GROQ_API_KEY": "Groq LLM",
    "NVIDIA_API_KEY": "NVIDIA NIM",
    "CODESTRAL_API_KEY": "Codestral",
    "OPENROUTER_API_KEY": "OpenRouter",
    "META_API_KEY": "Meta CORE",
    "ELSEVIER_API_KEY": "Elsevier/Scopus",
    "SPRINGER_OPEN_ACCESS_KEY": "Springer",
    "SEMANTIC_SCHOLAR_KEY": "Semantic Scholar",
    "CORE_API_KEY": "CORE API",
}

for key, desc in REQUIRED_KEYS.items():
    def check_key(k=key, d=desc):
        val = os.environ.get(k, "")
        if not val:
            return {"ok": False, "detail": f"MISSING â€” {d}", "error": f"{k} not set"}
        if val.startswith("your_") or val == "":
            return {"ok": False, "detail": f"PLACEHOLDER â€” {d}", "error": f"{k} is placeholder"}
        return {"ok": True, "detail": f"SET ({len(val)} chars) â€” {d}"}
    test(key, "API_KEY", check_key)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 2. API CONNECTIVITY TESTS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
print("\n" + "="*60)
print("ðŸŒ PHASE 2: API CONNECTIVITY")
print("="*60)

def test_gemini_api():
    from google import genai
    key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=key)
    resp = client.models.generate_content(model="gemini-2.0-flash-lite", contents="Reply with just 'OK'")
    text = resp.text.strip()
    return {"ok": "OK" in text.upper(), "detail": f"Response: {text[:50]}"}

def test_materials_project_api():
    from mp_api.client import MPRester
    key = os.environ.get("MP_API_KEY")
    if not key:
        return {"ok": False, "error": "MP_API_KEY missing"}
    with MPRester(key) as mpr:
        docs = mpr.materials.summary.search(formula="Si", num_chunks=1, chunk_size=1)
        return {"ok": len(docs) > 0, "detail": f"Found {len(docs)} Si entries, first: {docs[0].material_id if docs else 'NONE'}"}

def test_wolfram_api():
    key = os.environ.get("WOLFRAM_APP_ID")
    if not key:
        return {"ok": False, "error": "WOLFRAM_APP_ID missing"}
    import urllib.request
    url = f"http://api.wolframalpha.com/v1/result?appid={key}&i=2%2B2"
    resp = urllib.request.urlopen(url, timeout=10).read().decode()
    return {"ok": "4" in resp, "detail": f"2+2 = {resp.strip()}"}

def test_groq_api():
    from groq import Groq
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return {"ok": False, "error": "GROQ_API_KEY missing"}
    client = Groq(api_key=key)
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role":"user","content":"Reply with just OK"}],
        max_tokens=5
    )
    text = resp.choices[0].message.content.strip()
    return {"ok": "OK" in text.upper(), "detail": f"Response: {text[:50]}"}

test("Gemini API", "CONNECTIVITY", test_gemini_api)
test("Materials Project API", "CONNECTIVITY", test_materials_project_api)
test("Wolfram Alpha API", "CONNECTIVITY", test_wolfram_api)
test("Groq API", "CONNECTIVITY", test_groq_api)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 3. TOOL UNIT TESTS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
print("\n" + "="*60)
print("ðŸ§ª PHASE 3: TOOL UNIT TESTS")
print("="*60)

# --- GNoME Tool ---
def test_gnome_status():
    from gnome_tool import run_gnome_tool
    r = json.loads(run_gnome_tool(task="status"))
    return {"ok": r.get("dataset_downloaded") == True, "detail": f"{r.get('total_materials',0)} materials loaded"}

def test_gnome_search():
    from gnome_tool import run_gnome_tool
    r = json.loads(run_gnome_tool(task="search", elements=["Si","C"], max_results=3))
    return {"ok": r.get("matches",0) > 0, "detail": f"{r.get('matches')} matches, provenance: {r.get('provenance','MISSING')[:40]}"}

def test_gnome_explore():
    from gnome_tool import run_gnome_tool
    r = json.loads(run_gnome_tool(task="explore_system", elements=["Ti","O"]))
    mode = r.get("search_mode", "unknown")
    return {"ok": r.get("total_matches",0) > 0, "detail": f"mode={mode}, {r.get('total_matches')} matches"}

def test_gnome_app():
    from gnome_tool import run_gnome_tool
    r = json.loads(run_gnome_tool(task="find_for_app", application="pcm", max_results=3))
    return {"ok": r.get("matches",0) > 0, "detail": f"{r.get('matches')} PCM candidates"}

def test_gnome_element_exact():
    """Verify 'S' does NOT match 'Sb', 'Si', 'Sr'."""
    from gnome_tool import run_gnome_tool
    r = json.loads(run_gnome_tool(task="search", elements=["S"], max_results=5))
    if r.get("matches",0) == 0:
        return {"ok": False, "detail": "No S matches found"}
    for m in r.get("results",[]):
        els = m.get("Elements","")
        if "'Sb'" in els and "'S'" not in els:
            return {"ok": False, "error": f"False match: S matched Sb in {els}"}
        if "'Si'" in els and "'S'" not in els:
            return {"ok": False, "error": f"False match: S matched Si in {els}"}
    return {"ok": True, "detail": f"All {len(r['results'])} results contain exact 'S'"}

test("GNoME Status", "GNOME", test_gnome_status)
test("GNoME Search (Si,C)", "GNOME", test_gnome_search)
test("GNoME Explore System (Ti-O)", "GNOME", test_gnome_explore)
test("GNoME Find for App (PCM)", "GNOME", test_gnome_app)
test("GNoME Element Exact Match (Sâ‰ Sb)", "GNOME", test_gnome_element_exact)

# --- Memory Tools ---
def test_memory_store():
    from memory_tools import run_store_logic
    r = run_store_logic("omnitrix_test: diagnostic ping at " + time.strftime("%H:%M:%S"))
    return {"ok": "stored" in str(r).lower() or "success" in str(r).lower() or "âœ…" in str(r), "detail": str(r)[:100]}

def test_memory_retrieve():
    from memory_tools import run_retrieve_logic
    r = run_retrieve_logic("omnitrix_test", "diagnostic")
    return {"ok": r is not None and len(str(r)) > 5, "detail": str(r)[:100]}

test("Memory Store", "MEMORY", test_memory_store)
test("Memory Retrieve", "MEMORY", test_memory_retrieve)

# --- Wolfram Tool ---
def test_wolfram_query():
    from wolfram_tool import simple_wolfram_query
    r = simple_wolfram_query("speed of light in m/s")
    return {"ok": "299" in str(r), "detail": str(r)[:100]}

test("Wolfram Query", "WOLFRAM", test_wolfram_query)

# --- ArXiv Tool ---
def test_arxiv_search():
    from arxiv_tool import search_arxiv
    r = search_arxiv("perovskite solar cell", max_results=2)
    return {"ok": len(str(r)) > 50, "detail": f"Response length: {len(str(r))} chars"}

test("ArXiv Search", "ARXIV", test_arxiv_search)

# --- Physics Agent ---
def test_physics_calc():
    from physics_agent_tool import run_physics_calculation
    r = run_physics_calculation("Calculate kinetic energy of 2kg mass at 3 m/s", use_visualization=False)
    return {"ok": "9" in str(r) or "joule" in str(r).lower() or len(str(r)) > 20, "detail": str(r)[:100]}

test("Physics Calculation", "PHYSICS", test_physics_calc)

# --- Search Tool ---
def test_web_search():
    from search import run_search
    r = run_search("perovskite solar cell efficiency 2024")
    return {"ok": len(str(r)) > 50, "detail": f"Response length: {len(str(r))} chars"}

test("Web Search", "SEARCH", test_web_search)

# --- System Stats ---
def test_system_stats():
    from system_stats_tool import get_system_stats
    r = get_system_stats()
    return {"ok": "cpu" in str(r).lower() or "memory" in str(r).lower(), "detail": str(r)[:100]}

test("System Stats", "SYSTEM", test_system_stats)

# --- Python REPL ---
def test_python_repl():
    from python_repl_tool import run_python_code
    r = run_python_code("print(2+2)", timeout=5)
    return {"ok": "4" in str(r), "detail": str(r)[:100]}

test("Python REPL", "SYSTEM", test_python_repl)

# --- PDF Reader ---
def test_pdf_reader():
    from pdf_reader_tool import extract_text_from_pdf
    # Just test import/function exists
    return {"ok": callable(extract_text_from_pdf), "detail": "Function exists and is callable"}

test("PDF Reader", "FILE", test_pdf_reader)

# --- File Downloader ---
def test_file_downloader():
    from file_downloader_tool import download_file_from_url
    return {"ok": callable(download_file_from_url), "detail": "Function exists and is callable"}

test("File Downloader", "FILE", test_file_downloader)

# --- Computational Chemistry ---
def test_comp_chem():
    from computational_chemistry_tool import run_computational_chemistry
    r = run_computational_chemistry(molecule="water", task="optimize")
    return {"ok": len(str(r)) > 20, "detail": str(r)[:100]}

test("Computational Chemistry", "CHEMISTRY", test_comp_chem)

# --- Organic Chemistry ---
def test_organic():
    from organic_chemistry_tool import fetch_organic_molecule
    r = fetch_organic_molecule("caffeine")
    return {"ok": len(str(r)) > 20, "detail": str(r)[:100]}

test("Organic Molecule Fetch", "CHEMISTRY", test_organic)

# --- Chemical Reaction ---
def test_chemical_reaction():
    from chemical_reaction_tool import analyze_chemical_reaction
    r = analyze_chemical_reaction("2H2 + O2 -> 2H2O")
    return {"ok": len(str(r)) > 10, "detail": str(r)[:100]}

test("Chemical Reaction", "CHEMISTRY", test_chemical_reaction)

# --- Perovskite Builder ---
def test_perovskite():
    from perovskite_builder_tool import build_complex_perovskite
    r = build_complex_perovskite(A_site="Cs", B_site="Pb", X_site="I")
    return {"ok": len(str(r)) > 20, "detail": str(r)[:100]}

test("Perovskite Builder", "CHEMISTRY", test_perovskite)

# --- Materials Project Tool ---
def test_mp_search():
    from materials_project_tool import MaterialsProjectTool
    mp = MaterialsProjectTool()
    r = mp.get_data({"formula": "Si"})
    return {"ok": isinstance(r, (dict, list)) and len(str(r)) > 20, "detail": str(r)[:100]}

test("Materials Project Search", "MATERIALS_PROJECT", test_mp_search)

# --- MP Advanced Queries ---
def test_mp_advanced():
    from materials_project_tool import search_mp_by_formula
    r = search_mp_by_formula("GaN")
    return {"ok": len(str(r)) > 20, "detail": str(r)[:100]}

test("MP Search by Formula", "MATERIALS_PROJECT", test_mp_advanced)

# --- DFT Tool ---
def test_dft_tool():
    from dft_tool import run_dft_calculation
    r = run_dft_calculation(task="full_analysis", formula="Si")
    return {"ok": len(str(r)) > 20, "detail": str(r)[:100]}

test("DFT Calculation", "DFT", test_dft_tool)

# --- MD Tool ---
def test_md_tool():
    from md_tool import run_md_simulation
    r = run_md_simulation(task="full_analysis", formula="Si", steps=10)
    return {"ok": len(str(r)) > 20, "detail": str(r)[:100]}

test("MD Simulation", "MD", test_md_tool)

# --- Crystal Viewer ---
def test_crystal_viewer():
    from crystal_viewer_tool import crystal_viewer_tool
    return {"ok": callable(crystal_viewer_tool), "detail": "Function exists and is callable"}

test("Crystal Viewer", "VISUALIZATION", test_crystal_viewer)

# --- LaTeX Renderer ---
def test_latex():
    from latex_renderer_tool import run_latex_renderer
    r = run_latex_renderer(task="render_only", latex_content=r"E = mc^2")
    return {"ok": len(str(r)) > 10, "detail": str(r)[:100]}

test("LaTeX Renderer", "VISUALIZATION", test_latex)

# --- ReadWrite Tool ---
def test_readwrite():
    from readwrite_tool import readwrite_tool
    r = readwrite_tool("list_documents")
    return {"ok": True, "detail": str(r)[:100]}

test("ReadWrite Tool", "FILE", test_readwrite)

# --- CAD Tool ---
def test_cad():
    from cad_tool import run_cad_operation
    return {"ok": callable(run_cad_operation), "detail": "Function exists and is callable"}

test("CAD Tool", "CAD", test_cad)

# --- Music Tool ---
def test_music():
    from music_tool import play_music
    return {"ok": callable(play_music), "detail": "Function exists and is callable"}

test("Music Tool", "MEDIA", test_music)

# --- Google Maps ---
def test_gmaps():
    from google_maps_tool import run_google_maps_tool
    return {"ok": callable(run_google_maps_tool), "detail": "Function exists and is callable"}

test("Google Maps Tool", "MAPS", test_gmaps)

# --- Self Evolve ---
def test_self_evolve():
    from self_evolve_tool import self_evolve_tool
    return {"ok": callable(self_evolve_tool), "detail": "Function exists and is callable"}

test("Self-Evolve Tool", "SYSTEM", test_self_evolve)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 4. GENERATE REPORT
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
print("\n" + "="*60)
print("ðŸ“Š PHASE 4: BENCHMARK REPORT")
print("="*60)

total = len(results)
passed = sum(1 for r in results if r["status"] == "PASS")
failed = sum(1 for r in results if r["status"] == "FAIL")
errors = sum(1 for r in results if r["status"] == "ERROR")

print(f"\n  Total Tests: {total}")
print(f"  âœ… PASSED:   {passed}")
print(f"  âŒ FAILED:   {failed}")
print(f"  ðŸ’€ ERRORS:   {errors}")
print(f"  Score:       {passed}/{total} ({round(passed/total*100,1)}%)")

# Save detailed results
report_path = TOOLS_DIR / "omnitrix_results.json"
with open(report_path, "w") as f:
    json.dump({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": {"total": total, "passed": passed, "failed": failed, "errors": errors,
                     "score_pct": round(passed/total*100,1)},
        "results": results,
    }, f, indent=2)

print(f"\n  ðŸ“ Detailed results: {report_path}")

# Print failures
if failed + errors > 0:
    print(f"\n{'='*60}")
    print("ðŸ”´ FAILURES & ERRORS:")
    print("="*60)
    for r in results:
        if r["status"] in ("FAIL", "ERROR"):
            print(f"  [{r['status']}] {r['name']} ({r['category']})")
            if r["error"]:
                print(f"         Error: {r['error'][:150]}")
            if r["detail"]:
                print(f"         Detail: {r['detail'][:150]}")
