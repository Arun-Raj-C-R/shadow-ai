"""
CAD Tool for SHADOW/Shadow AI â€” STATEFUL LIVE EDITING
========================================================
LLM-driven parametric CAD engine with persistent session memory.

KEY DESIGN:
    - Persistent state: the model is remembered between calls
    - Incremental edits: "change the base width" modifies the existing model
    - Single browser tab: same HTML file is overwritten + auto-refreshes
    - Full parametric history for undo/rollback

Runs as a tool callable by the SHADOW main AI.
"""

import os
import json
import traceback
import copy
from datetime import datetime

# â”€â”€ Output directory â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CAD_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "cad_runs")
os.makedirs(CAD_OUTPUT_DIR, exist_ok=True)

# â”€â”€ State file â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CAD_STATE_FILE = os.path.join(CAD_OUTPUT_DIR, "_session_state.json")
CAD_LIVE_HTML = os.path.join(CAD_OUTPUT_DIR, "_live_viewer.html")

# â”€â”€ Browser tab tracking â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_browser_opened = False


def _save_state(state):
    """Persist the current model definition to disk."""
    try:
        tmp = CAD_STATE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, default=str)
        if os.path.exists(CAD_STATE_FILE):
            os.remove(CAD_STATE_FILE)
        os.rename(tmp, CAD_STATE_FILE)
    except Exception as e:
        print(f"   [CAD State] Save error: {e}")


def _load_state():
    """Load the persisted model definition. Returns None if no state."""
    try:
        if not os.path.exists(CAD_STATE_FILE):
            return None
        with open(CAD_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _clear_state():
    """Clear the persisted state (start fresh)."""
    try:
        if os.path.exists(CAD_STATE_FILE):
            os.remove(CAD_STATE_FILE)
    except Exception:
        pass


def _default_state(part_name="Part", material=None):
    """Create a fresh empty state."""
    return {
        "part_name": part_name,
        "material": material,
        "primitives": [],
        "sketches": [],
        "boolean_ops": [],
        "edit_history": [],
        "created_at": datetime.now().isoformat(),
        "last_modified": datetime.now().isoformat(),
        "version": 0,
    }


def _rebuild_session_from_state(state):
    """Reconstruct a live CADSession from the persisted state definition."""
    from cad_engine.cad_engine import CADSession

    session = CADSession(
        state.get("part_name", "Part"),
        material=state.get("material"),
        output_dir=CAD_OUTPUT_DIR,
    )

    # Rebuild all primitives
    for prim in state.get("primitives", []):
        ptype = prim.get("type", "box").lower()
        if ptype == "box":
            session.add_box(
                prim.get("lx", 10), prim.get("ly", 10), prim.get("lz", 10),
                origin=prim.get("origin"), name=prim.get("name")
            )
        elif ptype == "cylinder":
            session.add_cylinder(
                prim.get("radius", 5), prim.get("height", 10),
                origin=prim.get("origin"),
                num_sides=prim.get("num_sides", 32),
                name=prim.get("name")
            )
        elif ptype == "sphere":
            session.add_sphere(
                prim.get("radius", 5), center=prim.get("center"),
                name=prim.get("name")
            )
        elif ptype == "cone":
            session.add_cone(
                prim.get("radius", 5), prim.get("height", 10),
                origin=prim.get("origin"), name=prim.get("name")
            )
        elif ptype == "torus":
            session.add_torus(
                prim.get("major_radius", 10), prim.get("minor_radius", 3),
                center=prim.get("center"), name=prim.get("name")
            )

    # Rebuild sketches
    for sk_data in state.get("sketches", []):
        sketch, sk_idx = session.new_sketch(
            plane_origin=sk_data.get("plane_origin"),
            plane_normal=sk_data.get("plane_normal"),
        )
        for entity in sk_data.get("entities", []):
            etype = entity.get("type", "line").lower()
            if etype == "line":
                sketch.add_line(entity["p1"], entity["p2"])
            elif etype == "rectangle":
                sketch.add_rectangle(entity.get("x", 0), entity.get("y", 0),
                                     entity.get("w", 10), entity.get("h", 5))
            elif etype == "circle":
                sketch.add_circle(entity.get("center", (0, 0)), entity.get("radius", 5))
            elif etype == "polygon":
                sketch.add_polygon(entity.get("center", (0, 0)),
                                   entity.get("radius", 5), entity.get("sides", 6))
            elif etype == "arc":
                sketch.add_arc(entity.get("center", (0, 0)), entity.get("radius", 5),
                               entity.get("start_angle", 0), entity.get("end_angle", 3.14))

        operation = sk_data.get("operation", "extrude")
        if operation == "extrude":
            session.extrude_sketch(sk_idx, sk_data.get("distance", 10))
        elif operation == "revolve":
            session.revolve_sketch(sk_idx,
                                   axis_origin=sk_data.get("axis_origin"),
                                   axis_direction=sk_data.get("axis_direction"),
                                   angle=sk_data.get("angle"))

    # Rebuild boolean operations
    for bop in state.get("boolean_ops", []):
        op = bop.get("op", "union").lower()
        a, b = bop.get("a", 0), bop.get("b", 1)
        try:
            if op == "union":
                session.boolean_union(a, b)
            elif op == "subtract":
                session.boolean_subtract(a, b)
            elif op == "intersect":
                session.boolean_intersect(a, b)
        except Exception as e:
            print(f"   [CAD] Boolean {op} ({a},{b}) skipped: {e}")

    return session


def _render_live(session):
    """Render to the fixed live HTML file. Only opens browser on first call."""
    global _browser_opened
    import webbrowser

    # Always write to the same file
    session.render_html(filepath=CAD_LIVE_HTML, open_browser=False)

    # Only open browser tab ONCE per process lifetime
    if not _browser_opened:
        webbrowser.open(f"file:///{CAD_LIVE_HTML.replace(os.sep, '/')}")
        _browser_opened = True
        print(f"   [CAD] Browser opened: {CAD_LIVE_HTML}")
    else:
        print(f"   [CAD] Live viewer updated (refresh browser tab)")


def run_cad_operation(
    task: str,
    part_name: str = None,
    primitives: list = None,
    sketch_data: dict = None,
    boolean_ops: list = None,
    export_formats: list = None,
    material: str = None,
    render: bool = True,
    assembly_data: dict = None,
    detailed_instructions: str = "",
    # â”€â”€ Incremental edit parameters â”€â”€
    modify_body: int = None,
    modify_params: dict = None,
    remove_body: int = None,
    add_primitives: list = None,
    **kwargs,
) -> str:
    """
    Main CAD tool entry point. Called by Shadow AI.

    STATEFUL â€” remembers the model between calls.

    Args:
        task: What to do. One of:
            - "create"           : Create a new part (clears old state)
            - "add"              : Add primitives to the EXISTING model
            - "modify"           : Modify a body in the existing model (change dimensions)
            - "remove"           : Remove a body from the model
            - "boolean"          : Boolean operations on existing bodies
            - "sketch_extrude"   : Create a sketch and extrude
            - "export"           : Export to file formats
            - "analyze"          : Analyze geometry (volume, area, etc.)
            - "show"             : Re-render current model (no changes)
            - "clear"            : Clear all state and start fresh
            - "list"             : List all bodies in the current model
            - "undo"             : Undo last edit
            - "validate"         : Run the validation test suite
            - "demo"             : Run a demonstration

        part_name:    Name for the part (only needed on "create")
        primitives:   List of dicts defining primitives
        modify_body:  Index of body to modify (for "modify" task)
        modify_params: New parameters for the body (e.g. {"lx": 20, "radius": 8})
        remove_body:  Index of body to remove (for "remove" task)
        add_primitives: Primitives to add to existing model (for "add" task)
        render:       Whether to update the 3D viewer

    Returns:
        JSON string with results including current model state summary
    """
    start_time = datetime.now()
    print(f"\n{'='*60}")
    print(f"  CAD ENGINE (Stateful Live)")
    print(f"   Task: {task}")
    if part_name:
        print(f"   Part: {part_name}")
    print(f"   Time: {start_time.strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")

    output = {
        "task": task,
        "timestamp": start_time.isoformat(),
    }

    try:
        # â”€â”€ Load or create state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        state = _load_state()

        if task == "validate":
            print("   Running CAD Engine validation suite...")
            from cad_engine.validation import run_all_tests
            results = run_all_tests(verbose=True)
            output["validation"] = results
            output["status"] = "success" if results['failed'] == 0 else "partial"
            output["message"] = f"{results['passed']}/{results['total']} tests passed"
            return json.dumps(output, indent=2, default=str)

        elif task == "clear":
            _clear_state()
            output["status"] = "success"
            output["message"] = "CAD session cleared. Ready for new model."
            return json.dumps(output, indent=2, default=str)

        elif task == "list":
            if not state or not state.get("primitives"):
                output["message"] = "No model loaded. Use task='create' to start."
                output["bodies"] = []
            else:
                bodies = []
                for idx, p in enumerate(state.get("primitives", [])):
                    bodies.append({"index": idx, "type": p.get("type"), "name": p.get("name", f"Body_{idx}"), "params": p})
                output["bodies"] = bodies
                output["part_name"] = state.get("part_name", "Part")
                output["message"] = f"{len(bodies)} bodies in model '{state.get('part_name', 'Part')}'"
            output["status"] = "success"
            return json.dumps(output, indent=2, default=str)

        elif task == "undo":
            if state and state.get("edit_history"):
                # Restore from the previous version
                prev = state["edit_history"].pop()
                state["primitives"] = prev.get("primitives", [])
                state["sketches"] = prev.get("sketches", [])
                state["boolean_ops"] = prev.get("boolean_ops", [])
                state["version"] = state.get("version", 1) - 1
                state["last_modified"] = datetime.now().isoformat()
                _save_state(state)
                # Rebuild and render
                session = _rebuild_session_from_state(state)
                if render:
                    _render_live(session)
                output["status"] = "success"
                output["message"] = f"Undo complete. Version: {state['version']}"
                output["summary"] = session.summary()
            else:
                output["status"] = "error"
                output["message"] = "Nothing to undo."
            return json.dumps(output, indent=2, default=str)

        # â”€â”€ CREATE: Start a new model from scratch â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        elif task in ("create", "primitives", "demo"):

            if task == "demo":
                part_name = part_name or "DemoPart"
                primitives = [
                    {"type": "box", "lx": 40, "ly": 30, "lz": 5, "name": "Base"},
                    {"type": "cylinder", "radius": 8, "height": 25, "origin": [20, 15, 5], "name": "Boss"},
                    {"type": "sphere", "radius": 6, "center": [10, 10, 15], "name": "Dome"},
                ]

            state = _default_state(part_name or "Part", material)

            if primitives:
                for prim in primitives:
                    state["primitives"].append(prim)
                    print(f"   [+] Adding {prim.get('type', 'box')}: {prim.get('name', '')}")

            if boolean_ops:
                state["boolean_ops"] = boolean_ops

            state["last_modified"] = datetime.now().isoformat()
            state["version"] = 1
            _save_state(state)

            # Build and render
            session = _rebuild_session_from_state(state)
            summary = session.summary()
            print(summary)

            if render:
                _render_live(session)

            output["status"] = "success"
            output["part_name"] = state["part_name"]
            output["analysis"] = session.analyze()
            output["summary"] = summary
            output["body_count"] = len(state["primitives"])
            output["message"] = f"Model '{state['part_name']}' created with {len(state['primitives'])} bodies. Browser updated."

        # â”€â”€ ADD: Add bodies to existing model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        elif task == "add":
            if not state:
                state = _default_state(part_name or "Part", material)

            # Save snapshot for undo
            state.setdefault("edit_history", []).append({
                "primitives": copy.deepcopy(state["primitives"]),
                "sketches": copy.deepcopy(state.get("sketches", [])),
                "boolean_ops": copy.deepcopy(state.get("boolean_ops", [])),
            })
            # Keep only last 20 undo steps
            if len(state["edit_history"]) > 20:
                state["edit_history"] = state["edit_history"][-20:]

            new_prims = add_primitives or primitives or []
            for prim in new_prims:
                state["primitives"].append(prim)
                print(f"   [+] Adding {prim.get('type', 'box')}: {prim.get('name', '')}")

            if boolean_ops:
                state["boolean_ops"].extend(boolean_ops)

            state["version"] = state.get("version", 0) + 1
            state["last_modified"] = datetime.now().isoformat()
            _save_state(state)

            session = _rebuild_session_from_state(state)
            summary = session.summary()
            print(summary)

            if render:
                _render_live(session)

            output["status"] = "success"
            output["analysis"] = session.analyze()
            output["summary"] = summary
            output["body_count"] = len(state["primitives"])
            output["message"] = f"Added {len(new_prims)} body(ies). Total: {len(state['primitives'])}. Browser updated."

        # â”€â”€ MODIFY: Change parameters of an existing body â”€â”€â”€â”€â”€â”€â”€â”€â”€
        elif task == "modify":
            if not state or not state.get("primitives"):
                output["status"] = "error"
                output["message"] = "No model loaded. Use task='create' first."
                return json.dumps(output, indent=2, default=str)

            idx = modify_body if modify_body is not None else 0
            if idx < 0 or idx >= len(state["primitives"]):
                output["status"] = "error"
                output["message"] = f"Body index {idx} out of range. Model has {len(state['primitives'])} bodies (0-{len(state['primitives'])-1})."
                return json.dumps(output, indent=2, default=str)

            # Save snapshot for undo
            state.setdefault("edit_history", []).append({
                "primitives": copy.deepcopy(state["primitives"]),
                "sketches": copy.deepcopy(state.get("sketches", [])),
                "boolean_ops": copy.deepcopy(state.get("boolean_ops", [])),
            })
            if len(state["edit_history"]) > 20:
                state["edit_history"] = state["edit_history"][-20:]

            old_params = copy.deepcopy(state["primitives"][idx])
            params = modify_params or {}
            for key, value in params.items():
                state["primitives"][idx][key] = value
                print(f"   [~] Body {idx} ({old_params.get('name', old_params.get('type'))}): {key} = {old_params.get(key)} -> {value}")

            state["version"] = state.get("version", 0) + 1
            state["last_modified"] = datetime.now().isoformat()
            _save_state(state)

            session = _rebuild_session_from_state(state)
            summary = session.summary()
            print(summary)

            if render:
                _render_live(session)

            output["status"] = "success"
            output["modified_body"] = idx
            output["old_params"] = old_params
            output["new_params"] = state["primitives"][idx]
            output["analysis"] = session.analyze()
            output["summary"] = summary
            output["message"] = f"Body {idx} modified. Changes: {params}. Browser updated."

        # â”€â”€ REMOVE: Delete a body from the model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        elif task == "remove":
            if not state or not state.get("primitives"):
                output["status"] = "error"
                output["message"] = "No model loaded."
                return json.dumps(output, indent=2, default=str)

            idx = remove_body if remove_body is not None else -1
            if idx < 0:
                idx = len(state["primitives"]) + idx
            if idx < 0 or idx >= len(state["primitives"]):
                output["status"] = "error"
                output["message"] = f"Body index {idx} out of range."
                return json.dumps(output, indent=2, default=str)

            # Save snapshot for undo
            state.setdefault("edit_history", []).append({
                "primitives": copy.deepcopy(state["primitives"]),
                "sketches": copy.deepcopy(state.get("sketches", [])),
                "boolean_ops": copy.deepcopy(state.get("boolean_ops", [])),
            })

            removed = state["primitives"].pop(idx)
            print(f"   [-] Removed body {idx}: {removed.get('name', removed.get('type'))}")

            state["version"] = state.get("version", 0) + 1
            state["last_modified"] = datetime.now().isoformat()
            _save_state(state)

            session = _rebuild_session_from_state(state)
            summary = session.summary()
            print(summary)

            if render:
                _render_live(session)

            output["status"] = "success"
            output["removed"] = removed
            output["analysis"] = session.analyze()
            output["summary"] = summary
            output["message"] = f"Removed body {idx} ({removed.get('name', removed.get('type'))}). {len(state['primitives'])} remaining."

        # â”€â”€ SKETCH EXTRUDE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        elif task == "sketch_extrude":
            if not state:
                state = _default_state(part_name or "Part", material)

            state.setdefault("edit_history", []).append({
                "primitives": copy.deepcopy(state["primitives"]),
                "sketches": copy.deepcopy(state.get("sketches", [])),
                "boolean_ops": copy.deepcopy(state.get("boolean_ops", [])),
            })

            if sketch_data:
                state.setdefault("sketches", []).append(sketch_data)

            state["version"] = state.get("version", 0) + 1
            state["last_modified"] = datetime.now().isoformat()
            _save_state(state)

            session = _rebuild_session_from_state(state)
            summary = session.summary()
            print(summary)

            if render:
                _render_live(session)

            output["status"] = "success"
            output["analysis"] = session.analyze()
            output["summary"] = summary
            output["message"] = "Sketch extruded. Browser updated."

        # â”€â”€ BOOLEAN â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        elif task == "boolean":
            if not state or not state.get("primitives"):
                output["status"] = "error"
                output["message"] = "No model loaded."
                return json.dumps(output, indent=2, default=str)

            state.setdefault("edit_history", []).append({
                "primitives": copy.deepcopy(state["primitives"]),
                "sketches": copy.deepcopy(state.get("sketches", [])),
                "boolean_ops": copy.deepcopy(state.get("boolean_ops", [])),
            })

            if boolean_ops:
                state.setdefault("boolean_ops", []).extend(boolean_ops)

            state["version"] = state.get("version", 0) + 1
            state["last_modified"] = datetime.now().isoformat()
            _save_state(state)

            session = _rebuild_session_from_state(state)
            summary = session.summary()
            print(summary)

            if render:
                _render_live(session)

            output["status"] = "success"
            output["analysis"] = session.analyze()
            output["summary"] = summary
            output["message"] = f"Boolean operations applied. Browser updated."

        # â”€â”€ SHOW: Just re-render current state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        elif task == "show":
            if not state:
                output["status"] = "error"
                output["message"] = "No model loaded. Use task='create' first."
                return json.dumps(output, indent=2, default=str)

            session = _rebuild_session_from_state(state)
            summary = session.summary()
            print(summary)

            if render:
                _render_live(session)

            output["status"] = "success"
            output["summary"] = summary
            output["analysis"] = session.analyze()
            output["message"] = "Model re-rendered. Browser updated."

        # â”€â”€ ANALYZE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        elif task == "analyze":
            if not state:
                # Create from provided primitives
                state = _default_state(part_name or "Part", material)
                if primitives:
                    state["primitives"] = primitives
                    _save_state(state)

            session = _rebuild_session_from_state(state)
            output["analysis"] = session.analyze()
            output["status"] = "success"
            output["message"] = f"Analysis complete for {len(state.get('primitives', []))} bodies."

        # â”€â”€ EXPORT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        elif task == "export":
            if not state:
                state = _default_state(part_name or "Part", material)
                if primitives:
                    state["primitives"] = primitives
                    _save_state(state)

            session = _rebuild_session_from_state(state)

            formats = export_formats or ["stl"]
            exports = {}
            for fmt in formats:
                fmt = fmt.lower()
                if fmt == "stl":
                    exports["stl"] = session.export_stl()
                elif fmt == "obj":
                    exports["obj"] = session.export_obj()
                elif fmt == "step":
                    exports["step"] = session.export_step()
                elif fmt == "dxf":
                    exports["dxf"] = session.export_dxf()
            output["exports"] = exports
            output["status"] = "success"
            output["message"] = f"Exported to: {', '.join(formats)}"

        else:
            output["error"] = f"Unknown task: {task}. Use: create, add, modify, remove, boolean, sketch_extrude, export, analyze, show, clear, list, undo, validate, demo"

    except Exception as e:
        output["error"] = str(e)
        output["traceback"] = traceback.format_exc()
        print(f"   CAD Error: {e}")
        traceback.print_exc()

    # â”€â”€ Finalize â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    elapsed = (datetime.now() - start_time).total_seconds()
    output["elapsed_seconds"] = round(elapsed, 2)
    if "status" not in output:
        output["status"] = "error" if "error" in output else "success"

    # Attach current model summary for AI context
    state = _load_state()
    if state:
        output["current_model"] = {
            "part_name": state.get("part_name"),
            "body_count": len(state.get("primitives", [])),
            "version": state.get("version", 0),
            "bodies": [
                {"index": idx, "type": p.get("type"), "name": p.get("name", f"Body_{idx}")}
                for idx, p in enumerate(state.get("primitives", []))
            ],
        }

    result_json = json.dumps(output, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  CAD ENGINE COMPLETE ({elapsed:.1f}s)")
    print(f"{'='*60}")

    return result_json
