"""
ðŸ§¬ SELF-EVOLVING AI â€” ANTIGRAVITY TOOL
=======================================
Multi-agent system that allows the AI to identify limitations in its own code,
log them, and apply targeted improvements when asked.

SAFETY RULES:
- NEVER delete files or folders
- NEVER rewrite entire files â€” only patch specific line ranges
- ALWAYS create .bak backup before editing
- ALWAYS verify syntax after patching
- ALWAYS rollback if syntax check fails
- ONLY operate on D:\Project File\Shadow\Shadow\Brain\Shadow2\*.py
"""

import os
import json
import re
import shutil
import py_compile
import traceback
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# CONSTANTS
# ============================================================================
PROJECT_DIR = r"D:\Project File\Shadow\Shadow\Brain\Shadow2"
LOG_FILE = os.path.join(PROJECT_DIR, "self_evolution_log.txt")
NOTES_FILE = os.path.join(PROJECT_DIR, "evolution_notes.txt")
PLAN_FILE = os.path.join(PROJECT_DIR, "evolution_plan.json")
HISTORY_FILE = os.path.join(PROJECT_DIR, "evolution_history.json")

# Files that should NEVER be modified by the evolution system
PROTECTED_FILES = {"self_evolve_tool.py"}  # Don't let it modify itself

# Max lines to send to AI at once (to avoid token limits)
MAX_CHUNK_LINES = 400

# ============================================================================
# GROQ CLIENT
# ============================================================================
_groq_client = None

def _get_groq():
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set in .env")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


# ============================================================================
# ISSUE LOGGING â€” called by AI during conversation
# ============================================================================
def log_self_evolution_issue(issue: str, severity: str = "medium", file_hint: str = ""):
    """
    Log an issue/limitation that the AI identified in its own code.
    Called during normal conversation when the AI notices something wrong.
    """
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] SEVERITY={severity} FILE_HINT={file_hint}\n{issue}\n{'â”€'*60}\n"
        
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(entry)
        
        return f"âœ… Issue logged for self-evolution: {issue[:100]}..."
    except Exception as e:
        return f"âŒ Failed to log issue: {e}"


# ============================================================================
# SCANNER AGENT â€” reads Python files, takes notes
# ============================================================================
def _get_python_files():
    """Get all .py files in the project directory (not subdirectories)."""
    files = []
    for fname in os.listdir(PROJECT_DIR):
        if fname.endswith(".py") and fname not in PROTECTED_FILES:
            fpath = os.path.join(PROJECT_DIR, fname)
            if os.path.isfile(fpath):
                files.append(fpath)
    return sorted(files)


def _read_file_safe(filepath):
    """Read a file safely, return content or error message."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"ERROR READING FILE: {e}"


def _read_logged_issues():
    """Read all logged issues from the evolution log."""
    if not os.path.exists(LOG_FILE):
        return "No issues logged yet."
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "Could not read evolution log."


def _scanner_analyze_file(filepath, issues_text):
    """Scanner agent analyzes one Python file for potential improvements."""
    fname = os.path.basename(filepath)
    content = _read_file_safe(filepath)
    
    if content.startswith("ERROR"):
        return f"âš ï¸ Could not read {fname}: {content}"
    
    lines = content.split("\n")
    total_lines = len(lines)
    
    # For very large files, analyze in chunks
    notes = []
    chunk_size = MAX_CHUNK_LINES
    
    for start in range(0, total_lines, chunk_size):
        end = min(start + chunk_size, total_lines)
        chunk = "\n".join(f"{i+start+1}: {lines[i+start]}" for i in range(end - start))
        
        prompt = f"""You are a code review expert. Analyze this Python code chunk from file '{fname}' (lines {start+1}-{end} of {total_lines}).

KNOWN ISSUES TO ADDRESS:
{issues_text[:2000]}

CODE CHUNK:
```python
{chunk}
```

TASK: Identify ONLY real bugs, missing error handling, or improvements that match the logged issues.
DO NOT suggest style changes, renaming, or cosmetic fixes.
DO NOT suggest rewriting entire functions.

For each issue found, output EXACTLY this format:
ISSUE: [description]
FILE: {fname}
LINES: [start_line]-[end_line]
CURRENT_CODE: [the exact current code on those lines]
SUGGESTED_FIX: [the improved code for those specific lines only]
---

If no issues found in this chunk, output: NO_ISSUES_FOUND"""

        try:
            resp = _get_groq().chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1500
            )
            result = resp.choices[0].message.content.strip()
            if "NO_ISSUES_FOUND" not in result:
                notes.append(f"\n{'='*40}\nFILE: {fname} (lines {start+1}-{end})\n{'='*40}\n{result}")
        except Exception as e:
            notes.append(f"\nâš ï¸ Scanner error on {fname} chunk {start+1}-{end}: {e}")
    
    return "\n".join(notes) if notes else f"âœ… {fname}: No issues found"


def _run_scanner():
    """Run the Scanner agent across all Python files."""
    print("ðŸ” Scanner Agent: Reading all Python files...")
    
    issues_text = _read_logged_issues()
    py_files = _get_python_files()
    
    all_notes = [f"EVOLUTION SCAN â€” {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"]
    all_notes.append(f"Files to scan: {len(py_files)}\n")
    all_notes.append(f"Logged issues:\n{issues_text}\n")
    all_notes.append("=" * 60 + "\n")
    
    for fpath in py_files:
        fname = os.path.basename(fpath)
        print(f"  ðŸ“„ Scanning {fname}...")
        note = _scanner_analyze_file(fpath, issues_text)
        all_notes.append(note)
    
    # Save notes
    notes_content = "\n".join(all_notes)
    with open(NOTES_FILE, "w", encoding="utf-8") as f:
        f.write(notes_content)
    
    print(f"âœ… Scanner complete. Notes saved to {NOTES_FILE}")
    return notes_content


# ============================================================================
# ANALYZER AGENT â€” creates precise change plan from notes
# ============================================================================
def _run_analyzer(scanner_notes):
    """Analyzer agent creates a precise, actionable change plan."""
    print("ðŸ§  Analyzer Agent: Creating evolution plan...")
    
    # Truncate notes if too long for the AI
    if len(scanner_notes) > 8000:
        scanner_notes = scanner_notes[:8000] + "\n... (truncated)"
    
    prompt = f"""You are a senior software architect. Based on these code review notes, create a precise change plan.

SCANNER NOTES:
{scanner_notes}

CRITICAL RULES:
1. NEVER suggest deleting files or folders
2. NEVER suggest rewriting entire files
3. Only suggest changes to SPECIFIC line ranges
4. Each change must include the EXACT original code and the EXACT replacement
5. Changes must be minimal â€” fix only what's broken
6. Preserve ALL existing comments and docstrings
7. Do NOT change function signatures unless absolutely necessary
8. Maximum 5 changes per evolution cycle (prioritize by severity)

Output a JSON array of changes. Each change object must have:
- "file": filename (e.g. "code5.py")
- "description": what the change does
- "severity": "critical" | "high" | "medium" | "low"  
- "start_line": first line number to change
- "end_line": last line number to change
- "original_code": exact text of the original lines (with newlines)
- "new_code": exact text of the replacement lines (with newlines)

Output ONLY valid JSON. No markdown, no explanation.
If no changes needed, output: []"""

    try:
        resp = _get_groq().chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=3000
        )
        
        raw = resp.choices[0].message.content.strip()
        
        # Extract JSON from response
        # Try to find JSON array in the response
        json_match = re.search(r'\[.*\]', raw, re.DOTALL)
        if json_match:
            raw = json_match.group(0)
        
        changes = json.loads(raw)
        
        if not isinstance(changes, list):
            changes = []
        
        # Limit to 5 changes max per cycle
        changes = changes[:5]
        
        # Save plan
        plan = {
            "timestamp": datetime.now().isoformat(),
            "changes": changes,
            "total_changes": len(changes)
        }
        
        with open(PLAN_FILE, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)
        
        print(f"âœ… Analyzer complete. Plan has {len(changes)} changes.")
        return plan
        
    except json.JSONDecodeError as e:
        print(f"âš ï¸ Analyzer produced invalid JSON: {e}")
        return {"timestamp": datetime.now().isoformat(), "changes": [], "total_changes": 0}
    except Exception as e:
        print(f"âŒ Analyzer error: {e}")
        return {"timestamp": datetime.now().isoformat(), "changes": [], "total_changes": 0}


# ============================================================================
# SURGEON AGENT â€” applies precise line-range patches
# ============================================================================
def _create_backup(filepath):
    """Create a .bak backup of a file before modifying it."""
    bak_path = filepath + ".bak"
    try:
        shutil.copy2(filepath, bak_path)
        return bak_path
    except Exception as e:
        raise RuntimeError(f"Cannot create backup of {filepath}: {e}")


def _restore_backup(filepath):
    """Restore a file from its .bak backup."""
    bak_path = filepath + ".bak"
    if os.path.exists(bak_path):
        try:
            shutil.copy2(bak_path, filepath)
            print(f"  ðŸ”„ Restored {os.path.basename(filepath)} from backup")
            return True
        except Exception as e:
            print(f"  âŒ Could not restore backup: {e}")
            return False
    return False


def _verify_syntax(filepath):
    """Verify Python syntax of a file. Returns (ok, error_msg)."""
    try:
        py_compile.compile(filepath, doraise=True)
        return True, None
    except py_compile.PyCompileError as e:
        return False, str(e)


def _apply_single_patch(filepath, change):
    """
    Apply a single patch to a file by finding and replacing specific lines.
    Returns (success, message).
    """
    fname = os.path.basename(filepath)
    original_code = change.get("original_code", "")
    new_code = change.get("new_code", "")
    
    if not original_code or not new_code:
        return False, f"Empty original_code or new_code for {fname}"
    
    # Read the current file content
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return False, f"Cannot read {fname}: {e}"
    
    # Normalize line endings for comparison
    original_normalized = original_code.replace("\r\n", "\n").strip()
    content_normalized = content.replace("\r\n", "\n")
    
    # Find the original code in the file
    if original_normalized not in content_normalized:
        # Try with flexible whitespace matching
        # Remove trailing whitespace from each line for comparison
        orig_lines = [l.rstrip() for l in original_normalized.split("\n")]
        content_lines = [l.rstrip() for l in content_normalized.split("\n")]
        
        # Try to find the block of lines
        found_at = -1
        for i in range(len(content_lines) - len(orig_lines) + 1):
            match = True
            for j, orig_line in enumerate(orig_lines):
                if content_lines[i + j].rstrip() != orig_line:
                    match = False
                    break
            if match:
                found_at = i
                break
        
        if found_at == -1:
            return False, f"Could not find original code in {fname}. Code may have already changed."
        
        # Replace the lines
        new_lines = new_code.replace("\r\n", "\n").strip().split("\n")
        result_lines = content_normalized.split("\n")
        result_lines[found_at:found_at + len(orig_lines)] = new_lines
        new_content = "\n".join(result_lines)
    else:
        # Direct replacement
        new_content = content_normalized.replace(original_normalized, new_code.replace("\r\n", "\n").strip(), 1)
    
    # Write the patched content
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)
        return True, f"Patched {fname} successfully"
    except Exception as e:
        return False, f"Cannot write {fname}: {e}"


def _run_surgeon(plan):
    """Surgeon agent applies the planned changes with safety checks."""
    print("ðŸ”§ Surgeon Agent: Applying patches...")
    
    changes = plan.get("changes", [])
    if not changes:
        print("  â„¹ï¸ No changes to apply.")
        return {"applied": 0, "failed": 0, "skipped": 0, "details": []}
    
    results = {"applied": 0, "failed": 0, "skipped": 0, "details": []}
    
    for i, change in enumerate(changes):
        fname = change.get("file", "")
        description = change.get("description", "unknown change")
        
        print(f"\n  [{i+1}/{len(changes)}] {description}")
        print(f"    File: {fname}")
        
        # Validate file path
        if not fname.endswith(".py"):
            print(f"    âš ï¸ SKIPPED: Not a Python file")
            results["skipped"] += 1
            results["details"].append({"file": fname, "status": "skipped", "reason": "not a .py file"})
            continue
        
        filepath = os.path.join(PROJECT_DIR, fname)
        
        if not os.path.exists(filepath):
            print(f"    âš ï¸ SKIPPED: File does not exist")
            results["skipped"] += 1
            results["details"].append({"file": fname, "status": "skipped", "reason": "file not found"})
            continue
        
        if fname in PROTECTED_FILES:
            print(f"    âš ï¸ SKIPPED: Protected file")
            results["skipped"] += 1
            results["details"].append({"file": fname, "status": "skipped", "reason": "protected"})
            continue
        
        # Safety: Ensure we're in the right directory
        real_path = os.path.realpath(filepath)
        if not real_path.startswith(os.path.realpath(PROJECT_DIR)):
            print(f"    âŒ BLOCKED: File is outside project directory!")
            results["skipped"] += 1
            results["details"].append({"file": fname, "status": "blocked", "reason": "outside project dir"})
            continue
        
        # Step 1: Create backup
        try:
            bak_path = _create_backup(filepath)
            print(f"    ðŸ’¾ Backup created: {os.path.basename(bak_path)}")
        except Exception as e:
            print(f"    âŒ Cannot backup, skipping: {e}")
            results["failed"] += 1
            results["details"].append({"file": fname, "status": "failed", "reason": f"backup failed: {e}"})
            continue
        
        # Step 2: Apply the patch
        success, msg = _apply_single_patch(filepath, change)
        
        if not success:
            print(f"    âŒ Patch failed: {msg}")
            _restore_backup(filepath)
            results["failed"] += 1
            results["details"].append({"file": fname, "status": "failed", "reason": msg})
            continue
        
        # Step 3: Verify syntax
        syntax_ok, syntax_err = _verify_syntax(filepath)
        
        if not syntax_ok:
            print(f"    âŒ Syntax error after patch! Rolling back...")
            print(f"    Error: {syntax_err}")
            _restore_backup(filepath)
            results["failed"] += 1
            results["details"].append({"file": fname, "status": "rolled_back", "reason": f"syntax error: {syntax_err}"})
            continue
        
        print(f"    âœ… Applied and verified: {msg}")
        results["applied"] += 1
        results["details"].append({"file": fname, "status": "applied", "description": description})
    
    # Clean up .bak files for successful patches (keep failed ones)
    for detail in results["details"]:
        if detail["status"] == "applied":
            bak = os.path.join(PROJECT_DIR, detail["file"] + ".bak")
            if os.path.exists(bak):
                try:
                    os.remove(bak)
                except Exception:
                    pass
    
    return results


# ============================================================================
# HISTORY TRACKING
# ============================================================================
def _save_evolution_history(scan_summary, plan, results):
    """Save this evolution cycle to history."""
    try:
        history = []
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "changes_planned": plan.get("total_changes", 0),
            "changes_applied": results.get("applied", 0),
            "changes_failed": results.get("failed", 0),
            "changes_skipped": results.get("skipped", 0),
            "details": results.get("details", [])
        }
        
        history.append(entry)
        
        # Keep last 50 evolution cycles
        history = history[-50:]
        
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        print(f"âš ï¸ Could not save evolution history: {e}")


# ============================================================================
# MAIN TOOL ENTRY POINTS
# ============================================================================
def self_evolve_tool(action="evolve", **kwargs):
    """
    Main entry point for the self-evolution tool.
    
    Actions:
      - evolve: Run full evolution cycle (scan â†’ analyze â†’ patch)
      - scan_only: Only scan and take notes, don't apply changes
      - show_log: Show currently logged issues
      - show_history: Show evolution history
      - clear_log: Clear the issue log after evolution
    """
    try:
        if action == "evolve":
            return _full_evolution_cycle()
        elif action == "scan_only":
            notes = _run_scanner()
            return f"âœ… Scan complete. Notes:\n{notes[:3000]}"
        elif action == "show_log":
            issues = _read_logged_issues()
            return f"ðŸ“‹ Current issues:\n{issues[:3000]}"
        elif action == "show_history":
            return _show_history()
        elif action == "clear_log":
            if os.path.exists(LOG_FILE):
                os.remove(LOG_FILE)
            return "âœ… Evolution log cleared."
        else:
            return f"Unknown action: {action}. Use: evolve, scan_only, show_log, show_history, clear_log"
    except Exception as e:
        return f"âŒ Self-evolution error: {e}\n{traceback.format_exc()}"


def _full_evolution_cycle():
    """Run the complete evolution cycle: scan â†’ analyze â†’ patch."""
    print("\n" + "=" * 60)
    print("ðŸ§¬ SELF-EVOLUTION CYCLE STARTING")
    print("=" * 60)
    
    # Phase 1: Scanner
    scanner_notes = _run_scanner()
    
    if "No issues found" in scanner_notes and "NO_ISSUES_FOUND" in scanner_notes:
        return "âœ… Scanner found no issues. Code is healthy!"
    
    # Phase 2: Analyzer
    plan = _run_analyzer(scanner_notes)
    
    if not plan.get("changes"):
        return "âœ… Analyzer determined no changes are needed."
    
    # Phase 3: Surgeon
    results = _run_surgeon(plan)
    
    # Save history
    _save_evolution_history(scanner_notes[:500], plan, results)
    
    # Build report
    report = []
    report.append(f"ðŸ§¬ EVOLUTION CYCLE COMPLETE")
    report.append(f"  Applied: {results['applied']}")
    report.append(f"  Failed:  {results['failed']}")
    report.append(f"  Skipped: {results['skipped']}")
    
    for detail in results["details"]:
        status_icon = {"applied": "âœ…", "failed": "âŒ", "skipped": "âš ï¸", "rolled_back": "ðŸ”„", "blocked": "ðŸš«"}.get(detail["status"], "â“")
        report.append(f"  {status_icon} {detail['file']}: {detail.get('description', detail.get('reason', ''))}")
    
    return "\n".join(report)


def _show_history():
    """Show evolution history."""
    if not os.path.exists(HISTORY_FILE):
        return "No evolution history yet."
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
        
        if not history:
            return "No evolution history yet."
        
        lines = ["ðŸ“Š EVOLUTION HISTORY (last 10):"]
        for entry in history[-10:]:
            ts = entry.get("timestamp", "?")[:19]
            applied = entry.get("changes_applied", 0)
            failed = entry.get("changes_failed", 0)
            lines.append(f"  [{ts}] Applied: {applied}, Failed: {failed}")
        
        return "\n".join(lines)
    except Exception:
        return "Could not read evolution history."
