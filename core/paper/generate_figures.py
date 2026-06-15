# -*- coding: utf-8 -*-
"""
Aetheris Paper Figures Generator
================================
Generates publication-quality schematic diagrams and visual plots for the Aetheris paper.
Saves PNG files to d:\\Project File\\Shadow\\Shadow\\Brain\\Shadow2\\cli\\core\\paper\\figures\\
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path as MPath

# Set up output directory
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Use a clean, modern aesthetic style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'DejaVu Sans', 'Arial', 'Helvetica'],
    'text.color': '#f8fafc',
    'axes.labelcolor': '#f8fafc',
    'xtick.color': '#94a3b8',
    'ytick.color': '#94a3b8',
    'figure.facecolor': '#0f172a',
    'axes.facecolor': '#0f172a'
})


def add_drop_shadow(ax, rect_coords, zorder=1):
    """Draws a dark soft drop shadow for boxes on a dark canvas."""
    x, y, w, h = rect_coords
    shadow_offset_x = 0.04
    shadow_offset_y = -0.04
    shadow = patches.FancyBboxPatch(
        (x + shadow_offset_x, y + shadow_offset_y), w, h,
        boxstyle="round,pad=0.02", linewidth=0,
        facecolor='#020617', alpha=0.6, zorder=zorder
    )
    ax.add_patch(shadow)


def draw_glow_rect(ax, x, y, w, h, bg_color, border_color, label, text_color='#f8fafc', font_size=8, zorder=5):
    """Draws a rounded rectangular node with drop shadows and a glowing outer border."""
    # 1. Drop shadow (dark slate black)
    shadow = patches.FancyBboxPatch(
        (x + 0.05, y - 0.05), w, h,
        boxstyle="round,pad=0.03", linewidth=0,
        facecolor='#020617', alpha=0.6, zorder=zorder-2
    )
    ax.add_patch(shadow)

    # 2. Outer glowing ring (wider, semi-transparent border color)
    glow = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04", linewidth=3,
        edgecolor=border_color, facecolor='none', alpha=0.3, zorder=zorder-1
    )
    ax.add_patch(glow)

    # 3. Main Box
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02", linewidth=1.5,
        edgecolor=border_color, facecolor=bg_color, zorder=zorder
    )
    ax.add_patch(rect)

    # 4. Text Label
    ax.text(
        x + w/2, y + h/2, label,
        color=text_color, fontweight='bold',
        ha='center', va='center', fontsize=font_size, zorder=zorder+1
    )


def draw_bezier_curve(ax, start, end, control_offset=(0, 1.5), color='#6366f1', alpha=0.7, linewidth=1.5, arrow=True):
    """Draws a smooth quadratic Bezier curve between two points, optionally adding an arrowhead."""
    x1, y1 = start
    x2, y2 = end
    
    # Quadratic bezier control point
    cx = (x1 + x2)/2 + control_offset[0]
    cy = (y1 + y2)/2 + control_offset[1]
    
    verts = [
        (x1, y1),  # Start
        (cx, cy),  # Control
        (x2, y2)   # End
    ]
    codes = [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3]
    path = MPath(verts, codes)
    
    # Draw path
    patch = patches.PathPatch(path, facecolor='none', edgecolor=color, linewidth=linewidth, alpha=alpha, zorder=2)
    ax.add_patch(patch)
    
    if arrow:
        # Approximate tangent vector at the end of bezier
        dx = x2 - cx
        dy = y2 - cy
        length = np.hypot(dx, dy)
        if length > 0:
            dx, dy = dx/length, dy/length
            # Draw arrowhead pointing towards the end
            ax.annotate('', xy=(x2, y2), xytext=(x2 - 0.15*dx, y2 - 0.15*dy),
                        arrowprops=dict(arrowstyle="-|>", color=color, linewidth=linewidth, 
                                        mutation_scale=12, patchB=patch, shrinkA=0, shrinkB=0), zorder=3)


def draw_academic_rect(ax, x, y, w, h, bg_color, border_color, label, text_color='#0f172a', font_size=8, zorder=5):
    """Draws a clean, publication-ready light-themed node with soft light drop shadows."""
    shadow = patches.FancyBboxPatch(
        (x + 0.04, y - 0.04), w, h,
        boxstyle="round,pad=0.02", linewidth=0,
        facecolor='#cbd5e1', alpha=0.4, zorder=zorder-2
    )
    ax.add_patch(shadow)

    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02", linewidth=1.2,
        edgecolor=border_color, facecolor=bg_color, zorder=zorder
    )
    ax.add_patch(rect)

    ax.text(
        x + w/2, y + h/2, label,
        color=text_color, fontweight='bold',
        ha='center', va='center', fontsize=font_size, zorder=zorder+1
    )


def draw_live_architecture():
    """Generates Figure 1: Real-Time Multimodal Function Calling Loop (Stunning Light Mode/Academic)."""
    fig, ax = plt.subplots(figsize=(12.5, 9.0), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 9.5)
    ax.axis('off')

    # Main Canvas Frame
    canvas = patches.FancyBboxPatch((0.15, 0.15), 12.2, 9.2, boxstyle="round,pad=0.05", linewidth=1.5, edgecolor='#cbd5e1', facecolor='#ffffff', zorder=0)
    ax.add_patch(canvas)
    
    # Very faint grid dots for print texture
    for x in np.linspace(0.5, 12.0, 18):
        for y in np.linspace(0.5, 9.0, 14):
            ax.scatter(x, y, color='#cbd5e1', s=2, alpha=0.2, zorder=1)

    # Header Panel (Dark Slate text)
    ax.text(6.25, 9.0, "Aetheris Live Multimodal Cognitive Engine & Tool Orchestrator", fontsize=15, fontweight='bold', color='#0f172a', ha='center', zorder=2)
    ax.text(6.25, 8.7, "Real-Time WebSocket Streaming & Asynchronous Function-Calling Architecture", fontsize=10, color='#475569', ha='center', style='italic', zorder=2)

    # Core Orchestration Center Node
    draw_academic_rect(ax, 5.0, 4.3, 2.5, 2.8, '#f8fafc', '#0f172a', 
                       "AETHERIS CORE\n\n- Main Async Loop\n- Session Context\n- Homeostasis State\n  (Dopamine/Cortisol)\n- Dispatch Bus Router", 
                       text_color='#0f172a', font_size=8.5, zorder=10)

    # Left Column: User Input & Streaming Ingestion
    draw_academic_rect(ax, 0.6, 5.5, 2.2, 1.4, '#eff6ff', '#2563eb', 
                       "USER CLIENT\n\n- 16kHz PCM Mic Input\n- 1fps Frame Grabber\n- Low-latency Audio Out", 
                       text_color='#1e40af', font_size=8.0, zorder=10)

    draw_academic_rect(ax, 0.6, 3.4, 2.2, 1.4, '#f1f5f9', '#475569', 
                       "WEBSOCKET ENGINE\n\n- PyAudio I/O Loop\n- Image JPEG Compressor\n- Chunk Streamer", 
                       text_color='#334155', font_size=8.0, zorder=10)

    # Right Column: Gemini Live API Model
    draw_academic_rect(ax, 9.7, 4.5, 2.2, 2.2, '#faf5ff', '#7c3aed', 
                       "MULTIMODAL AI API\n(Gemini 2.5 / 3.1 Live)\n\n- Spatial Video Attention\n- Audio Tone Parser\n- Context Core\n- Structured JSON\n  Function Calling", 
                       text_color='#5b21b6', font_size=8.0, zorder=10)

    # Bottom Row: Scientific Tool Components
    tools = {
        'dft': (0.6, 0.8, 1.4, 1.2, "DFT SOLVER\n\n- Obara-Saika\n- ERI Integrals\n- DIIS SCF Loop", '#f0f9ff', '#0284c7', '#0369a1'),
        'cad': (2.5, 0.8, 1.4, 1.2, "PARAMETRIC CAD\n\n- B-Rep Kernel\n- Sketch Solver\n- CSG Operations", '#f0fdf4', '#16a34a', '#15803d'),
        'sim': (4.4, 0.8, 1.4, 1.2, "PHYSICS SIMS\n\n- Yee Grid FDTD\n- FEM Heat/Laplace\n- TCAD Solver", '#ecfdf5', '#059669', '#047857'),
        'db': (6.3, 0.8, 1.4, 1.2, "MATERIALS DB\n\n- GNoME Engine\n- Materials Project\n- PubChem Query", '#fffbeb', '#d97706', '#b45309'),
        'mem': (8.2, 0.8, 1.4, 1.2, "RAG MEMORY\n\n- ChromaDB Vector\n- Local Keyword\n- SPR Compression", '#fdf2f8', '#db2777', '#9d174d'),
        'evolve': (10.1, 0.8, 1.4, 1.2, "SELF-EVOLVE\n\n- Researcher Agent\n- Surgeon Patches\n- Reviewer Tests", '#fef2f2', '#dc2626', '#991b1b')
    }

    for name, (tx, ty, tw, th, label, bg, border, text_color) in tools.items():
        draw_academic_rect(ax, tx, ty, tw, th, bg, border, label, text_color=text_color, font_size=7.5, zorder=8)

    # Link Connections with Bezier Curves & Tangent Pointers
    # 1. User Client -> Websocket
    draw_bezier_curve(ax, (1.7, 5.5), (1.7, 4.8), control_offset=(-0.15, 0), color='#2563eb', arrow=True)
    # 2. Websocket -> Core Orchestrator
    draw_bezier_curve(ax, (2.8, 4.1), (5.0, 5.0), control_offset=(0, 0.4), color='#2563eb', arrow=True)
    ax.text(3.7, 4.8, "Raw Stream\n(Audio/Frame)", color='#1e40af', fontweight='bold', fontsize=7, ha='center')

    # 3. Core Orchestrator -> User Client (Audio Out Response)
    draw_bezier_curve(ax, (5.0, 5.7), (2.8, 6.2), control_offset=(0, 0.3), color='#16a34a', arrow=True)
    ax.text(3.9, 6.1, "Response Audio\n(Chunked PCM)", color='#15803d', fontweight='bold', fontsize=7, ha='center')

    # 4. Core Orchestrator <-> Gemini Live API
    draw_bezier_curve(ax, (7.5, 6.0), (9.7, 6.0), control_offset=(0, 0.25), color='#7c3aed', arrow=True)
    ax.text(8.6, 6.3, "WebSocket Session\n(Context Frames)", color='#5b21b6', fontweight='bold', fontsize=7, ha='center')
    
    draw_bezier_curve(ax, (9.7, 5.3), (7.5, 5.3), control_offset=(0, -0.25), color='#6d28d9', arrow=True)
    ax.text(8.6, 4.8, "Structured Tool Calls\n(JSON Command)", color='#6d28d9', fontweight='bold', fontsize=7, ha='center')

    # 5. Core Orchestrator to Scientific Tool Bus
    for name, (tx, ty, tw, th, _, _, border, _) in tools.items():
        start_pt = (6.25, 4.3)
        end_pt = (tx + tw/2, ty + th)
        
        # Quadratic curve sweeping down from Orchestrator base
        draw_bezier_curve(ax, start_pt, end_pt, control_offset=(0, -0.8), color='#64748b', alpha=0.5, linewidth=1.2, arrow=True)
        ax.scatter(end_pt[0], end_pt[1], color=border, s=15, zorder=12)

    # Tool Bus Label pill (Light background)
    pill = patches.FancyBboxPatch((4.5, 2.5), 3.5, 0.4, boxstyle="round,pad=0.05", linewidth=1.0, edgecolor='#94a3b8', facecolor='#f1f5f9', zorder=6)
    ax.add_patch(pill)
    ax.text(6.25, 2.7, "Structured JSON Tool Dispatch Bus", color='#475569', fontweight='bold', fontsize=8, ha='center', zorder=7)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "live_architecture.png"), dpi=300, facecolor='#ffffff', edgecolor='none')
    plt.close(fig)
    print("[PASS] Generated: live_architecture.png")


def draw_yee_grid():
    """Generates Figure 5: 2D FDTD Yee Grid Discretization Schema (Futuristic Dark Mode)."""
    fig, ax = plt.subplots(figsize=(8, 7.5), dpi=300)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_aspect('equal')
    
    # Draw grid lines
    for i in range(4):
        ax.plot([i, i], [-0.5, 3.5], color='#334155', linestyle='--', linewidth=1, zorder=1)
        ax.plot([-0.5, 3.5], [i, i], color='#334155', linestyle='--', linewidth=1, zorder=1)

    # Draw Yee Cell highlights
    rect = patches.Rectangle((1, 1), 1, 1, linewidth=2, edgecolor='#ef4444', facecolor='#7f1d1d', alpha=0.3, label='Yee Unit Cell', zorder=2)
    ax.add_patch(rect)
    
    # Grid coordinates
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['i-1', 'i', 'i+1', 'i+2'], fontsize=11, fontweight='bold', color='#cbd5e1')
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['j-1', 'j', 'j+1', 'j+2'], fontsize=11, fontweight='bold', color='#cbd5e1')
    
    # Field components placement
    ez_coords = [(1, 1), (2, 1), (1, 2), (2, 2)]
    for idx, (x, y) in enumerate(ez_coords):
        ax.scatter(x, y, color='#38bdf8', s=160, edgecolor='#0284c7', linewidth=1.5, zorder=5)
        ax.annotate(f"$E_z$ ({x},{y})", (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold', color='#38bdf8', fontsize=9, zorder=6)
        
    hx_coords = [(1.5, 1), (1.5, 2), (2.5, 1), (2.5, 2)]
    for idx, (x, y) in enumerate(hx_coords):
        ax.scatter(x, y, color='#34d399', s=120, marker='^', edgecolor='#059669', linewidth=1.5, zorder=5)
        if idx == 0:
            ax.annotate(r"$H_x$ (i+$\frac{1}{2}$, j)", (x, y), textcoords="offset points", xytext=(0,-15), ha='center', fontweight='bold', color='#34d399', fontsize=9, zorder=6)
            
    hy_coords = [(1, 1.5), (2, 1.5), (1, 2.5), (2, 2.5)]
    for idx, (x, y) in enumerate(hy_coords):
        ax.scatter(x, y, color='#fbbf24', s=120, marker='v', edgecolor='#d97706', linewidth=1.5, zorder=5)
        if idx == 0:
            ax.annotate(r"$H_y$ (i, j+$\frac{1}{2}$)", (x, y), textcoords="offset points", xytext=(-35,-3), ha='center', fontweight='bold', color='#fbbf24', fontsize=9, zorder=6)

    # Legend
    legend_elements = [
        patches.Patch(facecolor='#7f1d1d', edgecolor='#ef4444', alpha=0.5, label='Yee Unit Cell Boundary'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#38bdf8', markersize=11, markeredgecolor='#0284c7', label='$E_z$ Field (z-axis Node)'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#34d399', markersize=11, markeredgecolor='#059669', label='$H_x$ Field (Horizontal Edge)'),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='#fbbf24', markersize=11, markeredgecolor='#d97706', label='$H_y$ Field (Vertical Edge)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9, facecolor='#1e293b', edgecolor='#334155', fontsize=9.5)
    
    ax.set_title("2D FDTD Electromagnetic Yee Lattice Discretization Schema", fontsize=12, fontweight='bold', color='#f8fafc', pad=15)
    plt.tight_layout()
    
    fig.savefig(os.path.join(FIGURES_DIR, "fdtd_yee_grid.png"), dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print("[PASS] Generated: fdtd_yee_grid.png")


def draw_brep_structure():
    """Generates Figure 4: B-Rep Topological Hierarchy Model (Futuristic Dark Mode)."""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 9)
    ax.axis('off')

    # Reordered nodes to avoid line crossing: (x, y, color)
    nodes = {
        'Solid': (5.0, 8.0, '#38bdf8'),
        'Shell': (5.0, 6.7, '#0ea5e9'),
        'Face': (5.0, 5.4, '#0284c7'),
        'Wire': (2.0, 4.1, '#2563eb'),
        'Edge': (5.0, 2.5, '#f59e0b'),  # Centered Edge prevents cross lines
        'HalfEdge_A': (2.0, 2.5, '#a855f7'),
        'HalfEdge_B': (8.0, 2.5, '#a855f7'),
        'Vertex_A': (2.0, 0.8, '#f43f5e'),
        'Vertex_B': (8.0, 0.8, '#f43f5e')
    }

    # Add entity boxes with drop shadows
    for name, (x, y, color) in nodes.items():
        disp_name = name.split('_')[0]
        rect_coords = (x-0.8, y-0.35, 1.6, 0.7)
        add_drop_shadow(ax, rect_coords, zorder=2)
        
        rect = patches.FancyBboxPatch((x-0.8, y-0.35), 1.6, 0.7, boxstyle="round,pad=0.02", linewidth=1.2, edgecolor='#334155', facecolor=color, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y, disp_name, color='#f8fafc', fontweight='bold', ha='center', va='center', fontsize=9.5, zorder=4)

    # Connections to draw
    connections = [
        ('Solid', 'Shell', '1..*'),
        ('Shell', 'Face', '1..*'),
        ('Face', 'Wire', '1..* (Outer/Inner)'),
        ('Wire', 'HalfEdge_A', '1..* (Loop)'),
        ('HalfEdge_A', 'Vertex_A', 'origin'),
        ('HalfEdge_B', 'Vertex_B', 'origin'),
        ('HalfEdge_A', 'Edge', 'parent'),
        ('HalfEdge_B', 'Edge', 'parent')
    ]

    for start_node, end_node, label in connections:
        x1, y1, _ = nodes[start_node]
        x2, y2, _ = nodes[end_node]
        
        # Directed arrows
        ax.annotate('', xy=(x2, y2+0.35 if y2 < y1 else y2-0.35), xytext=(x1, y1-0.35 if y1 > y2 else y1+0.35),
                    arrowprops=dict(arrowstyle="-|>", color='#64748b', linewidth=1.5, zorder=1))
        
        # Label offset adjustment to prevent overlap with line
        ax.text((x1+x2)/2 + 0.15, (y1+y2)/2, label, color='#94a3b8', style='italic', fontsize=7.5, zorder=4)

    # Draw Twin bidirectional connection curved above the Edge node
    x1, y1, _ = nodes['HalfEdge_A']
    x2, y2, _ = nodes['HalfEdge_B']
    ax.annotate('', xy=(x1+0.8, y1), xytext=(x2-0.8, y2),
                arrowprops=dict(arrowstyle="<->", color='#c084fc', linewidth=2.0, connectionstyle="arc3,rad=-0.18", zorder=1))
    ax.text((x1+x2)/2, y1 - 0.7, "twin", color='#c084fc', fontweight='bold', ha='center', fontsize=8.5, zorder=4)

    # Title & Legend
    ax.text(5, 8.8, "Aetheris Boundary Representation (B-Rep) Topological Graph Schema", fontsize=12, fontweight='bold', color='#f8fafc', ha='center', zorder=1)
    
    ax.text(4.0, 0.6, "Entity Class Hierarchy:\n"
                      "â–  Boundary Topology (Solid, Shell, Face, Wire, Edge)\n"
                      "â–  Directed Connex (HalfEdge with pointer loops)\n"
                      "â–  Geometric Coordinates (Vertex Points)", 
            bbox=dict(boxstyle="round,pad=0.4", facecolor='#1e293b', edgecolor='#334155'),
            fontsize=8.5, color='#cbd5e1')
            
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "brep_hierarchy.png"), dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print("[PASS] Generated: brep_hierarchy.png")


def draw_dft_flowchart():
    """Generates Figure 3: Quantum Chemistry Density Functional Theory (DFT) SCF Loop Flowchart (Breathing Dark Mode)."""
    fig, ax = plt.subplots(figsize=(8.5, 10.5), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Spaced out coordinates: (name, x, y, label, bg_color)
    steps = {
        'init': (5, 11.2, '1. Input Molecular Geometry & Basis Set (STO-3G)', '#475569'),
        'integrals': (5, 9.9, '2. Analytic Integrals Builder (Obara-Saika / MD)\nOverlap S, Kinetic T, Nuclear V, ERI (4-Center Tensor)', '#334155'),
        'ortho': (5, 8.6, '3. LÃ¶wdin Symmetric Orthogonalization\nTransform Matrix: X = S^(-1/2)', '#1e293b'),
        'guess': (5, 7.3, '4. Initialize Core Density Matrix P', '#0f172a'),
        'fock': (5, 6.0, '5. Construct Fock/KS Matrix F\nRHF/UHF Hamiltonian + Numerical Grid XC V_xc', '#0284c7'),
        'diis': (5, 4.7, '6. DIIS Subspace Acceleration Extrapolation\nF_diis = Î£ c_i F_i subject to minimal err vec e_i', '#0369a1'),
        'diag': (5, 3.4, '7. Orthogonal Diagonalization (FC = SCE)\nUpdate Density P_new', '#075985'),
        'conv': (5, 1.9, '8. Check Convergence\n|P_new - P| < 10^(-8) ?', '#9a3412'),
        'results': (5, 0.5, '9. Output Converged Wavefunctions & Total Energy\nCompute Bandgap (HOMO-LUMO delta)', '#166534')
    }

    for name, (x, y, label, bg) in steps.items():
        if 'Check' in label:
            polygon = patches.Polygon([[x-2.2, y], [x, y+0.55], [x+2.2, y], [x, y-0.55]], 
                                       linewidth=1.2, edgecolor='#475569', facecolor=bg, zorder=5)
            polygon_shadow = patches.Polygon([[x-2.15, y-0.05], [x+0.05, y+0.5], [x+2.25, y-0.05], [x+0.05, y-0.6]], 
                                              linewidth=0, facecolor='#020617', alpha=0.5, zorder=4)
            ax.add_patch(polygon_shadow)
            ax.add_patch(polygon)
            text_color = '#f8fafc'
        else:
            rect_coords = (x-3.2, y-0.45, 6.4, 0.9)
            add_drop_shadow(ax, rect_coords, zorder=2)
            
            rect = patches.FancyBboxPatch((x-3.2, y-0.45), 6.4, 0.9, boxstyle="round,pad=0.02", 
                                          linewidth=1.2, edgecolor='#475569', facecolor=bg, zorder=3)
            ax.add_patch(rect)
            text_color = '#f8fafc'
            
        ax.text(x, y, label, color=text_color, fontweight='bold', ha='center', va='center', fontsize=8.5, zorder=6)

    # Connecting arrows
    arrows = [
        ('init', 'integrals'),
        ('integrals', 'ortho'),
        ('ortho', 'guess'),
        ('guess', 'fock'),
        ('fock', 'diis'),
        ('diis', 'diag'),
        ('diag', 'conv'),
        ('conv', 'results')
    ]

    for start, end in arrows:
        x1, y1 = steps[start][0], steps[start][1]
        x2, y2 = steps[end][0], steps[end][1]
        
        offset = 0.55 if 'Check' in steps[end][2] else 0.45
        ax.annotate('', xy=(x2, y2+offset), xytext=(x1, y1-0.45),
                    arrowprops=dict(arrowstyle="-|>", color='#64748b', linewidth=1.5, zorder=1))

    # Loop back path (No overlaps)
    ax.annotate('', xy=(steps['fock'][0]+3.2, steps['fock'][1]), xytext=(steps['conv'][0]+2.2, steps['conv'][1]),
                arrowprops=dict(arrowstyle="-|>", color='#f43f5e', linewidth=2.0, connectionstyle="bar,angle=90,fraction=-0.32", zorder=1))
    
    ax.text(5.25, 1.1, "YES (Converged)", color='#4ade80', fontweight='bold', fontsize=8.5)
    ax.text(8.0, 4.0, "NO (Iterate)", color='#f43f5e', fontweight='bold', fontsize=8.5)

    ax.text(5, 11.8, "Aetheris Computational Chemistry Self-Consistent Field (SCF) Solver", fontsize=12, fontweight='bold', color='#f8fafc', ha='center', zorder=1)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "dft_scf_flowchart.png"), dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print("[PASS] Generated: dft_scf_flowchart.png")


def draw_self_evolution():
    """Generates Figure 6: Autonomous Self-Evolution multi-agent workflow (Futuristic Dark Mode)."""
    fig, ax = plt.subplots(figsize=(9, 8.5), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Agent Boxes: (x, y, w, h, bg, text)
    agent_boxes = {
        'researcher': (0.8, 4.8, 2.6, 2.8, '#0369a1', 'RESEARCHER AGENT\n\n- Scans workspace files\n- Resolves dependencies\n- Identifies syntax errors\n- Outlines edits'),
        'surgeon': (4.7, 4.8, 2.6, 2.8, '#5b21b6', 'SURGEON AGENT\n\n- Reads instructions file\n- Matches targeted lines\n- Applies regex patches\n- Edits code locally'),
        'reviewer': (8.6, 4.8, 2.6, 2.8, '#14532d', 'REVIEWER AGENT\n\n- Runs static compilations\n- Checks Python imports\n- Validates changes\n- Patches runtime errors')
    }

    for name, (x, y, w, h, color, label) in agent_boxes.items():
        rect_coords = (x, y, w, h)
        add_drop_shadow(ax, rect_coords, zorder=2)
        
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", 
                                      linewidth=1.5, edgecolor='#334155', facecolor=color, alpha=0.95, zorder=3)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, color='#f8fafc', fontweight='bold', ha='center', va='center', fontsize=8, zorder=4)

    # Process files placed at lower coordinates (y=1.2 and y=3.0) to prevent overlaps
    rect_inst_coords = (3.15, 3.2, 1.8, 0.8)
    add_drop_shadow(ax, rect_inst_coords, zorder=5)
    rect_inst = patches.FancyBboxPatch((3.15, 3.2), 1.8, 0.8, boxstyle="round,pad=0.02", linewidth=1.2, edgecolor='#475569', facecolor='#1e293b', zorder=6)
    ax.add_patch(rect_inst)
    ax.text(4.05, 3.6, "surgeon_instructions.txt", color='#94a3b8', fontweight='bold', ha='center', va='center', fontsize=7.5, zorder=7)

    rect_code_coords = (4.7, 1.2, 2.6, 0.9)
    add_drop_shadow(ax, rect_code_coords, zorder=5)
    rect_code = patches.FancyBboxPatch((4.7, 1.2), 2.6, 0.9, boxstyle="round,pad=0.02", linewidth=1.2, edgecolor='#475569', facecolor='#854d0e', zorder=6)
    ax.add_patch(rect_code)
    ax.text(6.0, 1.65, "TARGET PYTHON CODEBASE\n(d:\\Project File\\...\\core\\)", color='#fef08a', fontweight='bold', ha='center', va='center', fontsize=7.5, zorder=7)

    # Connective linkages
    # 1. Researcher -> Instructions
    ax.annotate('', xy=(3.5, 4.0), xytext=(3.0, 4.8),
                arrowprops=dict(arrowstyle="-|>", color='#64748b', linewidth=1.5, zorder=10))
    
    # 2. Instructions -> Surgeon
    ax.annotate('', xy=(5.0, 4.8), xytext=(4.65, 4.0),
                arrowprops=dict(arrowstyle="-|>", color='#64748b', linewidth=1.5, zorder=10))
    
    # 3. Surgeon -> Codebase
    ax.annotate('', xy=(6.0, 2.1), xytext=(6.0, 4.8),
                arrowprops=dict(arrowstyle="-|>", color='#c084fc', linewidth=2.0, zorder=10))
    ax.text(6.15, 3.0, "String patches", color='#c084fc', fontweight='bold', fontsize=7.5)
    
    # 4. Reviewer <-> Codebase
    ax.annotate('', xy=(7.3, 1.65), xytext=(9.9, 4.8),
                arrowprops=dict(arrowstyle="<|-|>", color='#4ade80', linewidth=2.0, zorder=10))
    ax.text(8.8, 2.8, "Syntax & compile\ntest cycle", color='#4ade80', fontweight='bold', fontsize=7.5, ha='center')

    # 5. Reviewer reports errors to Surgeon
    ax.annotate('', xy=(7.3, 6.4), xytext=(8.6, 6.4),
                arrowprops=dict(arrowstyle="-|>", color='#f43f5e', linewidth=1.5, connectionstyle="arc3,rad=.15", zorder=10))
    ax.text(7.95, 7.05, "Feedback loops", color='#f43f5e', fontweight='bold', fontsize=7.5, ha='center')

    # Column Titles
    ax.text(2.1, 7.9, "Step 1: Diagnostics", color='#f8fafc', fontweight='bold', ha='center', fontsize=9.5)
    ax.text(6.0, 7.9, "Step 2: Surgeon Edits", color='#f8fafc', fontweight='bold', ha='center', fontsize=9.5)
    ax.text(9.9, 7.9, "Step 3: Verification", color='#f8fafc', fontweight='bold', ha='center', fontsize=9.5)

    ax.text(6, 8.6, "Aetheris Closed-Loop Self-Evolution Agentic Workflow", fontsize=12, fontweight='bold', color='#f8fafc', ha='center', zorder=1)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "self_evolution_workflow.png"), dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print("[PASS] Generated: self_evolution_workflow.png")


if __name__ == '__main__':
    print("Starting Aetheris figure generation...")
    draw_live_architecture()
    draw_yee_grid()
    draw_brep_structure()
    draw_dft_flowchart()
    draw_self_evolution()
    print("All paper figures generated successfully!")
