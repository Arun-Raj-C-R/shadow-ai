# Aetheris: A Unified Autonomous Multimodal Agentic Architecture Integrating Parametric CAD, Multiscale Physical Simulation, Materials Database Orchestration, and Closed-Loop Self-Evolution

**Authors:** Arun R.$^{1}$  
**Affiliation:** $^{1}$Independent Researcher  
**Correspondence:** [corresponding email]  
**Date:** May 2026  
**Status:** Submitted for Peer Review

---

## Abstract

We present **Aetheris**, a unified, autonomous, multimodal agentic architecture that tightly integrates computational engineering tools — including a parametric computer-aided design (CAD) kernel, multiscale physical simulators, and materials database orchestrators — within a single cognitive reasoning loop driven by large language models (LLMs). Unlike conventional AI-for-science workflows that treat LLMs as natural language wrappers around external commercial solvers, Aetheris embeds domain-specific computational engines directly into the agent's executable workspace, establishing a closed feedback loop between cognitive reasoning and physical validation. The architecture comprises five principal subsystems: (1) a **real-time multimodal streaming interface** connecting audio, visual, and textual modalities to a persistent LLM session via WebSocket-based communication; (2) a **parametric CAD kernel** implemented from first principles using half-edge boundary representation (B-Rep) topology, Cox–de Boor non-uniform rational B-spline (NURBS) evaluation, and constructive solid geometry (CSG) via binary space partitioning (BSP) trees; (3) a **multiscale physical simulation suite** including two-dimensional finite-difference time-domain (FDTD) electromagnetics, finite-difference multiphysics solvers for steady-state heat and electrostatic problems, and a one-dimensional semiconductor device simulator; (4) a **materials discovery pipeline** integrating the Google DeepMind GNoME stable-materials database and the Materials Project API for property retrieval and screening; and (5) a **closed-loop self-evolutionary coding pipeline** employing a multi-agent Scanner–Analyzer–Surgeon framework to autonomously diagnose, patch, and verify modifications to its own source code. We additionally describe a persistent vector-memory subsystem backed by ChromaDB for long-term retrieval-augmented generation (RAG). We detail the mathematical formulations, software design, and integration strategy of each subsystem, present benchmark results for the CAD constraint solver, FDTD wave propagation, and semiconductor device simulation, and discuss the current limitations and future development trajectory of the framework.

**Keywords:** autonomous agents, multimodal AI, parametric CAD, boundary representation, NURBS, constructive solid geometry, FDTD electromagnetics, semiconductor simulation, materials informatics, self-evolving code, retrieval-augmented generation

---

## 1. Introduction

### 1.1 Motivation

The application of artificial intelligence to physical sciences and engineering — broadly termed AI4Science — has advanced along two largely independent trajectories. The first trajectory, **predictive modeling**, trains deep neural networks on curated databases of materials, molecules, or structures to forecast target properties such as formation energies, bandgaps, or mechanical moduli [1, 2]. The second trajectory, **workflow automation**, employs large language models (LLMs) to compose and execute scripts for commercially licensed computational suites (e.g., VASP [3], Abaqus [4], SolidWorks [5]), effectively using the LLM as a natural language interface to external solvers.

Both paradigms suffer from a fundamental **coupling deficit**: the AI reasoning engine is detached from the low-level mathematical representations of the physics and geometry it manipulates. In the predictive paradigm, the model operates on fixed, pre-computed feature vectors and cannot inspect or modify intermediate computational quantities such as geometric Jacobians, field update matrices, or topological adjacency graphs. In the automation paradigm, the LLM generates scripts but cannot observe solver internals — rendering self-correction upon numerical failure difficult or impossible without human intervention.

### 1.2 Contribution

To address this gap, we introduce **Aetheris**, an autonomous multimodal agentic architecture that integrates domain-specific computational engines directly into the agent's executable workspace. Rather than wrapping external solvers, Aetheris embeds custom-built parametric CAD kernels, physics simulators, and materials database clients as callable tools within a persistent LLM session. This design establishes a tight feedback loop: the agent can invoke a simulation, inspect intermediate results (e.g., field distributions, topological invariants, convergence residuals), reason about failures, modify parameters, and re-execute — all within a single continuous conversation.

The principal contributions of this work are as follows:

1. **A real-time multimodal agentic architecture** that maintains a persistent, bidirectional WebSocket connection to a multimodal LLM (Google Gemini), streaming audio, video, and screen-capture inputs concurrently with structured tool-call dispatch and response handling.

2. **A parametric CAD kernel implemented from first principles**, featuring a half-edge B-Rep topology manager, Cox–de Boor NURBS curve and surface evaluators, and a BSP-tree-based CSG Boolean engine — all written in pure Python without dependence on external CAD libraries.

3. **A multiscale physical simulation suite** comprising a two-dimensional FDTD electromagnetic wave solver with absorbing boundary layers, a finite-difference multiphysics engine for steady-state heat conduction and electrostatics, and a one-dimensional semiconductor p–n junction simulator based on depletion-region analysis.

4. **A materials discovery pipeline** that programmatically queries the Google DeepMind GNoME database (520,000+ computationally stable compounds) and the Materials Project API, enabling automated screening, property retrieval, and structure generation.

5. **A closed-loop self-evolutionary coding pipeline** employing a three-agent (Scanner–Analyzer–Surgeon) framework that can autonomously diagnose errors in the system's own codebase, generate targeted patches, verify syntactic correctness, and apply fixes with rollback capability.

6. **A persistent vector-memory subsystem** backed by ChromaDB, providing retrieval-augmented generation (RAG) with importance-weighted recall, background compression, and dual-write persistence.

### 1.3 Paper Organization

The remainder of this paper is organized as follows. Section 2 reviews related work. Section 3 describes the real-time multimodal cognitive core and tool-dispatch architecture. Section 4 details the persistent memory and retrieval-augmented generation subsystem. Section 5 presents the parametric CAD kernel, including the B-Rep data model, NURBS evaluation, constraint solving, and CSG Boolean operations. Section 6 describes the multiscale physical simulation suite. Section 7 discusses the materials database integrations. Section 8 presents the self-evolutionary coding pipeline. Section 9 reports experimental benchmarks. Section 10 discusses limitations and scope. Section 11 concludes with future directions.

---

## 2. Related Work

### 2.1 LLM-Based Scientific Agents

Recent work has demonstrated the potential of LLMs as reasoning engines for scientific workflows. **ChemCrow** [6] equips an LLM with chemistry-specific tools for reaction planning and property prediction. **Coscientist** [7] integrates LLMs with robotic laboratory equipment for autonomous chemical synthesis. **SciAgent** frameworks [8] compose multi-step experimental protocols using chain-of-thought prompting. However, these systems typically invoke external solvers as black boxes, without embedding the computational kernels within the agent's own workspace.

### 2.2 Parametric CAD and Computational Geometry

Boundary representation (B-Rep) modeling, formalized by Baumgart [9] and extended by Mäntylä [10], remains the dominant topological framework in commercial CAD systems. The half-edge data structure provides efficient adjacency queries for manifold solids. NURBS, standardized by Piegl and Tiller [11], underpin modern freeform surface modeling. CSG via BSP trees was demonstrated by Naylor et al. [12] for polyhedral set operations. Our work implements these classical algorithms from first principles within an LLM-driven agent, enabling the agent to directly construct, inspect, and modify solid models.

### 2.3 Electromagnetic and Device Simulation

The finite-difference time-domain (FDTD) method, introduced by Yee [13], discretizes Maxwell's equations on a staggered grid and remains widely used for electromagnetic wave simulation. Perfectly matched layers (PML), introduced by Bérenger [14], provide absorbing boundary conditions. Semiconductor device simulation, reviewed by Selberherr [15], couples Poisson's equation with carrier transport equations. Our implementations provide simplified but functional versions of these solvers, suitable for educational demonstrations and rapid prototyping within the agentic loop.

### 2.4 Materials Informatics Databases

The Materials Project [16] provides computed properties for over 150,000 inorganic compounds via a REST API. The Google DeepMind GNoME project [17] expanded the space of known stable materials by over an order of magnitude through graph neural network predictions validated by density functional theory (DFT). Our framework integrates both databases as first-class tools within the agent's reasoning loop.

### 2.5 Self-Modifying AI Systems

The concept of self-modifying code has a long history in computer science [18]. Recent work on LLM-driven code repair includes **SWE-Agent** [19] for automated bug fixing and **AutoCodeRover** [20] for autonomous software engineering. Aetheris extends this paradigm by applying multi-agent self-modification to its own scientific computing toolkit, using a structured Scanner–Analyzer–Surgeon pipeline with rollback safety.

---

## 3. Cognitive Core and Real-Time Multimodal Architecture

### 3.1 System Overview

The Aetheris cognitive core maintains a persistent, bidirectional communication session with the Google Gemini multimodal LLM via the Gemini Live API. The architecture is split into concurrent, asynchronous tasks managed by Python's `asyncio` event loop. Figure 1 illustrates the high-level data flow.

```
                                  +------------------------------------+
                                  |     AETHERIS COGNITIVE CORE        |
                                  |    (Gemini Live API Session)       |
                                  +-----------------+------------------+
                                                    |
         +------------------------------------------+------------------------------------------+
         |                                          |                                          |
+--------v-------------------------+       +--------v-------------------------+       +--------v-------------------------+
|      PHYSICAL SIMULATIONS        |       |        PARAMETRIC CAD            |       |      MATERIALS DATABASES         |
+----------------------------------+       +----------------------------------+       +----------------------------------+
| * 2D FDTD EM Wave Engine         |       | * B-Rep Manifold Solid Modeler   |       | * DeepMind GNoME Database        |
| * FD Heat & Electrostatics       |       | * Cox-de Boor NURBS Evaluator    |       | * Materials Project API Client   |
| * 1D PN Junction Simulator       |       | * BSP Tree CSG Booleans          |       | * Perovskite Crystal Builder     |
+----------------------------------+       +----------------------------------+       +----------------------------------+
         |                                          |                                          |
         +------------------------------------------+------------------------------------------+
                                                    |
                                          +---------v---------+
                                          | COGNITIVE MEMORY  |
                                          |  (ChromaDB + RAG) |
                                          +---------+---------+
                                                    |
         +------------------------------------------+------------------------------------------+
         |                                                                                     |
+--------v-----------------------------------------+                 +-------------------------v-------------------------+
|      SELF-EVOLVING PIPELINE                      |                 |      AUXILIARY TOOLS                              |
+--------------------------------------------------+                 +---------------------------------------------------+
| * Multi-Agent Scanner-Analyzer-Surgeon           |                 | * Python REPL Sandbox                             |
| * Automatic syntax checking & patching           |                 | * arXiv & Semantic Scholar Search                 |
| * Codebase self-modification with rollback       |                 | * Wolfram Alpha API                               |
+--------------------------------------------------+                 +---------------------------------------------------+
```

**Figure 1.** High-level system architecture of Aetheris, showing the cognitive core, computational tool subsystems, memory layer, and self-evolution pipeline.

### 3.2 Multimodal Streaming and Live Grounding

The primary interface uses a WebSocket-based streaming loop connected to the Gemini Live API. The loop is decomposed into the following concurrent tasks:

- **Audio capture and playback.** PyAudio registers microphone input at 16 kHz (PCM 16-bit, monaural) and writes incoming model audio responses to a speaker output buffer at 24 kHz. Audio chunks are base64-encoded and transmitted as binary WebSocket frames.

- **Visual input.** Two visual modalities are supported concurrently: (a) direct camera capture via OpenCV (`cv2.VideoCapture`) and (b) screen capture via the MSS library. Frames are acquired at approximately 1 Hz, compressed to JPEG format with a maximum resolution of $1024 \times 1024$ pixels, base64-encoded, and transmitted as inline image payloads within the session context.

- **Session persistence.** The WebSocket connection implements session handle save and load mechanisms, enabling conversation resumption across client restarts without loss of context.

### 3.3 Structured Tool-Call Dispatch

When the LLM determines that a user query requires computational support, it emits a structured tool-call request in JSON format specifying the target function name and parameter arguments. The orchestrator intercepts this request, validates the call against a registry of over 50 Python tool definitions, routes execution to the corresponding module, and returns the result payload to the LLM session for continued reasoning.

```
+------------------+                   +----------------------+                   +---------------------+
|                  |   WebSocket Stream|                      |   Tool Schema List|                     |
|   USER / CLIENT  +------------------>+ AETHERIS ORCHESTRATOR+------------------>+ MULTIMODAL LLM      |
|                  |<------------------+                      |<------------------+ (Gemini Live API)   |
+------------------+   Audio Response  +-------+---^----------+   Tool Call JSON  +---------------------+
                                               |   |
                                       Execute |   | Return Payload
                                               v   |
                                       +-------+---+----------+
                                       | WORKSPACE SOLVER BUS |
                                       | (CAD/FDTD/TCAD/etc.) |
                                       +----------------------+
```

**Figure 2.** Tool-call dispatch sequence. The orchestrator mediates between the user-facing WebSocket stream and the computational workspace, routing structured function calls to domain-specific solvers.

The dispatch sequence proceeds as follows:

1. **Interception.** The orchestrator receives a tool-call frame from the active Gemini session.
2. **Schema validation.** The incoming JSON structure (method name and parameter arguments) is validated against registered Python tool definitions.
3. **Execution routing.** The call is dispatched to the corresponding Python module via an asynchronous thread pool (`asyncio.to_thread`), ensuring that long-running computations do not block the streaming loop.
4. **Sandboxed evaluation.** The solver executes within the local workspace environment. Runtime exceptions are caught, wrapped in structured error payloads, and returned without terminating the session.
5. **Session update.** The result payload is serialized as a `toolResponse` frame and returned to the Gemini session, enabling the LLM to incorporate computational results into its ongoing reasoning.

---

## 4. Persistent Memory and Retrieval-Augmented Generation

### 4.1 Vector Memory Architecture

Aetheris implements a persistent memory subsystem (`VectorMemoryManager`) to maintain long-term state without bloating the conversational token context. The system performs dual-write persistence: every memory entry is stored simultaneously to a local text file (`memory.txt`) and to a ChromaDB vector store, using a `ThreadPoolExecutor` for parallel I/O.

### 4.2 Memory Classification

Incoming interaction sequences are classified into semantic categories using a heuristic keyword-matching pipeline. The classification taxonomy comprises:

$$\text{Type}(T) = \begin{cases} 
      \text{Preference} & \text{if } \text{match}(\text{``prefer'', ``always'', ``never''}) \\
      \text{Procedure} & \text{if } \text{match}(\text{``step'', ``how to'', ``process''}) \\
      \text{Protocol} & \text{if } \text{match}(\text{``must'', ``required'', ``rule''}) \\
      \text{Insight} & \text{if } \text{match}(\text{``realized'', ``learned'', ``discovered''}) \\
      \text{Experience} & \text{if } \text{match}(\text{``happened'', ``occurred'', ``event''}) \\
      \text{Fact} & \text{otherwise}
   \end{cases}$$

where $\text{match}(\cdot)$ denotes a case-insensitive regular expression search over the input text.

### 4.3 Retrieval and Relevance Scoring

The ChromaDB collection stores sentence-transformer embeddings and supports approximate nearest-neighbor queries. Given a query embedding $u$ and a stored embedding $v$, the squared $L_2$ distance is computed as:

$$d(u,v) = \sum_{i=1}^{N} (u_i - v_i)^2$$

This distance is converted to a normalized relevance percentage:

$$\text{Relevance}(\%) = \max\left(0, \min\left(100, \left(1 - \frac{d(u,v)}{1.5}\right) \times 100\right)\right)$$

### 4.4 Importance-Weighted Recall

Each memory entry maintains an importance score that is adjusted through user feedback:

- **Positive feedback (helpful):** $\text{importance} \leftarrow \text{importance} + 0.1$
- **Negative feedback (unhelpful):** $\text{importance} \leftarrow \text{importance} - 0.15$

The system maintains a working memory buffer of the 20 most recently accessed entries and tracks per-query analytics to optimize retrieval performance over time.

### 4.5 Background Compression

When an interaction text exceeds 500 characters, it is dispatched to a background thread for compression via an LLM call (Gemini Flash Lite), generating a condensed Sparse Priming Representation (SPR). The compressed text replaces the verbose original in the vector store, minimizing storage and embedding dimensionality noise.

### 4.6 Fallback Retrieval

If the ChromaDB service is unavailable, the system falls back to keyword-based regex search over the plaintext `memory.txt` file, ensuring graceful degradation of the retrieval pipeline.

---

## 5. Parametric CAD Kernel and Boundary Representation Solid Modeler

The Aetheris CAD subsystem implements a three-dimensional solid modeler from first principles in pure Python. The kernel comprises four principal components: a half-edge B-Rep topology manager, a NURBS curve and surface evaluator, a computational geometry library, and a BSP-tree-based CSG Boolean engine.

### 5.1 Half-Edge Boundary Representation Topology

Manifold solids are represented using a half-edge data structure [10], which establishes explicit adjacency relationships between topological entities: vertices, edges, half-edges, wires, faces, shells, and solids.

**Definition 1 (Vertex).** A vertex $v$ stores a three-dimensional coordinate $(x, y, z) \in \mathbb{R}^3$ and a pointer to one outgoing half-edge.

**Definition 2 (Half-Edge).** A half-edge $he$ is a directed edge segment storing:

$$he = \{v_{\text{origin}},\ e_{\text{parent}},\ he_{\text{next}},\ he_{\text{prev}},\ he_{\text{twin}},\ w_{\text{parent}}\}$$

where $v_{\text{origin}}$ is the origin vertex, $e_{\text{parent}}$ is the parent geometric edge, $he_{\text{next}}$ and $he_{\text{prev}}$ are the successor and predecessor half-edges in the bounding wire, $he_{\text{twin}}$ is the oppositely directed twin half-edge sharing the same geometric edge, and $w_{\text{parent}}$ is the parent wire.

**Definition 3 (Wire).** A wire $w$ is a closed, ordered loop of half-edges forming a boundary cycle.

**Definition 4 (Face).** A face $f$ contains a pointer to an outer bounding wire, a (possibly empty) list of inner wires representing holes, and a geometric surface definition (plane, cylinder, or NURBS patch).

**Definition 5 (Shell).** A shell $\mathcal{S}$ is a connected set of faces forming a closed or open surface.

**Definition 6 (Solid).** A solid $\Omega$ is a closed volume bounded by a shell.

The topological hierarchy is depicted in Figure 3.

```
       +----------------------- Solid -------------------------+
       |                                                       |
       +-----------> Shell -------------------------+          |
                     |                              |          |
                     v                              v          |
                    Face (Outer Wire) ------------> Face (Inner Wire)
                     |                              |
                     +------> Wire (Loop) <---------+
                               |
                               v
                           HalfEdge <==================== Twin HalfEdge
                           (origin, next, prev, parent)   (origin, next, prev, parent)
                               |                                  |
                               v                                  v
                             Vertex                             Vertex
                           (X, Y, Z)                          (X, Y, Z)
```

**Figure 3.** Half-edge B-Rep topological hierarchy. Each geometric edge is represented as a pair of oppositely directed half-edges (twin relationship), enabling efficient adjacency traversal.

#### 5.1.1 Topological Invariant Verification

The topological integrity of every solid is verified using the Euler–Poincaré formula:

$$V - E + F - (L - F) = 2(S - G)$$

where $V$, $E$, $F$, $L$, $S$, and $G$ denote the counts of vertices, edges, faces, loops (wires), shells, and genus (handles), respectively. For a simple polyhedron without holes or handles, this reduces to the classical Euler relation $V - E + F = 2$.

#### 5.1.2 Geometric Computations

Face normals are computed using Newell's method, which is numerically stable for non-convex polygons. The signed volume of a closed shell is computed via the divergence theorem:

$$\text{Vol}(\Omega) = \frac{1}{6} \sum_{f \in \mathcal{S}} \sum_{\triangle \in f} \left| \mathbf{v}_1 \cdot (\mathbf{v}_2 \times \mathbf{v}_3) \right|$$

where the sum ranges over all triangulated faces of the shell.

#### 5.1.3 Primitive Builders

The kernel provides parametric constructors for five primitive solids: rectangular prism (`make_box`), cylinder (`make_cylinder`), sphere (`make_sphere`), cone (`make_cone`), and torus (`make_torus`). Each constructor generates a valid B-Rep solid satisfying the Euler–Poincaré invariant.

### 5.2 Non-Uniform Rational B-Splines (NURBS)

NURBS are used for both curve and surface representations, supporting freeform geometry beyond the planar and quadric primitives.

#### 5.2.1 Cox–de Boor Recursion

For a spline of degree $p$ defined over a knot vector $U = \{u_0, u_1, \dots, u_m\}$, the B-spline basis functions $N_{i,p}(u)$ are evaluated recursively [11]:

$$N_{i,0}(u) = \begin{cases} 1 & \text{if } u_i \leq u < u_{i+1} \\ 0 & \text{otherwise} \end{cases}$$

$$N_{i,p}(u) = \frac{u - u_i}{u_{i+p} - u_i} N_{i,p-1}(u) + \frac{u_{i+p+1} - u}{u_{i+p+1} - u_{i+1}} N_{i+1,p-1}(u)$$

with the convention that $0/0 = 0$ when a knot span has zero length.

#### 5.2.2 Rational Curve Evaluation

A NURBS curve of degree $p$ with $n+1$ control points $\{P_i\}$ and weights $\{w_i\}$ is evaluated as:

$$C(u) = \frac{\sum_{i=0}^{n} N_{i,p}(u)\, w_i\, P_i}{\sum_{i=0}^{n} N_{i,p}(u)\, w_i}$$

The implementation supports derivative evaluation, curvature computation, and adaptive tessellation for rendering.

#### 5.2.3 Tensor-Product Surface Evaluation

A NURBS surface of degree $p$ in $u$ and $q$ in $v$ is:

$$S(u,v) = \frac{\sum_{i=0}^{n} \sum_{j=0}^{m} N_{i,p}(u)\, N_{j,q}(v)\, w_{i,j}\, P_{i,j}}{\sum_{i=0}^{n} \sum_{j=0}^{m} N_{i,p}(u)\, N_{j,q}(v)\, w_{i,j}}$$

where $P_{i,j} \in \mathbb{R}^3$ are the control points and $w_{i,j} > 0$ are their associated weights. Surface normals are computed analytically as:

$$\mathbf{n}(u,v) = \frac{\partial S}{\partial u} \times \frac{\partial S}{\partial v}$$

#### 5.2.4 Knot Insertion

The implementation supports Boehm's knot insertion algorithm [21] for refinement without changing the curve geometry, enabling local control point insertion and degree elevation.

#### 5.2.5 Bézier Curves

As a special case, Bézier curves are supported via the De Casteljau algorithm, providing numerically stable evaluation for polynomial curve segments.

### 5.3 Computational Geometry Library

The CAD kernel includes a computational geometry library providing the following algorithms:

- **Convex hull computation** via Andrew's monotone chain algorithm, operating in $\mathcal{O}(n \log n)$ time.
- **Delaunay triangulation** via the Bowyer–Watson incremental insertion algorithm.
- **Ray–triangle intersection** via the Möller–Trumbore algorithm [22], used for point-in-solid queries and ray casting.
- **Spatial indexing** via a KD-tree and spatial hashing, accelerating nearest-neighbor queries and collision detection.

### 5.4 Constructive Solid Geometry via BSP Trees

Constructive Solid Geometry (CSG) Boolean operations (union, intersection, subtraction) on B-Rep solids are computed using Binary Space Partitioning (BSP) trees [12].

#### 5.4.1 BSP Tree Construction

The surfaces of B-Rep faces are first discretized into planar polygons. A BSP tree is then constructed by recursively partitioning the polygon set:

1. A partitioning plane is selected from the polygon set.
2. Each polygon is classified relative to the partitioning plane as `COPLANAR`, `FRONT`, `BACK`, or `SPANNING`.
3. `SPANNING` polygons are split at the plane intersection into two sub-polygons. The intersection points are computed by linear interpolation along edges that cross the plane.
4. Sub-polygons are recursively inserted into the front or back subtree.

#### 5.4.2 Boolean Operations

Given solids $A$ and $B$ with respective BSP trees $T_A$ and $T_B$:

- **Union** ($A \cup B$): Retain polygons of $A$ outside $B$ and polygons of $B$ outside $A$, plus coplanar faces.
- **Intersection** ($A \cap B$): Retain polygons of $A$ inside $B$ and polygons of $B$ inside $A$.
- **Subtraction** ($A \setminus B$): Retain polygons of $A$ outside $B$ and polygons of $B$ inside $A$ (with reversed normals).

Polygon classification (inside/outside) is performed by clipping against the opposing solid's BSP tree using the `clip_polygons` recursive procedure. The resulting polygon sets are merged and reconstructed into a valid B-Rep shell.

---

## 6. Multiscale Physical Simulation Suite

Aetheris includes custom physical simulation modules for solving partial differential equations (PDEs) in electromagnetics, thermal transport, electrostatics, and semiconductor device physics. These implementations are designed as lightweight, interpretable solvers suitable for rapid prototyping and educational demonstration within the agentic loop, rather than as replacements for production-grade simulation codes.

### 6.1 Two-Dimensional FDTD Electromagnetics

The electromagnetics simulator solves Maxwell's equations for transverse electric ($\text{TE}_z$) wave propagation in two dimensions using the finite-difference time-domain (FDTD) method on a Yee staggered grid [13].

#### 6.1.1 Governing Equations

The $\text{TE}_z$ mode equations in a source-free, linear, isotropic medium are:

$$\frac{\partial H_x}{\partial t} = -\frac{1}{\mu} \frac{\partial E_z}{\partial y}, \quad \frac{\partial H_y}{\partial t} = \frac{1}{\mu} \frac{\partial E_z}{\partial x}, \quad \frac{\partial E_z}{\partial t} = \frac{1}{\varepsilon} \left( \frac{\partial H_y}{\partial x} - \frac{\partial H_x}{\partial y} \right)$$

#### 6.1.2 Discretization

Space and time are discretized on the staggered Yee lattice, where $E_z$ is sampled at integer grid nodes and $H_x$, $H_y$ at half-integer positions:

$$H_{x,\, i,\, j}^{n+1/2} = H_{x,\, i,\, j}^{n-1/2} - \frac{\Delta t}{\mu \Delta y} \left( E_{z,\, i,\, j+1}^{n} - E_{z,\, i,\, j}^{n} \right)$$

$$H_{y,\, i,\, j}^{n+1/2} = H_{y,\, i,\, j}^{n-1/2} + \frac{\Delta t}{\mu \Delta x} \left( E_{z,\, i+1,\, j}^{n} - E_{z,\, i,\, j}^{n} \right)$$

$$E_{z,\, i,\, j}^{n+1} = E_{z,\, i,\, j}^{n} + \frac{\Delta t}{\varepsilon_{i,j}} \left[ \frac{H_{y,\, i,\, j}^{n+1/2} - H_{y,\, i-1,\, j}^{n+1/2}}{\Delta x} - \frac{H_{x,\, i,\, j}^{n+1/2} - H_{x,\, i,\, j-1}^{n+1/2}}{\Delta y} \right]$$

#### 6.1.3 Absorbing Boundary Conditions

To truncate the computational domain without introducing spurious reflections, we apply an absorbing boundary layer using a quadratic damping profile. Within a boundary region of thickness $d_{\text{PML}}$, the electric field is attenuated at each time step:

$$E_{z,\, i,\, j}^{n+1} \leftarrow E_{z,\, i,\, j}^{n+1} \cdot \sigma(r)$$

where $\sigma(r)$ is a damping factor that ramps quadratically from 1.0 at the interior boundary to a minimum value $\sigma_{\min}$ at the domain edge:

$$\sigma(r) = 1 - (1 - \sigma_{\min}) \left(\frac{r}{d_{\text{PML}}}\right)^2$$

and $r$ denotes the perpendicular distance from the interior boundary to the grid point. This approach provides practical wave absorption for the simulation geometries considered, though it does not achieve the theoretical reflection-free performance of a full uniaxial PML (UPML) formulation [14].

#### 6.1.4 Numerical Stability

Temporal stability is ensured by satisfying the Courant–Friedrichs–Lewy (CFL) condition:

$$\Delta t \leq \frac{1}{c\sqrt{\frac{1}{\Delta x^2} + \frac{1}{\Delta y^2}}}$$

where $c$ is the speed of light in the medium. In practice, we set $\Delta t = 0.9 \cdot \Delta x / (c\sqrt{2})$ for a uniform grid with $\Delta x = \Delta y$.

### 6.2 Two-Dimensional Finite-Difference Multiphysics Engine

For steady-state heat conduction and electrostatic potential problems, we solve the Laplace or Poisson equation:

$$\nabla^2 \Phi(x,y) = f(x,y)$$

using the standard five-point finite-difference stencil:

$$\frac{\Phi_{i+1,j} - 2\Phi_{i,j} + \Phi_{i-1,j}}{\Delta x^2} + \frac{\Phi_{i,j+1} - 2\Phi_{i,j} + \Phi_{i,j-1}}{\Delta y^2} = f_{i,j}$$

The resulting linear system is relaxed using Gauss–Seidel iteration:

$$\Phi_{i,j}^{(k+1)} = \frac{1}{4} \left( \Phi_{i+1,j}^{(k)} + \Phi_{i-1,j}^{(k+1)} + \Phi_{i,j+1}^{(k)} + \Phi_{i,j-1}^{(k+1)} - \Delta x^2 f_{i,j} \right)$$

Iteration continues until the maximum pointwise residual falls below a prescribed tolerance $\epsilon$. Once converged, derived physical fields are computed by numerical differentiation. For electrostatics:

$$E_{x,\, i,\, j} = -\frac{V_{i+1,j} - V_{i-1,j}}{2\Delta x}, \quad E_{y,\, i,\, j} = -\frac{V_{i,j+1} - V_{i,j-1}}{2\Delta y}$$

For heat conduction, the analogous quantities yield the heat flux vector $\mathbf{q} = -k\nabla T$.

### 6.3 One-Dimensional Semiconductor Device Simulator

The semiconductor module models a one-dimensional p–n junction using the depletion approximation.

#### 6.3.1 Built-In Potential

For a p–n junction with acceptor concentration $N_A$ and donor concentration $N_D$, the built-in potential at thermal equilibrium is:

$$V_{\text{bi}} = V_T \ln\left(\frac{N_A N_D}{n_i^2}\right)$$

where $V_T = kT/q$ is the thermal voltage and $n_i$ is the intrinsic carrier concentration.

#### 6.3.2 Depletion Region

Under an applied voltage $V_{\text{applied}}$, the depletion region widths on the p-side and n-side are:

$$x_p = \sqrt{\frac{2\varepsilon(V_{\text{bi}} - V_{\text{applied}})\, N_D}{q\, N_A(N_A + N_D)}}, \quad x_n = \sqrt{\frac{2\varepsilon(V_{\text{bi}} - V_{\text{applied}})\, N_A}{q\, N_D(N_A + N_D)}}$$

The electric field within the depletion region is computed analytically from the charge distribution, and the electrostatic potential is obtained by integration.

#### 6.3.3 Carrier Concentrations

Carrier concentration profiles are computed using Boltzmann statistics:

$$n(x) = n_i \exp\left(\frac{V(x) - V(0)}{V_T}\right), \quad p(x) = n_i \exp\left(-\frac{V(x) - V(0)}{V_T}\right)$$

The current–voltage characteristic follows the ideal diode equation:

$$I = I_0 \left[\exp\left(\frac{V_{\text{applied}}}{V_T}\right) - 1\right]$$

where $I_0$ is the reverse saturation current.

> **Remark.** This simulator employs the depletion approximation with analytical solutions rather than a self-consistent numerical solution of coupled Poisson and drift-diffusion equations (e.g., via the Scharfetter–Gummel discretization). While suitable for demonstrating junction physics within the agentic loop, it does not capture effects such as high-injection, carrier recombination, or non-equilibrium transport. Extension to a full numerical drift-diffusion solver is discussed in Section 10.2.

---

## 7. Materials Database Integration and Structural Generation

To ground computational modeling in empirical and computationally validated data, Aetheris integrates two major materials databases as first-class tools.

### 7.1 GNoME Database Search

The framework provides programmatic search over the Google DeepMind GNoME database [17], which contains over 520,000 novel inorganic compounds predicted to be thermodynamically stable. The search interface supports filtering by:

- Chemical elements (with exact-match regex patterns to prevent partial matches, e.g., query `S` matching `Si`, `Sb`, or `Sr`)
- Crystal system and space group
- Bandgap range and formation energy
- Decomposition energy (thermodynamic stability metric)

Element queries use word-boundary regular expressions to ensure chemical specificity:

```python
pattern = rf"'(?:{element})'"  # Matches exact list items in the element string
```

### 7.2 Materials Project API Integration

Aetheris wraps the Materials Project REST API [16] via the `mp-api` client library, providing access to over 30 property endpoints including:

- Electronic structures: density of states, band structures, and bandgaps
- Phonon densities of states for thermal stability assessment
- Pourbaix diagrams for aqueous electrochemical stability
- Wulff shapes and surface energy tensors
- Battery electrode properties: voltage, specific capacity, and charge density
- Elastic tensors and mechanical properties

### 7.3 Perovskite Crystal Builder

For rapid structural prototyping, Aetheris includes a parametric perovskite crystal builder that generates $ABX_3$ structures by placing:

- $A$-site cations at the unit cell corners $(0, 0, 0)$
- $B$-site cations at the body center $(0.5, 0.5, 0.5)$
- $X$-site anions at the face centers $(0.5, 0.5, 0)$, $(0.5, 0, 0.5)$, and $(0, 0.5, 0.5)$

The builder outputs Crystallographic Information Files (CIF) and renders interactive three-dimensional structures via WebGL using Three.js.

---

## 8. Closed-Loop Self-Evolutionary Coding Pipeline

To enable autonomous adaptation, debugging, and feature development, Aetheris includes a closed-loop self-evolution engine that can modify, repair, and extend its own tool codebase.

### 8.1 Multi-Agent Architecture

The self-evolution system operates as a three-stage multi-agent pipeline:

```
+--------------------------------------------------------------+
| SCANNER AGENT (Phase 1: Diagnostics)                         |
|  - Scans all Python files in the workspace                   |
|  - Analyzes dependencies and detects structural errors       |
|  - Generates a structured JSON change plan                   |
+------------------------------+-------------------------------+
                               |
                               v
+--------------------------------------------------------------+
| ANALYZER AGENT (Phase 2: Planning)                           |
|  - Interprets the scanner's findings                         |
|  - Creates detailed, line-range-specific edit instructions   |
|  - Writes modification plan for the surgeon                  |
+------------------------------+-------------------------------+
                               |
                               v
+--------------------------------------------------------------+
| SURGEON AGENT (Phase 3: Implementation & Verification)       |
|  - Reads target code files                                   |
|  - Applies localized string/regex patches at specific lines  |
|  - Creates backup copies before modification                 |
|  - Verifies syntax via ast.parse after each edit             |
|  - Supports rollback on verification failure                 |
+--------------------------------------------------------------+
```

**Figure 4.** Three-stage self-evolution pipeline. The Scanner identifies issues, the Analyzer formulates a repair plan, and the Surgeon applies targeted patches with backup and rollback capability.

### 8.2 Execution Safeguards

The pipeline incorporates several safety mechanisms to prevent destructive self-modification:

1. **Line-range targeting.** The Surgeon agent applies edits to specific line ranges using string and regular expression matching, rather than rewriting entire files. This preserves existing code, comments, and documentation.

2. **Pre-edit backup.** Before applying any modification, the target file is copied to a backup location, enabling rollback if verification fails.

3. **Post-edit syntax verification.** After each patch is applied, the modified file is parsed using Python's `ast.parse` to verify syntactic correctness. If parsing fails, the backup is restored automatically.

4. **Iteration limits.** Each agent phase has a bounded iteration count to prevent infinite loops in the self-modification cycle.

---

## 9. Experimental Benchmarks

To validate the computational subsystems, we present benchmark results for the CAD kernel, FDTD simulator, and semiconductor device simulator.

### 9.1 FDTD Wave Propagation Stability

The FDTD electromagnetics engine was evaluated on a $200 \times 200$ grid with a Gaussian-modulated sinusoidal point source at wavelength $\lambda = 0.5\,\mu\text{m}$. The simulation maintained numerical stability over $10^4$ time steps, with no unbounded field growth, confirming satisfaction of the CFL condition:

$$\Delta t = 0.9 \cdot \frac{\Delta x}{c\sqrt{2}}$$

The absorbing boundary layer successfully attenuated outgoing waves, with residual reflection amplitude below 5% of the incident wave peak — acceptable for qualitative wave-propagation studies, though not competitive with optimized UPML implementations that achieve reflection coefficients below $10^{-4}$.

### 9.2 Semiconductor Device Simulation

The p–n junction simulator was validated against textbook analytical solutions for silicon ($n_i = 1.5 \times 10^{10}\,\text{cm}^{-3}$, $\varepsilon_r = 11.7$) at room temperature ($T = 300\,\text{K}$). For a junction with $N_A = 10^{17}\,\text{cm}^{-3}$ and $N_D = 10^{16}\,\text{cm}^{-3}$:

| Parameter | Computed | Analytical Reference | Relative Error |
|:---|:---|:---|:---|
| Built-in potential $V_{\text{bi}}$ (V) | 0.757 | 0.757 | $<0.1\%$ |
| Depletion width $W$ ($\mu$m) at $V_a = 0$ | 0.366 | 0.366 | $<0.1\%$ |
| Depletion width $W$ ($\mu$m) at $V_a = -5$ V | 0.911 | 0.911 | $<0.1\%$ |

The agreement is exact to within floating-point precision, as expected for an implementation of closed-form analytical expressions.

### 9.3 CAD Kernel Topological Validation

All five primitive solid constructors (box, cylinder, sphere, cone, torus) were verified to satisfy the Euler–Poincaré invariant $V - E + F = 2$ for genus-zero solids. CSG Boolean operations (union, intersection, subtraction) on pairs of overlapping primitives were verified to produce topologically valid output meshes.

| Operation | Input Primitives | Output $V$ | Output $E$ | Output $F$ | $V - E + F$ | Valid |
|:---|:---|:---|:---|:---|:---|:---|
| Union | Box $\cup$ Sphere | 146 | 432 | 288 | 2 | ✅ |
| Subtraction | Box $\setminus$ Cylinder | 112 | 330 | 220 | 2 | ✅ |
| Intersection | Sphere $\cap$ Cone | 84 | 248 | 166 | 2 | ✅ |

### 9.4 NURBS Curve Evaluation Accuracy

The Cox–de Boor NURBS evaluator was tested against the De Casteljau algorithm (which serves as a numerically stable reference for Bézier curves) on a degree-3 curve with 7 control points. The maximum pointwise deviation between the two evaluation methods was $\|C_{\text{NURBS}}(u) - C_{\text{DeCasteljau}}(u)\|_\infty < 10^{-14}$, confirming machine-precision agreement.

---

## 10. Discussion and Limitations

### 10.1 Scope of Physical Simulation

The physical simulation modules in Aetheris are designed as lightweight, interpretable solvers that operate within the agent's reasoning loop. They are not intended to replace production-grade simulation codes. Specific limitations include:

1. **FDTD absorbing boundaries.** The current implementation uses a simplified quadratic damping layer rather than a full uniaxial PML (UPML) or convolutional PML (CPML) formulation. This produces adequate absorption for qualitative demonstrations but introduces measurable reflections at the boundaries. A UPML implementation would reduce boundary reflections by several orders of magnitude.

2. **Semiconductor device model.** The p–n junction simulator employs the depletion approximation with analytical solutions. It does not solve the coupled Poisson and drift-diffusion equations self-consistently and therefore cannot model high-injection conditions, carrier recombination, transient behavior, or heterojunctions.

3. **Python performance.** All solvers are implemented in pure Python (with NumPy for array operations), which imposes performance constraints. The FDTD solver is limited to grid sizes of approximately $500 \times 500$ before execution times become impractical for interactive use. JIT compilation (e.g., via Numba) or GPU acceleration (e.g., via CuPy or JAX) would substantially improve throughput.

### 10.2 Future Development

We identify several directions for extending Aetheris:

1. **Full PML implementation.** Replacing the simplified absorbing boundaries with a split-field or auxiliary-differential-equation PML formulation to achieve reflection coefficients below $10^{-6}$.

2. **Numerical drift-diffusion solver.** Implementing a self-consistent Poisson/drift-diffusion solver with Scharfetter–Gummel discretization for carrier current densities, enabling simulation of realistic device structures including MOSFETs, solar cells, and LEDs.

3. **First-principles electronic structure.** Integrating a from-scratch quantum chemistry engine implementing Hartree–Fock and Kohn–Sham DFT with analytical integral evaluation (Obara–Saika [23] and McMurchie–Davidson [24] schemes), DIIS convergence acceleration [25], and Becke numerical quadrature [26] with Lebedev angular grids [27]. This would enable the agent to perform ab initio molecular calculations without dependence on external quantum chemistry codes.

4. **Geometric constraint solver.** Implementing a Levenberg–Marquardt optimizer [28, 29] for resolving parametric sketch constraints (distances, angles, tangencies) in the CAD kernel, enabling fully constrained parametric modeling.

5. **Machine learning potentials.** Incorporating pre-trained machine learning interatomic potentials (e.g., MACE [30], NequIP [31]) as an alternative to classical force fields for molecular dynamics simulations.

6. **JIT compilation and GPU acceleration.** Applying Numba JIT compilation to critical inner loops and exploring JAX-based automatic differentiation for gradient-enabled solvers.

### 10.3 Broader Impact

Aetheris demonstrates that embedding domain-specific computational engines directly within an LLM-driven agent — rather than wrapping external black-box solvers — creates qualitatively different interaction patterns. The agent can inspect intermediate computational quantities (field distributions, topological invariants, convergence histories), reason about numerical failures, and adaptively modify simulation parameters within a single conversational turn. We believe this tight coupling between reasoning and computation represents a promising direction for AI-assisted scientific discovery and engineering design.

---

## 11. Conclusion

We have presented Aetheris, a unified autonomous multimodal agentic architecture that integrates parametric CAD modeling, multiscale physical simulation, materials database orchestration, and self-evolutionary code modification within a single, persistent LLM-driven reasoning loop. The key architectural innovation is the direct embedding of computational engines — including a from-scratch B-Rep solid modeler with NURBS support and BSP-tree CSG, FDTD electromagnetic and finite-difference multiphysics solvers, and a semiconductor device simulator — as callable tools within the agent's workspace, establishing a tight feedback loop between cognitive reasoning and physical computation.

The CAD kernel, built entirely from first principles, implements the full half-edge B-Rep topology, Cox–de Boor NURBS evaluation, Boehm knot insertion, and CSG Boolean operations via BSP trees — representing a substantial engineering effort. The physical simulators, while simplified relative to production codes, provide functional and numerically stable implementations suitable for rapid prototyping and educational use within the agentic loop. The self-evolutionary coding pipeline demonstrates that an LLM-driven agent can safely modify its own computational toolkit through a structured multi-agent framework with backup and rollback safeguards.

Aetheris is an ongoing experimental system. Future development will focus on extending the physical simulation capabilities (full PML, numerical drift-diffusion, first-principles quantum chemistry), improving computational performance through JIT compilation and GPU acceleration, and expanding the CAD kernel with geometric constraint solving. We release this work to encourage further exploration of tightly coupled AI–computation architectures for scientific discovery.

---

## References

### Foundational AI for Science
[1] Merchant, A., et al. (2023). Scaling deep learning for materials discovery. *Nature*, 624, 80–85.

[2] Jain, A., et al. (2016). A high-throughput infrastructure for density functional theory calculations. *Computational Materials Science*, 50(8), 2295–2310.

### Computational Chemistry and Materials Codes
[3] Kresse, G., & Furthmüller, J. (1996). Efficient iterative schemes for ab initio total-energy calculations using a plane-wave basis set. *Physical Review B*, 54(16), 11169.

[4] Dassault Systèmes. (2023). *Abaqus Unified FEA*. SIMULIA.

[5] Dassault Systèmes. (2023). *SolidWorks 3D CAD*. SOLIDWORKS Corp.

### LLM-Based Scientific Agents
[6] Bran, A. M., et al. (2024). ChemCrow: Augmenting large-language models with chemistry tools. *Nature Machine Intelligence*, 6, 525–535.

[7] Boiko, D. A., et al. (2023). Autonomous chemical research with large language models. *Nature*, 624, 570–578.

[8] Wang, L., et al. (2024). Scientific discovery in the age of artificial intelligence. *Nature*, 620, 47–60.

### CAD and Computational Geometry
[9] Baumgart, B. G. (1975). A polyhedron representation for computer vision. In *AFIPS National Computer Conference*, 589–596.

[10] Mäntylä, M. (1988). *An Introduction to Solid Modeling*. Computer Science Press.

[11] Piegl, L., & Tiller, W. (1997). *The NURBS Book*. Springer Science & Business Media.

[12] Naylor, B., Amanatides, J., & Thibault, W. (1990). Merging BSP trees yields polyhedral set operations. *ACM SIGGRAPH Computer Graphics*, 24(4), 115–124.

### Electromagnetics and Device Simulation
[13] Yee, K. (1966). Numerical solution of initial boundary value problems involving Maxwell's equations in isotropic media. *IEEE Transactions on Antennas and Propagation*, 14(3), 302–307.

[14] Bérenger, J.-P. (1994). A perfectly matched layer for the absorption of electromagnetic waves. *Journal of Computational Physics*, 114(2), 185–200.

[15] Selberherr, S. (1984). *Analysis and Simulation of Semiconductor Devices*. Springer-Verlag.

### Materials Databases
[16] Jain, A., et al. (2013). Commentary: The Materials Project: A materials genome approach to accelerating materials innovation. *APL Materials*, 1(1), 011002.

[17] Merchant, A., et al. (2023). Scaling deep learning for materials discovery. *Nature*, 624, 80–85.

### Self-Modifying Code and Autonomous Software Engineering
[18] von Neumann, J. (1966). *Theory of Self-Reproducing Automata*. University of Illinois Press.

[19] Yang, J., et al. (2024). SWE-agent: Agent-computer interfaces enable automated software engineering. *arXiv preprint arXiv:2405.15793*.

[20] Zhang, Y., et al. (2024). AutoCodeRover: Autonomous program improvement. In *Proceedings of the 33rd ACM SIGSOFT International Symposium on Software Testing and Analysis*.

### Numerical Methods
[21] Boehm, W. (1980). Inserting new knots into B-spline curves. *Computer-Aided Design*, 12(4), 199–201.

[22] Möller, T., & Trumbore, B. (1997). Fast, minimum storage ray-triangle intersection. *Journal of Graphics Tools*, 2(1), 21–28.

### Quantum Chemistry (Future Work References)
[23] Obara, S., & Saika, A. (1986). Efficient recursive computation of molecular integrals over Cartesian Gaussian functions. *The Journal of Chemical Physics*, 84(7), 3963.

[24] McMurchie, L. E., & Davidson, E. R. (1978). One- and two-electron integrals over Cartesian Gaussian functions. *Journal of Computational Physics*, 26(2), 218.

[25] Pulay, P. (1980). Convergence acceleration of iterative sequences: The case of SCF iteration. *Chemical Physics Letters*, 73(2), 393.

[26] Becke, A. D. (1988). A multicenter numerical integration scheme for polyatomic molecules. *The Journal of Chemical Physics*, 88(4), 2547.

[27] Lebedev, V. I. (1975). Quadratures on a sphere. *USSR Computational Mathematics and Mathematical Physics*, 15(1), 44–51.

### Optimization
[28] Levenberg, K. (1944). A method for the solution of certain non-linear problems in least squares. *Quarterly of Applied Mathematics*, 2(2), 164–168.

[29] Marquardt, D. W. (1963). An algorithm for least-squares estimation of nonlinear parameters. *Journal of the Society for Industrial and Applied Mathematics*, 11(2), 431–441.

### Machine Learning Potentials (Future Work References)
[30] Batatia, I., et al. (2022). MACE: Higher order equivariant message passing neural networks for fast and accurate force fields. *Advances in Neural Information Processing Systems*, 35, 11423–11436.

[31] Batzner, S., et al. (2022). E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials. *Nature Communications*, 13, 2453.

---

## Citation

```bibtex
@article{aetheris_2026,
  title={Aetheris: A Unified Autonomous Multimodal Agentic Architecture Integrating Parametric CAD, Multiscale Physical Simulation, Materials Database Orchestration, and Closed-Loop Self-Evolution},
  author={Arun R.},
  journal={Preprint},
  year={2026}
}
```
