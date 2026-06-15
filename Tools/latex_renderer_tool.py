"""
SHADOW Derivation & Visualization Engine
==========================================
Split-panel real-time derivation tool:
  - Left panel: Step-by-step derivation with auto-scroll + typewriter reveal
  - Right panel: Interactive 2D/3D visualization (Plotly + Three.js)

Dependencies: openai (for NVIDIA API), dotenv
"""

import os
import json
import re
import logging
import traceback
import webbrowser
from datetime import datetime
from typing import Dict

from dotenv import load_dotenv
load_dotenv()

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
RENDER_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "latex_renders")
os.makedirs(RENDER_OUTPUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("latex_renderer_tool")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# HTML TEMPLATE â€” Split-panel derivation + visualization
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SHADOW Derivation Engine</title>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

* { margin:0; padding:0; box-sizing:border-box; }
:root {
  --bg: #0f0f1a; --surface: #1a1a2e; --surface2: #16213e;
  --border: #2a2a4a; --text: #e0e0f0; --text-dim: #8888aa;
  --accent: #7c3aed; --accent2: #06b6d4; --accent3: #f59e0b;
  --eq-bg: #111128; --success: #22c55e; --glow: rgba(124,58,237,0.15);
}
html,body { height:100%; font-family:'Inter',sans-serif; background:var(--bg); color:var(--text); overflow:hidden; }

/* â”€â”€ Header â”€â”€ */
.header {
  height:48px; display:flex; align-items:center; padding:0 20px;
  background:linear-gradient(135deg, var(--surface) 0%, var(--surface2) 100%);
  border-bottom:1px solid var(--border);
}
.header h1 { font-size:15px; font-weight:600; letter-spacing:1px; }
.header h1 span { color:var(--accent); }
.header .badge { margin-left:auto; font-size:11px; color:var(--accent2);
  background:rgba(6,182,212,0.1); padding:3px 10px; border-radius:20px; border:1px solid rgba(6,182,212,0.2); }

/* â”€â”€ Split Layout â”€â”€ */
.split { display:flex; height:calc(100vh - 48px); }

/* â”€â”€ Derivation Panel (left) â”€â”€ */
.derivation-panel {
  width:50%; overflow-y:auto; padding:24px 28px;
  border-right:1px solid var(--border);
  scroll-behavior:smooth;
}
.derivation-panel::-webkit-scrollbar { width:6px; }
.derivation-panel::-webkit-scrollbar-track { background:var(--bg); }
.derivation-panel::-webkit-scrollbar-thumb { background:var(--border); border-radius:3px; }

.deriv-title { font-size:20px; font-weight:700; margin-bottom:20px;
  background:linear-gradient(90deg, var(--accent), var(--accent2));
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; }

/* Step card */
.step { opacity:0; transform:translateY(16px); margin-bottom:32px;
  animation:stepIn 0.5s ease forwards; border-left:3px solid var(--accent);
  padding-left:16px; }
.step .step-num { font-size:12px; font-weight:700; text-transform:uppercase;
  color:var(--accent); letter-spacing:1.5px; margin-bottom:10px;
  display:block; }
.step .equation {
  background:var(--eq-bg); padding:18px 22px; border-radius:8px;
  font-size:14px; overflow-x:auto; margin:12px 0 16px 0;
  border:1px solid var(--border); box-shadow:0 2px 12px var(--glow);
  display:block;
}
.step .explanation { font-size:14px; line-height:1.8; color:var(--text-dim);
  margin-top:14px; padding:4px 6px; display:block; }

@keyframes stepIn { to { opacity:1; transform:translateY(0); } }

/* Stagger animations */
.step:nth-child(1) { animation-delay:0.1s; }
.step:nth-child(2) { animation-delay:0.8s; }
.step:nth-child(3) { animation-delay:1.5s; }
.step:nth-child(4) { animation-delay:2.2s; }
.step:nth-child(5) { animation-delay:2.9s; }
.step:nth-child(6) { animation-delay:3.6s; }
.step:nth-child(7) { animation-delay:4.3s; }
.step:nth-child(8) { animation-delay:5.0s; }
.step:nth-child(9) { animation-delay:5.7s; }
.step:nth-child(10){ animation-delay:6.4s; }

/* â”€â”€ Visualization Panel (right) â”€â”€ */
.viz-panel {
  width:50%; display:flex; flex-direction:column; background:var(--surface);
}
.viz-header { padding:12px 20px; font-size:13px; font-weight:600;
  color:var(--accent2); border-bottom:1px solid var(--border);
  display:flex; align-items:center; gap:8px; }
.viz-header::before { content:''; width:8px; height:8px; border-radius:50%;
  background:var(--success); animation:pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.4;} }

#viz-container { flex:1; position:relative; }
#plot2d { width:100%; height:100%; min-height:400px; }
#plot3d { width:100%; height:100%; min-height:400px; position:absolute; top:0; left:0; display:none; }

/* â”€â”€ Render-only mode (no split) â”€â”€ */
.render-only { max-width:800px; margin:40px auto; padding:30px; }
.render-only .equation { background:var(--eq-bg); padding:20px; border-radius:8px;
  font-size:1.3em; text-align:center; border:1px solid var(--border);
  box-shadow:0 4px 20px var(--glow); margin:20px 0; }
.render-only .explanation { font-size:15px; line-height:1.8; color:var(--text-dim); }
</style>
</head>
<body>

<div class="header">
  <h1><span>SHADOW</span> Derivation Engine</h1>
  <div class="badge">%%BADGE%%</div>
</div>

<div class="split" id="splitView" style="display:%%SPLIT_DISPLAY%%">
  <div class="derivation-panel" id="derivPanel">
    <div class="deriv-title">%%TITLE%%</div>
    %%STEPS_HTML%%
  </div>
  <div class="viz-panel">
    <div class="viz-header">Interactive Visualization</div>
    <div id="viz-container">
      <div id="plot2d"></div>
      <canvas id="plot3d"></canvas>
    </div>
  </div>
</div>

<div class="render-only" id="renderOnly" style="display:%%RENDER_DISPLAY%%">
  %%RENDER_HTML%%
</div>

<script>
// Auto-scroll derivation panel as steps appear
(function() {
  const panel = document.getElementById('derivPanel');
  if (!panel) return;
  const steps = panel.querySelectorAll('.step');
  const delays = [];
  steps.forEach((s, i) => { delays.push(100 + i * 700); });
  delays.forEach((d, i) => {
    setTimeout(() => {
      steps[i].scrollIntoView({ behavior: 'smooth', block: 'end' });
    }, d + 400);
  });
})();

// Visualization code injected by AI
%%VIZ_CODE%%
</script>
</body>
</html>"""


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# AI DERIVATION ENGINE â€” Groq (GPT-OSS 120B @ ~500 tok/s)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

def _call_derivation_ai(query: str, latex_hint: str) -> Dict:
    """Call Groq GPT-OSS 120B for ultra-fast step-by-step derivation."""
    api_key = GROQ_API_KEY or NVIDIA_API_KEY
    if not api_key:
        return {"error": "No API key (GROQ_API_KEY or NVIDIA_API_KEY)", "steps": []}

    try:
        from openai import OpenAI
    except ImportError:
        return {"error": "openai not installed", "steps": []}

    # Use Groq if key available, else fall back to NVIDIA
    if GROQ_API_KEY:
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
        model = "openai/gpt-oss-120b"
    else:
        client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)
        model = "meta/llama-3.1-70b-instruct"

    system_prompt = r"""You are an expert physicist and mathematician. Return ONLY valid JSON.

Return a JSON object with this structure:
{
  "title": "Short title",
  "steps": [
    {"equation": "LaTeX equation using $$ delimiters", "explanation": "1-2 sentence explanation"},
    ...more steps...
  ]
}

RULES:
- 4-8 derivation steps. Each: ONE equation + SHORT explanation.
- Valid LaTeX. Use $$ for display math.
- JSON ONLY. No markdown wrapping."""

    user_prompt = f"Derive step by step: {query}"
    if latex_hint:
        user_prompt += f"\nReference: {latex_hint}"

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1500,
        )
        raw = completion.choices[0].message.content.strip()

        # Robust JSON extraction
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r'\{[\s\S]*\}', raw)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    data = {"title": "Derivation", "steps": [{"equation": latex_hint or query, "explanation": raw[:200]}]}
            else:
                data = {"title": "Derivation", "steps": [{"equation": latex_hint or query, "explanation": raw[:200]}]}

        return data

    except Exception as e:
        logger.error(f"Derivation API error: {e}")
        return {"error": str(e), "title": "Derivation", "steps": []}


def _build_viz(query: str, latex: str) -> str:
    """Build reliable Plotly visualization based on topic detection."""
    q = (query + " " + latex).lower()
    DL = """paper_bgcolor:'#1a1a2e',plot_bgcolor:'#1a1a2e',font:{color:'#e0e0f0'},
      xaxis:{gridcolor:'#2a2a4a',zerolinecolor:'#2a2a4a'},
      yaxis:{gridcolor:'#2a2a4a',zerolinecolor:'#2a2a4a'},
      margin:{t:40,b:40,l:50,r:20},legend:{bgcolor:'rgba(0,0,0,0)',font:{color:'#e0e0f0'}}"""

    # â”€â”€ Quantum: infinite potential well â”€â”€
    if any(w in q for w in ["potential well","particle in a box","infinite well","quantum well"]):
        return f"""(function(){{
        var x=[],p1=[],p2=[],p3=[],pd1=[],pd2=[],pd3=[];
        for(var i=0;i<=300;i++){{var xi=i/300;
          x.push(xi);
          p1.push(Math.sqrt(2)*Math.sin(Math.PI*xi));
          p2.push(Math.sqrt(2)*Math.sin(2*Math.PI*xi));
          p3.push(Math.sqrt(2)*Math.sin(3*Math.PI*xi));
          pd1.push(2*Math.pow(Math.sin(Math.PI*xi),2));
          pd2.push(2*Math.pow(Math.sin(2*Math.PI*xi),2));
          pd3.push(2*Math.pow(Math.sin(3*Math.PI*xi),2));
        }}
        Plotly.newPlot('plot2d',[
          {{x:x,y:p1,name:'\u03c8\u2081 (n=1)',line:{{color:'#7c3aed',width:2}}}},
          {{x:x,y:p2,name:'\u03c8\u2082 (n=2)',line:{{color:'#06b6d4',width:2}}}},
          {{x:x,y:p3,name:'\u03c8\u2083 (n=3)',line:{{color:'#f59e0b',width:2}}}},
          {{x:x,y:pd1,name:'|\u03c8\u2081|\u00b2',line:{{color:'#7c3aed',width:1,dash:'dot'}}}},
          {{x:x,y:pd2,name:'|\u03c8\u2082|\u00b2',line:{{color:'#06b6d4',width:1,dash:'dot'}}}},
          {{x:x,y:pd3,name:'|\u03c8\u2083|\u00b2',line:{{color:'#f59e0b',width:1,dash:'dot'}}}}
        ],{{title:'Wavefunctions & Probability Densities',{DL},
          xaxis:{{title:'x/L',gridcolor:'#2a2a4a'}},yaxis:{{title:'Amplitude',gridcolor:'#2a2a4a'}}}});
        }})();"""

    # â”€â”€ Quantum: harmonic oscillator â”€â”€
    if any(w in q for w in ["harmonic oscillator","quantum oscillator","hermite"]):
        return f"""(function(){{
        function hermite(n,x){{if(n==0)return 1;if(n==1)return 2*x;return 2*x*hermite(n-1,x)-2*(n-1)*hermite(n-2,x);}}
        function psi(n,x){{var norm=Math.pow(Math.PI,-0.25)/Math.sqrt(Math.pow(2,n)*([1,1,2,6,24][n]||1));
          return norm*hermite(n,x)*Math.exp(-x*x/2);}}
        var x=[]; for(var i=-5;i<=5;i+=0.05)x.push(i);
        var traces=[];
        var colors=['#7c3aed','#06b6d4','#f59e0b','#22c55e','#f43f5e'];
        for(var n=0;n<5;n++){{var y=x.map(function(xi){{return psi(n,xi)+n;}});
          traces.push({{x:x,y:y,name:'n='+n+' (E='+(n+0.5)+'\u0127\u03c9)',line:{{color:colors[n],width:2}}}});
          traces.push({{x:x,y:x.map(function(xi){{return 0.5*xi*xi+n;}}),name:'',line:{{color:colors[n],width:1,dash:'dot'}},showlegend:false}});}}
        Plotly.newPlot('plot2d',traces,{{title:'Quantum Harmonic Oscillator Wavefunctions',{DL}}});
        }})();"""

    # â”€â”€ Perturbation theory â”€â”€
    if any(w in q for w in ["perturbation","perturbed","first order correction","energy correction"]):
        return f"""(function(){{
        var lam=[],E0=[],E1=[],E2=[]; 
        for(var l=0;l<=1;l+=0.01){{lam.push(l);
          E0.push(1); E1.push(1+l*0.5); E2.push(1+l*0.5+l*l*0.1);}}
        var lam2=[],Eb0=[],Eb1=[],Eb2=[];
        for(var l=0;l<=1;l+=0.01){{lam2.push(l);
          Eb0.push(4); Eb1.push(4+l*0.3); Eb2.push(4+l*0.3+l*l*(-0.05));}}
        Plotly.newPlot('plot2d',[
          {{x:lam,y:E0,name:'E\u2080\u207d\u2070\u207e (unperturbed)',line:{{color:'#7c3aed',width:2,dash:'dot'}}}},
          {{x:lam,y:E1,name:'E\u2080 + 1st order',line:{{color:'#06b6d4',width:2}}}},
          {{x:lam,y:E2,name:'E\u2080 + 2nd order',line:{{color:'#f59e0b',width:2}}}},
          {{x:lam2,y:Eb0,name:'E\u2081\u207d\u2070\u207e',line:{{color:'#22c55e',width:2,dash:'dot'}}}},
          {{x:lam2,y:Eb1,name:'E\u2081 + 1st',line:{{color:'#f43f5e',width:2}}}},
          {{x:lam2,y:Eb2,name:'E\u2081 + 2nd',line:{{color:'#a855f7',width:2}}}}
        ],{{title:'Energy Levels vs Perturbation Strength \u03bb',{DL},
          xaxis:{{title:'\u03bb (perturbation strength)',gridcolor:'#2a2a4a'}},
          yaxis:{{title:'Energy',gridcolor:'#2a2a4a'}}}});
        }})();"""

    # â”€â”€ Band structure / Bloch / condensed matter â”€â”€
    if any(w in q for w in ["band structure","bloch","brillouin","kronig","solid state","condensed matter","lattice","crystal"]):
        return f"""(function(){{
        var k=[],E1=[],E2=[],E3=[],Ef=[];
        for(var i=-314;i<=314;i++){{var ki=i/100; k.push(ki);
          E1.push(-2*Math.cos(ki)+0.5*ki*ki*0.1);
          E2.push(2-1.5*Math.cos(ki)+0.3*ki*ki*0.1+3);
          E3.push(1-Math.cos(2*ki)+7);
          Ef.push(4.5);
        }}
        Plotly.newPlot('plot2d',[
          {{x:k,y:E1,name:'Valence Band',line:{{color:'#7c3aed',width:2}}}},
          {{x:k,y:E2,name:'Conduction Band',line:{{color:'#06b6d4',width:2}}}},
          {{x:k,y:E3,name:'Upper Band',line:{{color:'#f59e0b',width:2}}}},
          {{x:k,y:Ef,name:'Fermi Level',line:{{color:'#ef4444',width:2,dash:'dash'}}}}
        ],{{title:'Electronic Band Structure',{DL},
          xaxis:{{title:'k (wave vector)',gridcolor:'#2a2a4a'}},
          yaxis:{{title:'E (energy)',gridcolor:'#2a2a4a'}}}});
        }})();"""

    # â”€â”€ Schrodinger / wave equation â”€â”€
    if any(w in q for w in ["schrodinger","schroedinger","wave equation","wavefunction","psi"]):
        return f"""(function(){{
        var x=[],re=[],im=[],prob=[];
        var k=5,sigma=0.5;
        for(var i=-5;i<=5;i+=0.02){{x.push(i);
          var env=Math.exp(-i*i/(2*sigma*sigma));
          re.push(env*Math.cos(k*i)); im.push(env*Math.sin(k*i)); prob.push(env*env);}}
        Plotly.newPlot('plot2d',[
          {{x:x,y:re,name:'Re(\u03c8)',line:{{color:'#7c3aed',width:2}}}},
          {{x:x,y:im,name:'Im(\u03c8)',line:{{color:'#06b6d4',width:2}}}},
          {{x:x,y:prob,name:'|\u03c8|\u00b2',line:{{color:'#f59e0b',width:3}},fill:'tozeroy',fillcolor:'rgba(245,158,11,0.15)'}}
        ],{{title:'Gaussian Wave Packet',{DL},
          xaxis:{{title:'x',gridcolor:'#2a2a4a'}},yaxis:{{title:'Amplitude',gridcolor:'#2a2a4a'}}}});
        }})();"""

    # â”€â”€ Maxwell / electromagnetic / AC circuits â”€â”€
    if any(w in q for w in ["maxwell","electromagnetic","em wave","ac circuit","johnson","impedance","rlc"]):
        return f"""(function(){{
        var t=[],V=[],I=[],P=[];
        var f=50,R=100,L=0.1,C=0.0001,omega=2*Math.PI*f;
        var Z=Math.sqrt(R*R+Math.pow(omega*L-1/(omega*C),2));
        var phi=Math.atan((omega*L-1/(omega*C))/R);
        for(var i=0;i<=400;i++){{var ti=i*0.02/f; t.push(ti*1000);
          V.push(220*Math.sin(omega*ti));
          I.push((220/Z)*Math.sin(omega*ti-phi)*100);
          P.push(220*(220/Z)*Math.sin(omega*ti)*Math.sin(omega*ti-phi)/100);}}
        Plotly.newPlot('plot2d',[
          {{x:t,y:V,name:'Voltage (V)',line:{{color:'#7c3aed',width:2}}}},
          {{x:t,y:I,name:'Current x100 (A)',line:{{color:'#06b6d4',width:2}}}},
          {{x:t,y:P,name:'Power/100 (W)',line:{{color:'#f59e0b',width:2}},fill:'tozeroy',fillcolor:'rgba(245,158,11,0.1)'}}
        ],{{title:'AC Circuit: V, I, Power (R='+R+'\u03a9, f='+f+'Hz)',{DL},
          xaxis:{{title:'Time (ms)',gridcolor:'#2a2a4a'}},yaxis:{{title:'Amplitude',gridcolor:'#2a2a4a'}}}});
        }})();"""

    # â”€â”€ Thermodynamics â”€â”€
    if any(w in q for w in ["thermodynamic","entropy","carnot","heat engine","boltzmann","partition function","free energy"]):
        return f"""(function(){{
        var T=[],S=[],F=[],Cv=[];
        for(var i=1;i<=500;i++){{T.push(i);
          S.push(3*1.38e-23*6.022e23*Math.log(i/300)+50);
          F.push(-1.38e-23*6.022e23*i*Math.log(i/300));
          Cv.push(3*1.38e-23*6.022e23*(1-150*150/(i*i))*Math.exp(-150/i));}}
        Plotly.newPlot('plot2d',[
          {{x:T,y:S,name:'Entropy S(T)',line:{{color:'#7c3aed',width:2}},yaxis:'y'}},
          {{x:T,y:Cv,name:'Heat Capacity Cv(T)',line:{{color:'#06b6d4',width:2}}}}
        ],{{title:'Thermodynamic Properties vs Temperature',{DL},
          xaxis:{{title:'T (K)',gridcolor:'#2a2a4a'}},yaxis:{{title:'Value',gridcolor:'#2a2a4a'}}}});
        }})();"""

    # â”€â”€ Generic trig / wave â”€â”€
    if any(w in q for w in ["sin","cos","tan","wave","fourier","oscillat"]):
        return f"""(function(){{
        var x=[],y1=[],y2=[],y3=[];
        for(var i=-10;i<=10;i+=0.05){{x.push(i);
          y1.push(Math.sin(i)); y2.push(Math.sin(2*i)/2); y3.push(Math.sin(i)+Math.sin(2*i)/2+Math.sin(3*i)/3);}}
        Plotly.newPlot('plot2d',[
          {{x:x,y:y1,name:'sin(x)',line:{{color:'#7c3aed',width:2}}}},
          {{x:x,y:y2,name:'sin(2x)/2',line:{{color:'#06b6d4',width:2}}}},
          {{x:x,y:y3,name:'Fourier Sum',line:{{color:'#f59e0b',width:3}}}}
        ],{{title:'Wave Components & Fourier Superposition',{DL}}});
        }})();"""

    # â”€â”€ Exponential / decay â”€â”€
    if any(w in q for w in ["exponential","decay","half life","radioactive","e^","exp"]):
        return f"""(function(){{
        var t=[],N=[],dN=[]; var N0=1000,lam=0.1;
        for(var i=0;i<=50;i+=0.2){{t.push(i); N.push(N0*Math.exp(-lam*i)); dN.push(N0*lam*Math.exp(-lam*i));}}
        Plotly.newPlot('plot2d',[
          {{x:t,y:N,name:'N(t) = N\u2080 e^(-\u03bbt)',line:{{color:'#7c3aed',width:2}}}},
          {{x:t,y:dN,name:'dN/dt (decay rate)',line:{{color:'#ef4444',width:2,dash:'dot'}}}}
        ],{{title:'Exponential Decay',{DL},xaxis:{{title:'Time',gridcolor:'#2a2a4a'}},yaxis:{{title:'N',gridcolor:'#2a2a4a'}}}});
        }})();"""

    # â”€â”€ Quadratic / polynomial â”€â”€
    if any(w in q for w in ["x^2","quadratic","parabola","polynomial"]):
        return f"""(function(){{
        var x=[],y1=[],y2=[];
        for(var i=-5;i<=5;i+=0.1){{x.push(i); y1.push(i*i); y2.push(i*i*i-3*i);}}
        Plotly.newPlot('plot2d',[
          {{x:x,y:y1,name:'x\u00b2',line:{{color:'#7c3aed',width:2}}}},
          {{x:x,y:y2,name:'x\u00b3 - 3x',line:{{color:'#06b6d4',width:2}}}}
        ],{{title:'Polynomial Functions',{DL}}});
        }})();"""

    # â”€â”€ Default: generic function plot â”€â”€
    return f"""(function(){{
    var x=[],y1=[],y2=[];
    for(var i=-5;i<=5;i+=0.05){{x.push(i); y1.push(Math.sin(i)*Math.exp(-0.1*i*i)); y2.push(Math.cos(i)*Math.exp(-0.1*i*i));}}
    Plotly.newPlot('plot2d',[
      {{x:x,y:y1,name:'f(x)',line:{{color:'#7c3aed',width:2}}}},
      {{x:x,y:y2,name:'g(x)',line:{{color:'#06b6d4',width:2}}}}
    ],{{title:'Mathematical Visualization',{DL}}});
    }})();"""


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MAIN TOOL ENTRY POINT
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def run_latex_renderer(
    task: str,
    latex_content: str = "",
    query: str = "",
    generate_plot: bool = False
) -> str:
    """
    Main tool entry point.

    Args:
        task: "render_only", "render_and_plot", or "derive" (full step-by-step derivation)
        latex_content: LaTeX string to render
        query: Natural language description
        generate_plot: Whether to generate visualization

    Returns:
        JSON string with results and HTML file path.
    """
    start = datetime.now()
    print(f"\n{'='*60}")
    print(f"  DERIVATION ENGINE | Task: {task}")
    print(f"  Time: {start.strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")

    output = {"task": task, "timestamp": start.isoformat(), "latex": latex_content}

    try:
        if task in ("render_and_plot", "derive") or generate_plot:
            # Full derivation mode
            print(">> Calling Groq GPT-OSS 120B for derivation...")
            ai_data = _call_derivation_ai(query or latex_content, latex_content)

            if ai_data.get("error"):
                print(f"!! API Error: {ai_data['error']}")
                output["ai_error"] = ai_data["error"]

            title = ai_data.get("title", "Derivation")
            steps = ai_data.get("steps", [])
            # Always build reliable visualization locally
            viz_code = _build_viz(query or latex_content, latex_content)

            # Build steps HTML
            steps_html = ""
            for i, step in enumerate(steps):
                eq = step.get("equation", "")
                exp = step.get("explanation", "")
                # Ensure display math
                if eq and not eq.strip().startswith("$$") and not eq.strip().startswith("\\["):
                    eq = f"$$ {eq} $$"
                steps_html += f"""<div class="step">
  <div class="step-num">Step {i+1}</div>
  <div class="equation">{eq}</div>
  <div class="explanation">{exp}</div>
</div>\n"""

            if not steps_html:
                # Fallback: render the raw latex as a single step
                display = latex_content.strip()
                if display and not display.startswith("$$"):
                    display = f"$$ {display} $$"
                steps_html = f"""<div class="step">
  <div class="step-num">Result</div>
  <div class="equation">{display}</div>
  <div class="explanation">{query}</div>
</div>"""


            # Compile HTML
            html = HTML_TEMPLATE
            html = html.replace("%%TITLE%%", title)
            html = html.replace("%%STEPS_HTML%%", steps_html)
            html = html.replace("%%VIZ_CODE%%", viz_code)
            html = html.replace("%%BADGE%%", "Interactive Plot")
            html = html.replace("%%SPLIT_DISPLAY%%", "flex")
            html = html.replace("%%RENDER_DISPLAY%%", "none")
            html = html.replace("%%RENDER_HTML%%", "")

            output["steps_count"] = len(steps)
            output["plot_generated"] = True
            print(f"-> Generated {len(steps)} derivation steps + visualization")

        else:
            # Simple render-only mode
            display = latex_content.strip()
            if display and not display.startswith("$$") and not display.startswith("\\["):
                display = f"$$ {display} $$"

            render_html = f"""<div class="equation">{display}</div>
<div class="explanation">{query}</div>"""

            html = HTML_TEMPLATE
            html = html.replace("%%TITLE%%", "")
            html = html.replace("%%STEPS_HTML%%", "")
            html = html.replace("%%VIZ_CODE%%", "")
            html = html.replace("%%BADGE%%", "LaTeX Render")
            html = html.replace("%%SPLIT_DISPLAY%%", "none")
            html = html.replace("%%RENDER_DISPLAY%%", "block")
            html = html.replace("%%RENDER_HTML%%", render_html)
            print("-> Render-only mode")

        # Save and open
        fid = f"deriv_{int(start.timestamp())}.html"
        path = os.path.join(RENDER_OUTPUT_DIR, fid)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

        output["html_file_path"] = path
        print(f"-> Opening in browser...")
        webbrowser.open(f"file://{os.path.abspath(path)}")
        output["status"] = "success"

    except Exception as e:
        output["error"] = str(e)
        output["traceback"] = traceback.format_exc()
        print(f"XX Error: {e}")
        output["status"] = "error"

    elapsed = (datetime.now() - start).total_seconds()
    output["elapsed_seconds"] = round(elapsed, 2)
    print(f"\n>> COMPLETE ({elapsed:.1f}s)\n{'='*60}")
    return json.dumps(output, indent=2, default=str)


if __name__ == "__main__":
    result = run_latex_renderer(
        task="derive",
        latex_content=r"\psi(x) = A\sin(n\pi x / L)",
        query="Derive the wavefunction for a particle in an infinite potential well",
        generate_plot=True
    )
    print(result)
