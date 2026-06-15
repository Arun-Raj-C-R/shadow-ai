import sys
sys.stdout.reconfigure(encoding='utf-8')

f = r'd:\Project File\Shadow\Shadow\Brain\Shadow2\cli\code5.py'
c = open(f, 'r', encoding='utf-8').read()

old = '- "render_and_plot" \u2192 Render LaTeX AND generate an interactive Plotly graph using AI'
new = ('- "render_and_plot" \u2192 Render LaTeX AND show an interactive visualization\n'
       '- "derive"          \u2192 FULL step-by-step derivation: left=derivation, right=interactive plot\n'
       '\n'
       'ALWAYS use task="derive" when user asks to derive, prove, or explain step-by-step.\n'
       'The tool auto-detects the physics topic and generates the matching visualization.\n'
       'Set generate_plot=true. Provide the query in natural language.')

if old in c:
    c = c.replace(old, new, 1)
    open(f, 'w', encoding='utf-8').write(c)
    print('REPLACED successfully')
else:
    print('NOT FOUND')
