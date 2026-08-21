"""Structural sanity check for paper/main.tex (no LaTeX toolchain required)."""

import os
import re
import collections

TEX_PATH = os.path.join('paper', 'main.tex')
TEX_DIR = os.path.dirname(TEX_PATH)

s = open(TEX_PATH, encoding='utf-8').read()

begins = re.findall(r'\\begin\{(\w+\*?)\}', s)
ends = re.findall(r'\\end\{(\w+\*?)\}', s)
cb, ce = collections.Counter(begins), collections.Counter(ends)
bad_env = {k: (cb[k], ce[k]) for k in set(cb) | set(ce) if cb[k] != ce[k]}
print("env mismatch:", bad_env or "none")

labels = set(re.findall(r'\\label\{([^}]+)\}', s))
refs = set(re.findall(r'\\(?:eq)?ref\{([^}]+)\}', s))
print("undefined refs:", refs - labels or "none")
print("unused labels:", labels - refs or "none")

bib = set(re.findall(r'\\bibitem\{([^}]+)\}', s))
cites = set()
for c in re.findall(r'\\cite\{([^}]+)\}', s):
    cites |= {x.strip() for x in c.split(',')}
print("undefined cites:", cites - bib or "none")
print("uncited bibitems:", sorted(bib - cites) or "none")

BACKSLASH = chr(92)
for m in re.finditer(r'\\begin\{tabular\}\{((?:[^{}]|\{[^{}]*\})*)\}(.*?)\\end\{tabular\}',
                     s, re.S):
    spec, body = m.group(1), m.group(2)
    clean = re.sub(r'@\{[^{}]*\}', '', spec)
    ncol = len(re.findall(r'[lcr]|p\{[^}]*\}', clean))
    bad = []
    for line in body.split(BACKSLASH * 2):
        line = re.sub(r'\\multicolumn\{(\d+)\}\{[^}]*\}\{[^}]*\}',
                      lambda mm: '&' * (int(mm.group(1)) - 1), line)
        line = re.sub(r'\\multirow\{[^}]*\}\{[^}]*\}\{[^}]*\}', '', line)
        core = re.sub(r'(?<!\\)%.*', '', line)
        if '&' not in core:
            continue
        n = core.count('&') + 1
        if n != ncol:
            bad.append((n, ' '.join(core.split())[:70]))
    print(f"tabular spec={spec!r} ncol={ncol} bad_rows={len(bad)}")
    for n, t in bad[:8]:
        print(f"    {n} cols: {t}")

print("dollars balanced:", s.count('$') % 2 == 0)
braces = s.count('{') - s.count('}')
print("brace delta:", braces)

# \graphicspath{{../}{./}} means paths resolve against these roots.
roots = [os.path.join(TEX_DIR, '..'), TEX_DIR]
for fig in re.findall(r'\\includegraphics\[[^\]]*\]\{([^}]+)\}', s):
    found = any(os.path.exists(os.path.join(r, fig)) for r in roots)
    print(f"figure {fig}: resolves={found}")
