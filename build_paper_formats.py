"""
Render paper/main.tex to Markdown and Word.

The .tex is the single source of truth; both outputs are generated from
it so the three versions cannot drift. Layout follows the IEEE
conference convention used by the reference submission: numbered
sections in Roman numerals, lettered subsections, "TABLE n." captions
above each table, "Fig. n." captions below each figure, and a two-column
body in the Word version with wide floats breaking out to full width.

Requires python-docx. Run from the repository root:
    python build_paper_formats.py
"""

import os
import re

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Inches, RGBColor

TEX = os.path.join('paper', 'main.tex')
OUT_MD = os.path.join('paper', 'main.md')
OUT_DOCX = os.path.join('paper', 'main.docx')

ROMAN = ['', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X',
         'XI', 'XII']
LETTER = ' ABCDEFGHIJKLMNOPQRSTUVWXYZ'


# ------------------------------------------------------------------
# Inline LaTeX -> text
# ------------------------------------------------------------------
GREEK = {
    r'\lambda': 'lambda', r'\gamma': 'gamma', r'\varphi': 'phi',
    r'\pi': 'pi', r'\rho': 'rho', r'\tau': 'tau', r'\sigma': 'sigma',
    r'\Delta': 'Delta', r'\alpha': 'alpha', r'\mathcal': '',
    r'\varepsilon': 'eps', r'\bowtie': 'JOIN', r'\dagger': '+',
    r'\times': '×', r'\le': '<=', r'\ge': '>=', r'\in': 'in',
    r'\subseteq': '⊆', r'\rightarrow': '->', r'\to': '->',
    r'\approx': '≈', r'\pm': '±', r'\cdot': '·', r'\ldots': '...',
    r'\dots': '...', r'\exists': 'exists', r'\forall': 'for all',
    r'\neq': '≠', r'\equiv': '=', r'\notin': 'not in',
    r'\leftarrow': '<-', r'\hat': '', r'\bar': '',
}


def strip_math(s):
    """Turn inline math into readable plain text."""
    s = re.sub(r'\\mathrm\{([^{}]*)\}', r'\1', s)
    s = re.sub(r'\\mathcal\{([^{}]*)\}', r'\1', s)
    s = re.sub(r'\\text(?:bf|it|sc)?\{([^{}]*)\}', r'\1', s)
    s = re.sub(r'\\emph\{([^{}]*)\}', r'\1', s)
    s = re.sub(r'\\operatorname\{([^{}]*)\}', r'\1', s)
    # superscripts / subscripts
    s = re.sub(r'\^\{?2\}?', '^2', s)
    s = re.sub(r'\^\{([^{}]*)\}', r'^\1', s)
    s = re.sub(r'_\{([^{}]*)\}', r'_\1', s)
    s = re.sub(r'\\frac\{([^{}]*)\}\{([^{}]*)\}', r'(\1)/(\2)', s)
    for k, v in GREEK.items():
        s = s.replace(k, v)
    s = s.replace('{,}', ',').replace('\\,', ' ').replace('\\;', ' ')
    s = s.replace('\\!', '').replace('\\ ', ' ')
    s = re.sub(r'[{}]', '', s)
    return s


ACCENTS = {
    r'{\o}': '\u00f8', r'\o': '\u00f8',
    r"{\'i}": '\u00ed', r"{\'a}": '\u00e1', r"{\'e}": '\u00e9',
    r"{\'o}": '\u00f3', r"{\'u}": '\u00fa', r"{\'y}": '\u00fd',
    r'{\"o}': '\u00f6', r'{\"a}': '\u00e4', r'{\"u}': '\u00fc',
    r'{\i}': '\u0131', r'{\ss}': '\u00df',
}


def inline(s, md=True):
    """Convert an inline LaTeX fragment to Markdown or plain text."""
    s = re.sub(r'(?<!\\)%.*', '', s)          # comments
    # structural macros that carry no body text
    s = re.sub(r'\\label\{[^}]*\}', '', s)
    s = re.sub(r'\\author\{(?:[^{}]|\{[^{}]*\})*\}', '', s)
    s = re.sub(r'\\(?:maketitle|IEEEoverridecommandlockouts|centering|'
               r'toprule|midrule|bottomrule|small|footnotesize|noindent)\b',
               '', s)
    for k, v in ACCENTS.items():
        s = s.replace(k, v)
    # "\ " is an escaped inter-word space; in the source it is often
    # followed by a line break rather than a literal space.
    s = re.sub(r'\\(\s)', r'\1', s)
    s = s.replace('\\,', '\u2009').replace('\\%', '%')
    s = s.replace('\\&', '&').replace('\\#', '#')
    s = s.replace('\\_', '_').replace('\\$', '$')

    # citations and cross-references
    s = re.sub(r'~?\\cite\{([^}]*)\}', lambda m: ' [' + ', '.join(
        str(BIBNUM.get(x.strip(), x.strip())) for x in m.group(1).split(',')) + ']', s)
    s = re.sub(r'~?\\(?:eq)?ref\{([^}]*)\}', lambda m: ' ' + _refname(m.group(1)), s)

    # math
    s = re.sub(r'\$([^$]*)\$', lambda m: strip_math(m.group(1)), s)

    # font commands
    if md:
        s = re.sub(r'\\(?:hlt|textbf)\{([^{}]*)\}', r'**\1**', s)
        s = re.sub(r'\\emph\{([^{}]*)\}', r'*\1*', s)
        s = re.sub(r'\\textit\{([^{}]*)\}', r'*\1*', s)
        s = re.sub(r'\\texttt\{([^{}]*)\}', r'`\1`', s)
        s = re.sub(r'\\textsc\{([^{}]*)\}', r'\1', s)
    else:
        s = re.sub(r'\\(?:hlt|textbf|emph|textit|texttt|textsc)\{([^{}]*)\}',
                   r'\1', s)

    s = s.replace('---', '\u2014').replace('--', '\u2013')
    s = s.replace('~', ' ')
    s = s.replace("``", '\u201c').replace("''", '\u201d')
    s = re.sub(r'\\[a-zA-Z]+\b', '', s)       # leftover macros
    s = re.sub(r'[{}]', '', s)
    return re.sub(r'\s+', ' ', s).strip()


REFMAP = {}
BIBNUM = {}


def _refname(label):
    return REFMAP.get(label, label)


# ------------------------------------------------------------------
# Parse the .tex body into a block list
# ------------------------------------------------------------------
def parse(tex):
    body = tex.split(r'\begin{document}', 1)[1].split(r'\end{document}')[0]

    # bibitem keys -> reference numbers, needed before any \cite is rendered
    for n, key in enumerate(re.findall(r'\\bibitem\{([^}]*)\}', body), 1):
        BIBNUM[key] = n

    blocks = []
    sec_n, sub_n = 0, 0
    tab_n, fig_n, alg_n, eq_n = 0, 0, 0, 0

    # ---- first pass: assign numbers to every label ----
    for m in re.finditer(
            r'\\section\{|\\subsection\{|\\begin\{table\*?\}|'
            r'\\begin\{figure\*?\}|\\begin\{algorithm\}|'
            r'\\begin\{equation\}|\\begin\{align\}|\\label\{([^}]*)\}', body):
        tok = m.group(0)
        if tok.startswith(r'\section'):
            sec_n += 1
            sub_n = 0
            cur = ('sec', ROMAN[sec_n])
        elif tok.startswith(r'\subsection'):
            sub_n += 1
            cur = ('sub', f'{ROMAN[sec_n]}-{LETTER[sub_n]}')
        # Bare numbers: the prose already supplies "Table"/"Fig."/"Algorithm".
        elif tok.startswith(r'\begin{table'):
            tab_n += 1
            cur = ('tab', ROMAN[tab_n])
        elif tok.startswith(r'\begin{figure'):
            fig_n += 1
            cur = ('fig', str(fig_n))
        elif tok.startswith(r'\begin{algorithm'):
            alg_n += 1
            cur = ('alg', str(alg_n))
        elif tok.startswith(r'\begin{equation') or tok.startswith(r'\begin{align'):
            eq_n += 1
            cur = ('eq', f'({eq_n})')
        elif m.group(1):
            kind = m.group(1).split(':')[0]
            if kind in ('sec',):
                # the surrounding prose already supplies the word "Section"
                REFMAP[m.group(1)] = cur[1]
            elif kind == 'tab':
                REFMAP[m.group(1)] = cur[1]
            elif kind == 'fig':
                REFMAP[m.group(1)] = cur[1]
            elif kind == 'alg':
                REFMAP[m.group(1)] = cur[1]
            elif kind == 'eq':
                REFMAP[m.group(1)] = cur[1]

    # ---- second pass: emit blocks ----
    sec_n, sub_n, tab_n, fig_n, alg_n = 0, 0, 0, 0, 0
    i = 0
    while i < len(body):
        m = re.compile(
            r'\\title\{|\\begin\{abstract\}|\\begin\{IEEEkeywords\}|'
            r'\\section\*?\{|\\subsection\{|\\begin\{table\*?\}|'
            r'\\begin\{figure\*?\}|\\begin\{algorithm\}|'
            r'\\begin\{itemize\}|\\begin\{thebibliography\}|'
            r'\\begin\{(?:equation|align)\*?\}').search(body, i)
        if not m:
            blocks.append(('para', body[i:]))
            break
        if m.start() > i:
            blocks.append(('para', body[i:m.start()]))
        tok = m.group(0)

        if tok.startswith(r'\title'):
            txt, j = braced(body, m.end() - 1)
            blocks.append(('title', inline(txt)))
        elif tok.startswith(r'\begin{abstract}'):
            j = body.index(r'\end{abstract}', m.end())
            blocks.append(('abstract', body[m.end():j]))
            j += len(r'\end{abstract}')
        elif tok.startswith(r'\begin{IEEEkeywords}'):
            j = body.index(r'\end{IEEEkeywords}', m.end())
            blocks.append(('keywords', body[m.end():j]))
            j += len(r'\end{IEEEkeywords}')
        elif tok.startswith(r'\section'):
            txt, j = braced(body, m.end() - 1)
            if tok.endswith('*{'):
                blocks.append(('sec', ('', inline(txt))))
            else:
                sec_n += 1
                sub_n = 0
                blocks.append(('sec', (ROMAN[sec_n] + '.', inline(txt))))
        elif tok.startswith(r'\subsection'):
            txt, j = braced(body, m.end() - 1)
            sub_n += 1
            blocks.append(('sub', (LETTER[sub_n] + '.', inline(txt))))
        elif tok.startswith(r'\begin{table'):
            env = 'table*' if '*' in tok else 'table'
            j = body.index(r'\end{%s}' % env, m.end()) + len(r'\end{%s}' % env)
            tab_n += 1
            blocks.append(('table', (ROMAN[tab_n], body[m.end():j], '*' in tok)))
        elif tok.startswith(r'\begin{figure'):
            env = 'figure*' if '*' in tok else 'figure'
            j = body.index(r'\end{%s}' % env, m.end()) + len(r'\end{%s}' % env)
            fig_n += 1
            blocks.append(('figure', (fig_n, body[m.end():j], '*' in tok)))
        elif tok.startswith(r'\begin{algorithm}'):
            j = body.index(r'\end{algorithm}', m.end()) + len(r'\end{algorithm}')
            alg_n += 1
            blocks.append(('algorithm', (alg_n, body[m.end():j])))
        elif tok.startswith(r'\begin{itemize}'):
            j = body.index(r'\end{itemize}', m.end())
            blocks.append(('itemize', body[m.end():j]))
            j += len(r'\end{itemize}')
        elif tok.startswith(r'\begin{thebibliography}'):
            j = body.index(r'\end{thebibliography}', m.end())
            blocks.append(('bib', body[m.end():j]))
            j += len(r'\end{thebibliography}')
        else:  # equation / align
            env = re.match(r'\\begin\{([a-z]+\*?)\}', tok).group(1)
            j = body.index(r'\end{%s}' % env, m.end())
            blocks.append(('equation', body[m.end():j]))
            j += len(r'\end{%s}' % env)
        i = j
    return blocks


def braced(s, start):
    """Return (contents, index_after) for the brace group beginning at s[start]."""
    assert s[start] == '{'
    depth, i = 0, start
    while i < len(s):
        if s[i] == '{':
            depth += 1
        elif s[i] == '}':
            depth -= 1
            if depth == 0:
                return s[start + 1:i], i + 1
        i += 1
    raise ValueError('unbalanced braces')


def parse_tabular(src, md=True):
    """Extract (caption, rows) from a table environment body.

    md=False strips Markdown emphasis markers, which must not leak into
    DOCX cells where emphasis is carried by run formatting instead.
    """
    cap = ''
    mc = re.search(r'\\caption\{', src)
    if mc:
        cap, _ = braced(src, mc.end() - 1)
    mt = re.search(r'\\begin\{tabular\}\{((?:[^{}]|\{[^{}]*\})*)\}', src)
    if not mt:
        return inline(cap, md), []
    inner = src[mt.end():src.index(r'\end{tabular}', mt.end())]

    rows = []
    for raw in inner.split('\\\\'):
        raw = re.sub(r'\\(?:top|mid|bottom)rule', '', raw)
        raw = re.sub(r'\\cmidrule\(?[^)]*\)?\{[^}]*\}', '', raw)
        raw = re.sub(r'\\addlinespace(\[[^\]]*\])?', '', raw)
        if not raw.strip():
            continue
        cells = []
        for c in split_cells(raw):
            c = re.sub(r'\\multicolumn\{\d+\}\{[^{}]*\}\{(.*)\}', r'\1', c.strip(),
                       flags=re.S)
            c = re.sub(r'\\multirow\{[^{}]*\}\{[^{}]*\}\{(.*?)\}', r'\1', c, flags=re.S)
            c = c.replace(r'\footnotesize', '')
            cells.append(inline(c, md))
        if any(cells):
            rows.append(cells)
    return inline(cap, md), rows


def split_cells(row):
    """Split a tabular row on & that are not inside braces."""
    out, depth, cur = [], 0, ''
    for ch in row:
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
        if ch == '&' and depth == 0:
            out.append(cur)
            cur = ''
        else:
            cur += ch
    out.append(cur)
    return out


# ------------------------------------------------------------------
# Markdown writer
# ------------------------------------------------------------------
def to_markdown(blocks):
    L = []
    for kind, payload in blocks:
        if kind == 'title':
            L += [f'# {payload}', '', '**Anonymous authors**', '']
        elif kind == 'abstract':
            L += [f'***Abstract**\u2014{inline(payload)}*', '']
        elif kind == 'keywords':
            L += [f'***Index Terms**\u2014{inline(payload)}*', '']
        elif kind == 'sec':
            num, txt = payload
            L += ['', f'## {num} {txt}'.strip(), '']
        elif kind == 'sub':
            num, txt = payload
            L += ['', f'### {num} {txt}', '']
        elif kind == 'para':
            for p in re.split(r'\n\s*\n', payload):
                t = inline(p)
                if t:
                    L += [t, '']
        elif kind == 'itemize':
            for item in payload.split(r'\item')[1:]:
                t = inline(item)
                if t:
                    L.append(f'- {t}')
            L.append('')
        elif kind == 'equation':
            t = strip_math(re.sub(r'\\label\{[^}]*\}', '', payload))
            t = re.sub(r'\s+', ' ', t.replace('\\\\', ' ; ')).strip()
            L += ['> ' + t, '']
        elif kind == 'table':
            num, src, _wide = payload
            cap, rows = parse_tabular(src)
            L += [f'**TABLE {num}.** {cap}', '']
            if rows:
                ncol = max(len(r) for r in rows)
                head = rows[0] + [''] * (ncol - len(rows[0]))
                L.append('| ' + ' | '.join(head) + ' |')
                L.append('|' + '---|' * ncol)
                for r in rows[1:]:
                    r = r + [''] * (ncol - len(r))
                    L.append('| ' + ' | '.join(r) + ' |')
            L.append('')
        elif kind == 'figure':
            num, src, _wide = payload
            cap, _ = parse_tabular(src)
            mi = re.search(r'\\includegraphics\[[^\]]*\]\{([^}]*)\}', src)
            path = mi.group(1).replace('.pdf', '.png') if mi else ''
            L += [f'![Fig. {num}](../{path})', '',
                  f'**Fig. {num}.** {cap}', '']
        elif kind == 'algorithm':
            num, src = payload
            mc = re.search(r'\\caption\{', src)
            cap = inline(braced(src, mc.end() - 1)[0]) if mc else ''
            L += [f'**Algorithm {num}.** {cap}', '', '```']
            for ln in re.findall(r'\\(?:STATE|REQUIRE|ENSURE|REPEAT|UNTIL|RETURN)\b(.*)',
                                 src):
                L.append(inline(ln, md=False))
            L += ['```', '']
        elif kind == 'bib':
            L += ['', '## References', '']
            items = payload.split(r'\bibitem')[1:]
            for n, it in enumerate(items, 1):
                _, rest = braced(it, it.index('{'))
                L.append(f'[{n}] {inline(it[it.index("}") + 1:])}')
            L.append('')
    return '\n'.join(L)


# ------------------------------------------------------------------
# DOCX writer
# ------------------------------------------------------------------
def set_columns(section, n):
    cols = section._sectPr.xpath('./w:cols')[0]
    cols.set(qn('w:num'), str(n))
    cols.set(qn('w:space'), '240')


def shade(cell, hexcolor='D9E2F3'):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:fill'), hexcolor)
    tcPr.append(shd)


def to_docx(blocks, path):
    doc = Document()
    st = doc.styles['Normal']
    st.font.name = 'Times New Roman'
    st.font.size = Pt(10)
    st.paragraph_format.space_after = Pt(0)
    st.paragraph_format.first_line_indent = Inches(0.2)
    st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.27), Inches(11.69)   # A4
    for side in ('left_margin', 'right_margin'):
        setattr(sec, side, Inches(0.63))
    sec.top_margin, sec.bottom_margin = Inches(0.75), Inches(0.75)

    def para(text, size=10, bold=False, italic=False, align=None,
             indent=None, space_before=0, space_after=0):
        p = doc.add_paragraph()
        r = p.add_run(text)
        r.font.size = Pt(size)
        r.bold = bold
        r.italic = italic
        if align is not None:
            p.alignment = align
        p.paragraph_format.first_line_indent = Inches(indent if indent is not None else 0)
        p.paragraph_format.space_before = Pt(space_before)
        p.paragraph_format.space_after = Pt(space_after)
        return p

    single_now = True   # title block starts single-column

    def ensure_columns(n):
        nonlocal single_now
        want_single = (n == 1)
        if want_single == single_now:
            return
        s = doc.add_section(WD_SECTION.CONTINUOUS)
        set_columns(s, n)
        single_now = want_single

    set_columns(sec, 1)

    for kind, payload in blocks:
        if kind == 'title':
            para(payload, size=20, align=WD_ALIGN_PARAGRAPH.CENTER,
                 space_after=10)
            para('Anonymous authors', size=11,
                 align=WD_ALIGN_PARAGRAPH.CENTER, space_after=14)
        elif kind == 'abstract':
            ensure_columns(2)
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.first_line_indent = Inches(0.2)
            r = p.add_run('Abstract\u2014')
            r.bold = True
            r.italic = True
            r.font.size = Pt(9)
            r2 = p.add_run(inline(payload, md=False))
            r2.bold = True
            r2.font.size = Pt(9)
            p.paragraph_format.space_after = Pt(6)
        elif kind == 'keywords':
            p = doc.add_paragraph()
            p.paragraph_format.first_line_indent = Inches(0.2)
            r = p.add_run('Index Terms\u2014')
            r.bold = True
            r.italic = True
            r.font.size = Pt(9)
            r2 = p.add_run(inline(payload, md=False))
            r2.italic = True
            r2.font.size = Pt(9)
            p.paragraph_format.space_after = Pt(8)
        elif kind == 'sec':
            ensure_columns(2)
            num, txt = payload
            para(f'{num} {txt}'.strip().upper(), size=10,
                 align=WD_ALIGN_PARAGRAPH.CENTER, space_before=10,
                 space_after=4)
        elif kind == 'sub':
            num, txt = payload
            para(f'{num} {txt}', size=10, italic=True, space_before=6,
                 space_after=2)
        elif kind == 'para':
            for chunk in re.split(r'\n\s*\n', payload):
                t = inline(chunk, md=False)
                if t:
                    para(t, indent=0.2, space_after=3)
        elif kind == 'itemize':
            for item in payload.split(r'\item')[1:]:
                t = inline(item, md=False)
                if t:
                    p = doc.add_paragraph(t, style='List Bullet')
                    for r in p.runs:
                        r.font.size = Pt(10)
                        r.font.name = 'Times New Roman'
        elif kind == 'equation':
            t = strip_math(re.sub(r'\\label\{[^}]*\}', '', payload))
            t = re.sub(r'\s+', ' ', t.replace('\\\\', '   ')).strip()
            para(t, size=10, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER,
                 space_before=4, space_after=4)
        elif kind == 'table':
            num, src, wide = payload
            cap, rows = parse_tabular(src, md=False)
            ensure_columns(1 if wide else 2)
            para(f'TABLE {num}.', size=8, align=WD_ALIGN_PARAGRAPH.CENTER,
                 space_before=8)
            para(cap.upper() if len(cap) < 90 else cap, size=8,
                 align=WD_ALIGN_PARAGRAPH.CENTER, space_after=3)
            if rows:
                ncol = max(len(r) for r in rows)
                t = doc.add_table(rows=0, cols=ncol)
                t.style = 'Table Grid'
                t.alignment = WD_TABLE_ALIGNMENT.CENTER
                for ri, row in enumerate(rows):
                    cells = t.add_row().cells
                    for ci in range(ncol):
                        txt = row[ci] if ci < len(row) else ''
                        cp = cells[ci].paragraphs[0]
                        cp.paragraph_format.first_line_indent = Inches(0)
                        cp.alignment = (WD_ALIGN_PARAGRAPH.LEFT if ci == 0
                                        else WD_ALIGN_PARAGRAPH.CENTER)
                        rr = cp.add_run(txt)
                        rr.font.size = Pt(8)
                        rr.font.name = 'Times New Roman'
                        if ri == 0:
                            rr.bold = True
                            shade(cells[ci])
                para('', size=6, space_after=6)
            if wide:
                ensure_columns(2)
        elif kind == 'figure':
            num, src, wide = payload
            cap, _ = parse_tabular(src, md=False)
            mi = re.search(r'\\includegraphics\[[^\]]*\]\{([^}]*)\}', src)
            ensure_columns(1 if wide else 2)
            if mi:
                png = mi.group(1).replace('.pdf', '.png')
                if os.path.exists(png):
                    p = doc.add_paragraph()
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    p.paragraph_format.first_line_indent = Inches(0)
                    p.add_run().add_picture(
                        png, width=Inches(6.8 if wide else 3.2))
            para(f'Fig. {num}.  {cap}', size=8,
                 align=WD_ALIGN_PARAGRAPH.CENTER, space_after=8)
            if wide:
                ensure_columns(2)
        elif kind == 'algorithm':
            num, src = payload
            mc = re.search(r'\\caption\{', src)
            cap = inline(braced(src, mc.end() - 1)[0], md=False) if mc else ''
            para(f'Algorithm {num}.  {cap}', size=9, bold=True,
                 space_before=8, space_after=2)
            for n, ln in enumerate(re.findall(
                    r'\\(?:STATE|REQUIRE|ENSURE|REPEAT|UNTIL|RETURN)\b(.*)', src), 1):
                p = para(f'{n}:  {inline(ln, md=False)}', size=9)
                for r in p.runs:
                    r.font.name = 'Consolas'
            para('', size=6, space_after=6)
        elif kind == 'bib':
            items = payload.split(r'\bibitem')[1:]
            for n, it in enumerate(items, 1):
                txt = inline(it[it.index('}') + 1:], md=False)
                p = para(f'[{n}] {txt}', size=8, space_after=2)
                p.paragraph_format.left_indent = Inches(0.22)
                p.paragraph_format.first_line_indent = Inches(-0.22)

    doc.save(path)


def main():
    tex = open(TEX, encoding='utf-8').read()
    blocks = parse(tex)
    print(f"parsed {len(blocks)} blocks; "
          f"{sum(1 for k, _ in blocks if k == 'table')} tables, "
          f"{sum(1 for k, _ in blocks if k == 'figure')} figures")

    md = to_markdown(blocks)
    open(OUT_MD, 'w', encoding='utf-8').write(md)
    print(f"wrote {OUT_MD} ({len(md.splitlines())} lines)")

    to_docx(blocks, OUT_DOCX)
    print(f"wrote {OUT_DOCX} ({os.path.getsize(OUT_DOCX) // 1024} KB)")


if __name__ == '__main__':
    main()
