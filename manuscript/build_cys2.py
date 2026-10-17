#!/usr/bin/env python3
"""Render the revised GAALI paper into the CyS template, preserving its styles."""
import html
import os
import re
import shutil
import zipfile
from PIL import Image

SRC = 'revised.md'
TPL = 'tpl'
OUT = 'build'
EMU = 914400

# ---------------------------------------------------------------- references
REFS = [  # (key used in text, bold author/year part, remainder)
 ("Adhikari et al. 2024", "Adhikari, A., Manandhar, P., Khanal, R., Wagle, S., Acharya, P., Bal, B.K. (2024).",
  " Profanity and Offensiveness Detection in Nepali Language Using Bi-directional LSTM Models. Proceedings of the 21st International Conference on Natural Language Processing (ICON), pp. 515–521."),
 ("Bansal et al. 2022", "Bansal, V., Tyagi, M., Sharma, R., Gupta, V., Xin, Q. (2022).",
  " A Transformer Based Approach for Abuse Detection in Code Mixed Indic Languages. ACM Transactions on Asian and Low-Resource Language Information Processing. DOI: 10.1145/3571818."),
 ("Cohen 1960", "Cohen, J. (1960).",
  " A Coefficient of Agreement for Nominal Scales. Educational and Psychological Measurement, Vol. 20, No. 1, pp. 37–46."),
 ("Conneau et al. 2020", "Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L., Stoyanov, V. (2020).",
  " Unsupervised Cross-Lingual Representation Learning at Scale. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 8440–8451."),
 ("Davidson et al. 2017", "Davidson, T., Warmsley, D., Macy, M., Weber, I. (2017).",
  " Automated Hate Speech Detection and the Problem of Offensive Language. Proceedings of the International AAAI Conference on Web and Social Media, Vol. 11, No. 1, pp. 512–515."),
 ("Devlin et al. 2019", "Devlin, J., Chang, M.-W., Lee, K., Toutanova, K. (2019).",
  " BERT: Pre-Training of Deep Bidirectional Transformers for Language Understanding. Proceedings of NAACL-HLT 2019, pp. 4171–4186."),
 ("Jhaveri et al. 2022", "Jhaveri, M., Ramaiya, D., Chadha, H.S. (2022).",
  " Toxicity Detection for Indic Multilingual Social Media Content. arXiv:2201.00598."),
 ("Joachims 1998", "Joachims, T. (1998).",
  " Text Categorization with Support Vector Machines: Learning with Many Relevant Features. Machine Learning: ECML-98, Springer, pp. 137–142."),
 ("Kakwani et al. 2020", "Kakwani, D., Kunchukuttan, A., Golla, S., Gokul, N.C., Bhattacharyya, A., Khapra, M.M., et al. (2020).",
  " IndicNLPSuite: Monolingual Corpora, Evaluation Benchmarks and Pre-Trained Multilingual Language Models for Indian Languages. Findings of the ACL: EMNLP 2020, pp. 4948–4961."),
 ("Khanuja et al. 2021", "Khanuja, S., Bansal, D., Mehtani, S., Khosla, S., Dey, A., Gopalan, B., et al. (2021).",
  " MuRIL: Multilingual Representations for Indian Languages. arXiv:2103.10730."),
 ("Kumar et al. 2018", "Kumar, R., Ojha, A.K., Malmasi, S., Zampieri, M. (2018).",
  " Benchmarking Aggression Identification in Social Media. Proceedings of the First Workshop on Trolling, Aggression and Cyberbullying (TRAC-2018), pp. 1–11."),
 ("Manning et al. 2008", "Manning, C.D., Raghavan, P., Schütze, H. (2008).",
  " Introduction to Information Retrieval. Cambridge University Press."),
 ("Mathur et al. 2018", "Mathur, P., Shah, R., Sawhney, R., Mahata, D. (2018).",
  " Detecting Offensive Tweets in Hindi-English Code-Switched Language. Proceedings of the Sixth International Workshop on Natural Language Processing for Social Media, pp. 18–26."),
 ("McCallum and Nigam 1998", "McCallum, A., Nigam, K. (1998).",
  " A Comparison of Event Models for Naive Bayes Text Classification. AAAI-98 Workshop on Learning for Text Categorization, pp. 41–48."),
 ("Monroe et al. 2008", "Monroe, B.L., Colaresi, M.P., Quinn, K.M. (2008).",
  " Fightin’ Words: Lexical Feature Selection and Evaluation for Identifying the Content of Political Conflict. Political Analysis, Vol. 16, No. 4, pp. 372–403."),
 ("Niraula et al. 2021", "Niraula, N.B., Dulal, S., Koirala, D. (2021).",
  " Offensive Language Detection in Nepali Social Media. Proceedings of the 5th Workshop on Online Abuse and Harms (WOAH 2021), pp. 67–75."),
 ("Poletto et al. 2021", "Poletto, F., Basile, V., Sanguinetti, M., Bosco, C., Patti, V. (2021).",
  " Resources and Benchmark Corpora for Hate Speech Detection: A Systematic Review. Language Resources and Evaluation, Vol. 55, No. 2, pp. 477–523."),
 ("Singh et al. 2020", "Singh, O.M., Timilsina, S., Bal, B.K., Joshi, A. (2020).",
  " Aspect Based Abusive Sentiment Detection in Nepali Social Media Texts. 2020 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM), pp. 301–308."),
 ("Sokolova and Lapalme 2009", "Sokolova, M., Lapalme, G. (2009).",
  " A Systematic Analysis of Performance Measures for Classification Tasks. Information Processing & Management, Vol. 45, No. 4, pp. 427–437."),
 ("Timilsina et al. 2020", "Timilsina, S., Singh, O.M., Joshi, A. (2020).",
  " Detecting Offensive Social Media Text in Nepali Language. Poster presented at the 2020 International Conference on Social Computing and Social Media."),
 ("Vidgen et al. 2019", "Vidgen, B., Harris, A., Nguyen, D., Tromble, R., Hale, S., Margetts, H. (2019).",
  " Challenges and Frontiers in Abusive Content Detection. Proceedings of the Third Workshop on Abusive Language Online, pp. 80–93."),
 ("Zampieri et al. 2019a", "Zampieri, M., Malmasi, S., Nakov, P., Rosenthal, S., Farra, N., Kumar, R. (2019).",
  " Predicting the Type and Target of Offensive Posts in Social Media. Proceedings of NAACL-HLT 2019, Vol. 1, pp. 1415–1420."),
 ("Zampieri et al. 2019b", "Zampieri, M., Malmasi, S., Nakov, P., Rosenthal, S., Farra, N., Kumar, R. (2019).",
  " SemEval-2019 Task 6: Identifying and Categorizing Offensive Language in Social Media (OffensEval). Proceedings of the 13th International Workshop on Semantic Evaluation, pp. 75–86."),
]

AUTHORS = [("Bishnu Shah", "1,\u2020"), ("Tanusree Nath", "1,2,\u2020"),
           ("Sukriti Singh", "1"), ("Manjari Gupta", "1")]
AFFIL = [
 "\u00b9 Banaras Hindu University, Department of Computer Science, Varanasi, India",
 "\u00b2 Norwegian University of Science and Technology, Department of Computer Science, Gj\u00f8vik, Norway",
]
EMAILS = "[email]@bhu.ac.in, tanusreenath32@gmail.com, [email]@bhu.ac.in, [email]@bhu.ac.in"
FOOTMARK = ("\u2020 Bishnu Shah and Tanusree Nath contributed equally to this work "
            "and are joint first authors.")

# ---------------------------------------------------------------- helpers
def esc(t):
    return html.escape(t, quote=False)

def runs(text, base=''):
    """Convert a markdown inline string into a list of <w:r> with bold/italic/code."""
    out = []
    tok = re.split(r'(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)', text)
    for t in tok:
        if not t:
            continue
        if t.startswith('**') and t.endswith('**'):
            rpr, body = '<w:b/>', t[2:-2]
        elif t.startswith('*') and t.endswith('*'):
            rpr, body = '<w:i/>', t[1:-1]
        elif t.startswith('`') and t.endswith('`'):
            rpr, body = '<w:rFonts w:ascii="Courier New" w:hAnsi="Courier New"/><w:sz w:val="18"/>', t[1:-1]
        else:
            rpr, body = '', t
        out.append(f'<w:r><w:rPr>{base}{rpr}<w:lang w:val="en-US"/></w:rPr>'
                   f'<w:t xml:space="preserve">{esc(body)}</w:t></w:r>')
    return ''.join(out)

def para(text, style=None, jc=None, first_line=None, sz=None, before=None, after=None, keep=False):
    ppr = []
    if style: ppr.append(f'<w:pStyle w:val="{style}"/>')
    if keep: ppr.append('<w:keepNext/>')
    sp = ''
    if before is not None: sp += f' w:before="{before}"'
    if after is not None: sp += f' w:after="{after}"'
    if sp: ppr.append(f'<w:spacing{sp}/>')
    if first_line is not None: ppr.append(f'<w:ind w:firstLine="{first_line}"/>')
    if jc: ppr.append(f'<w:jc w:val="{jc}"/>')
    base = f'<w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/>' if sz else ''
    if base: ppr.append(f'<w:rPr>{base}<w:lang w:val="en-US"/></w:rPr>')
    return f'<w:p><w:pPr>{"".join(ppr)}</w:pPr>{runs(text, base)}</w:p>'

def img_para(path, width_in, jc='center', after=120, trailer='', eqnum=False):
    w, h = Image.open(path).size
    cx, cy = int(width_in * EMU), int(width_in * h / w * EMU)
    rid = REL[path]
    drawing = (f'<w:r><w:rPr><w:noProof/></w:rPr><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
      f'<wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{DOCPR[0]}" name="Picture {DOCPR[0]}"/>'
      f'<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:graphicData '
      f'uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic '
      f'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:nvPicPr>'
      f'<pic:cNvPr id="{DOCPR[0]}" name="Picture {DOCPR[0]}"/><pic:cNvPicPr/></pic:nvPicPr>'
      f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
      f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
      f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic>'
      f'</wp:inline></w:drawing></w:r>')
    DOCPR[0] += 1
    if eqnum:
        return (f'<w:tbl><w:tblPr><w:tblW w:w="4532" w:type="dxa"/>'
                f'<w:tblBorders><w:top w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
                f'<w:left w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
                f'<w:bottom w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
                f'<w:right w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
                f'<w:insideH w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
                f'<w:insideV w:val="none" w:sz="0" w:space="0" w:color="auto"/></w:tblBorders>'
                f'<w:tblCellMar><w:top w:w="0" w:type="dxa"/><w:left w:w="0" w:type="dxa"/>'
                f'<w:bottom w:w="0" w:type="dxa"/><w:right w:w="0" w:type="dxa"/></w:tblCellMar>'
                f'<w:tblLook w:val="0000"/></w:tblPr>'
                f'<w:tblGrid><w:gridCol w:w="3882"/><w:gridCol w:w="650"/></w:tblGrid>'
                f'<w:tr><w:tc><w:tcPr><w:tcW w:w="3882" w:type="dxa"/>'
                f'<w:vAlign w:val="center"/></w:tcPr>'
                f'<w:p><w:pPr><w:spacing w:before="60" w:after="{after}"/>'
                f'<w:ind w:firstLine="0"/><w:jc w:val="center"/></w:pPr>{drawing}</w:p></w:tc>'
                f'<w:tc><w:tcPr><w:tcW w:w="650" w:type="dxa"/><w:vAlign w:val="center"/></w:tcPr>'
                f'<w:p><w:pPr><w:spacing w:before="60" w:after="{after}"/>'
                f'<w:ind w:firstLine="0"/><w:jc w:val="right"/></w:pPr>{trailer}</w:p></w:tc>'
                f'</w:tr></w:tbl>')
    return (f'<w:p><w:pPr><w:spacing w:before="60" w:after="{after}"/><w:ind w:firstLine="0"/>'
            f'<w:jc w:val="center"/></w:pPr>{drawing}</w:p>')


def table(rows, widths):
    total = sum(widths)
    grid = ''.join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    body = ''
    for ri, row in enumerate(rows):
        cells = ''
        for ci, cell in enumerate(row):
            b = '<w:b/>' if ri == 0 else ''
            base = f'<w:sz w:val="18"/><w:szCs w:val="18"/>{b}'
            jc = 'left' if ci == 0 and ri > 0 else 'center'
            cells += (f'<w:tc><w:tcPr><w:tcW w:w="{widths[ci]}" w:type="dxa"/>'
                      f'<w:shd w:val="clear" w:color="auto" w:fill="FFFFFF"/><w:vAlign w:val="center"/></w:tcPr>'
                      f'<w:p><w:pPr><w:ind w:left="-28" w:right="-28" w:firstLine="0"/>'
                      f'<w:jc w:val="{jc}"/><w:rPr>{base}</w:rPr></w:pPr>{runs(cell, base)}</w:p></w:tc>')
        hdr = '<w:trPr><w:tblHeader/></w:trPr>' if ri == 0 else ''
        body += f'<w:tr>{hdr}{cells}</w:tr>'
    return (f'<w:tbl><w:tblPr><w:tblW w:w="{total}" w:type="dxa"/><w:jc w:val="center"/>'
            f'<w:tblBorders><w:top w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
            f'<w:bottom w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
            f'<w:insideH w:val="single" w:sz="6" w:space="0" w:color="auto"/></w:tblBorders>'
            f'<w:shd w:val="clear" w:color="auto" w:fill="FFFFFF"/>'
            f'<w:tblLook w:val="04A0" w:firstRow="1" w:lastRow="0" w:firstColumn="1" '
            f'w:lastColumn="0" w:noHBand="0" w:noVBand="1"/></w:tblPr>'
            f'<w:tblGrid>{grid}</w:tblGrid>{body}</w:tbl>')

# ---------------------------------------------------------------- citations
def build_citation_map(body_text):
    pat = re.compile(r'([A-Z][A-Za-zÀ-ſ\-]+)((?: and [A-Z][A-Za-z\-]+)| et al\.)?,?\s*\(?(\d{4}[ab]?)\)?')
    keys = {r[0] for r in REFS}
    order = []
    for m in pat.finditer(body_text):
        sur, suf, yr = m.group(1), (m.group(2) or '').strip(), m.group(3)
        k = f"{sur}{(' '+suf) if suf else ''} {yr}"
        if k in keys and k not in order:
            order.append(k)
    for r in REFS:                       # anything never cited keeps a slot at the end
        if r[0] not in order: order.append(r[0])
    return {k: i + 1 for i, k in enumerate(order)}

def numericise(text, cmap):
    # (Author, Year; Author, Year)  ->  [n, m]
    def group_sub(m):
        inner = m.group(1)
        parts = re.split(r';\s*', inner)
        nums = []
        for p in parts:
            mm = re.match(r'([A-Z][A-Za-zÀ-ſ\-]+)((?: and [A-Z][A-Za-z\-]+)| et al\.)?,?\s*(\d{4}[ab]?)$', p.strip())
            if not mm: return m.group(0)
            k = f"{mm.group(1)}{(' '+mm.group(2).strip()) if mm.group(2) else ''} {mm.group(3)}"
            if k not in cmap: return m.group(0)
            nums.append(cmap[k])
        return '[' + ', '.join(str(n) for n in sorted(nums)) + ']'

    # "(`model-id`; Author et al., Year)" -> "(`model-id`, [n])"
    def tail_sub(m):
        head, sur, suf, yr = m.group(1), m.group(2), (m.group(3) or '').strip(), m.group(4)
        k = f"{sur}{(' '+suf) if suf else ''} {yr}"
        return f"({head}, [{cmap[k]}])" if k in cmap else m.group(0)
    text = re.sub(r'\((`[^`]+`);\s*([A-Z][A-Za-z\u00c0-\u017f\-]+)((?: and [A-Z][A-Za-z\-]+)| et al\.)?,?\s*(\d{4}[ab]?)\)',
                  tail_sub, text)
    text = re.sub(r'\(([^()]*?\d{4}[ab]?)\)', group_sub, text)
    # Author et al. (Year) -> Author et al. [n]
    def inline_sub(m):
        k = f"{m.group(1)}{(' '+m.group(2).strip()) if m.group(2) else ''} {m.group(3)}"
        return f"{m.group(1)}{m.group(2) or ''} [{cmap[k]}]" if k in cmap else m.group(0)
    text = re.sub(r'([A-Z][A-Za-zÀ-ſ\-]+)((?: and [A-Z][A-Za-z\-]+)| et al\.)?\s*\[(\d{4}[ab]?)\]', m0 := (lambda m: m.group(0)), text)
    text = re.sub(r'([A-Z][A-Za-zÀ-ſ\-]+)((?: and [A-Z][A-Za-z\-]+)| et al\.)?\s*\((\d{4}[ab]?)\)', inline_sub, text)
    return text

# ---------------------------------------------------------------- parse source
src = open(SRC).read()
body_md, refs_md = src.split('**References**')
CMAP = build_citation_map(body_md)

DOCPR = [100]
REL = {}
SPAN = []            # parallel list: 'full' or 'col' for each block in P

def emit(xml, span='col'):
    P.append(xml); SPAN.append(span)

def collect_images(md):
    return re.findall(r'!\[\]\(([^)]+)\)', md)

imgs = collect_images(body_md)
for i, p in enumerate(dict.fromkeys(imgs)):
    REL[p] = f'rIdImg{i}'

# ---------------------------------------------------------------- emit body
P = []
emit(para('GAALI: A Pilot Benchmark for Offensive Language Detection in Romanized Code-Mixed Nepali', style='Ttulo'), 'full')

auth = ''
for i, (nm, sup) in enumerate(AUTHORS):
    if i: auth += '<w:r><w:rPr><w:lang w:val="en-US"/></w:rPr><w:t xml:space="preserve">, </w:t></w:r>'
    auth += (f'<w:r><w:rPr><w:lang w:val="en-US"/></w:rPr><w:t xml:space="preserve">{esc(nm)}</w:t></w:r>'
             f'<w:r><w:rPr><w:vertAlign w:val="superscript"/><w:lang w:val="en-US"/></w:rPr>'
             f'<w:t>{esc(sup)}</w:t></w:r>')
emit(f'<w:p><w:pPr><w:spacing w:before="480" w:after="240"/><w:ind w:firstLine="0"/>'
         f'<w:jc w:val="center"/></w:pPr>{auth}</w:p>', 'full')
for a in AFFIL:
    emit(para(a, jc='center', first_line=0, sz=18, after=0), 'full')
emit(para(EMAILS, jc='center', first_line=0, sz=18, before=60, after=240), 'full')
emit(para(FOOTMARK, jc='center', first_line=0, sz=16, after=240), 'full')

AFTER_TABLE = [False]
PENDING_CAPTION = [False]

def flush_para(text, first=False):
    t = numericise(text, CMAP)
    before = 160 if AFTER_TABLE[0] else None
    AFTER_TABLE[0] = False
    emit(para(t, first_line=(0 if first else 284), jc='both', before=before))

lines = body_md.split('\n')
i = 0
in_abstract = False
prev_heading = True
TABLE_BUF = []
while i < len(lines):
    ln = lines[i]
    st = ln.strip()

    if not st:
        i += 1; continue

    # title line already emitted
    if st.startswith('**GAALI: A Pilot Benchmark'):
        i += 1; continue

    # abstract
    if st == '**Abstract**':
        i += 1
        while i < len(lines) and not lines[i].strip(): i += 1
        ab = numericise(lines[i].strip(), CMAP)
        emit(f'<w:p><w:pPr><w:spacing w:after="120"/><w:ind w:firstLine="0"/><w:jc w:val="both"/>'
                 f'<w:rPr><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr></w:pPr>'
                 f'<w:r><w:rPr><w:b/><w:sz w:val="18"/><w:szCs w:val="18"/><w:lang w:val="en-US"/></w:rPr>'
                 f'<w:t xml:space="preserve">Abstract. </w:t></w:r>'
                 f'{runs(ab, "<w:sz w:val=\'18\'/><w:szCs w:val=\'18\'/>".replace(chr(39), chr(34)))}</w:p>', 'full')
        i += 1; continue

    if st.startswith('*[A Spanish version'):
        i += 1; continue

    if st.startswith('**Keywords'):
        kw = st.replace('**Keywords:', '').replace('**', '').strip()
        emit(f'<w:p><w:pPr><w:spacing w:after="240"/><w:ind w:firstLine="0"/><w:jc w:val="both"/>'
                 f'<w:rPr><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr></w:pPr>'
                 f'<w:r><w:rPr><w:b/><w:sz w:val="18"/><w:szCs w:val="18"/><w:lang w:val="en-US"/></w:rPr>'
                 f'<w:t xml:space="preserve">Keywords. </w:t></w:r>'
                 f'{runs(kw, "<w:sz w:val=\'18\'/><w:szCs w:val=\'18\'/>".replace(chr(39), chr(34)))}</w:p>', 'full')
        i += 1; continue

    # headings  **N. Title**  /  **N.N. Title**  /  **N.N.N Title**
    m = re.match(r'^\*\*(\d)(?:\.(\d))?(?:\.(\d))?\.?\s+(.+?)\*\*$', st)
    if m:
        lvl = 1 + (1 if m.group(2) else 0) + (1 if m.group(3) else 0)
        style = {1: 'Ttulo1', 2: 'Ttulo2', 3: 'Ttulo3'}[lvl]
        sp = {1: 'w:before="240" w:after="160"', 2: 'w:before="180" w:after="140"',
              3: 'w:before="120" w:after="100"'}[lvl]
        emit(f'<w:p><w:pPr><w:pStyle w:val="{style}"/><w:spacing {sp}/></w:pPr>'
                 f'<w:r><w:rPr><w:lang w:val="en-US"/></w:rPr><w:t>{esc(m.group(4))}</w:t></w:r></w:p>')
        prev_heading = True
        i += 1; continue

    # unnumbered headings
    if re.match(r'^\*\*(Data and Code Availability)\*\*$', st):
        emit(f'<w:p><w:pPr><w:pStyle w:val="Ttulo1"/><w:numPr><w:ilvl w:val="0"/>'
                 f'<w:numId w:val="0"/></w:numPr><w:spacing w:before="240" w:after="160"/></w:pPr>'
                 f'<w:r><w:rPr><w:lang w:val="en-US"/></w:rPr><w:t>Data and Code Availability</w:t></w:r></w:p>')
        prev_heading = True
        i += 1; continue

    # table / figure captions
    mcap = re.match(r'^\*\*(Table|Figure) (\d+)\.\*\*\s*(.*)$', st)
    if mcap:
        PENDING_CAPTION[0] = (mcap.group(1) == 'Table')
        label = 'Table' if mcap.group(1) == 'Table' else 'Fig.'
        cap = numericise(mcap.group(3), CMAP)
        base = '<w:sz w:val="18"/><w:szCs w:val="18"/>'
        cap_before = 240 if AFTER_TABLE[0] else 200
        AFTER_TABLE[0] = False
        emit(f'<w:p><w:pPr><w:keepNext/><w:spacing w:before="{cap_before}" w:after="80"/>'
                 f'<w:ind w:firstLine="0"/><w:jc w:val="center"/>'
                 f'<w:rPr>{base}</w:rPr></w:pPr>'
                 f'<w:r><w:rPr><w:b/>{base}<w:lang w:val="en-US"/></w:rPr>'
                 f'<w:t xml:space="preserve">{label} {mcap.group(2)}. </w:t></w:r>'
                 f'{runs(cap, base)}</w:p>')
        prev_heading = True
        i += 1; continue

    # images (figures and equations)
    mimg = re.match(r'^!\[\]\(([^)]+)\)\{width="([\d.]+)in"[^}]*\}\s*(\*\*\(\d+\)\*\*)?', st)
    if mimg:
        path, w = mimg.group(1), float(mimg.group(2))
        num = mimg.group(3)
        trailer = ''
        if num:
            n = num.strip('*()')
            trailer = (f'<w:r><w:rPr><w:lang w:val="en-US"/></w:rPr>'
                       f'<w:t xml:space="preserve">({n})</w:t></w:r>')
        emit(img_para(path, min(w, 2.6), after=(60 if num else 50),
                          trailer=trailer, eqnum=bool(num)))
        prev_heading = True
        i += 1; continue

    # markdown tables
    if st.startswith('|'):
        rows = []
        while i < len(lines) and lines[i].strip().startswith('|'):
            r = lines[i].strip()
            if not re.match(r'^\|[\s\-:|]+\|$', r):
                rows.append([c.strip() for c in r.strip('|').split('|')])
            i += 1
        ncol = max(len(r) for r in rows)
        rows = [r + [''] * (ncol - len(r)) for r in rows]
        wide_text = any(len(c) > 22 for r in rows for c in r)
        span = 'full' if (ncol >= 7 or (ncol >= 5 and wide_text)) else 'col'
        avail = 9404 if span == 'full' else 4532
        frac = 0.30 if ncol <= 4 else (0.26 if ncol <= 6 else 0.17)
        first = int(avail * frac)
        rest = (avail - first) // (ncol - 1) if ncol > 1 else avail
        widths = [first] + [rest] * (ncol - 1)
        widths[-1] += avail - sum(widths)
        rows = [[numericise(c, CMAP) for c in r] for r in rows]
        # the caption emitted just before must share the table's span
        if SPAN and PENDING_CAPTION[0]:
            SPAN[-1] = span
        PENDING_CAPTION[0] = False
        emit(table(rows, widths), span)
        AFTER_TABLE[0] = True
        prev_heading = True
        continue

    # bullets
    if st.startswith('• ') or st.startswith('- '):
        txt = numericise(st[2:], CMAP)
        emit(f'<w:p><w:pPr><w:pStyle w:val="Prrafodelista"/><w:numPr><w:ilvl w:val="0"/>'
                 f'<w:numId w:val="19"/></w:numPr><w:spacing w:before="20" w:after="20"/>'
                 f'<w:contextualSpacing w:val="0"/><w:jc w:val="both"/></w:pPr>{runs(txt)}</w:p>')
        prev_heading = False
        i += 1; continue

    # body paragraph (may be bold-led, e.g. limitations entries)
    flush_para(st, first=prev_heading)
    prev_heading = False
    i += 1

# references
emit('<w:p><w:pPr><w:pStyle w:val="Ttulo1"/><w:numPr><w:ilvl w:val="0"/>'
         '<w:numId w:val="0"/></w:numPr><w:spacing w:before="240" w:after="160"/></w:pPr>'
         '<w:r><w:rPr><w:lang w:val="en-US"/></w:rPr><w:t>References</w:t></w:r></w:p>')
ordered = sorted(REFS, key=lambda r: CMAP[r[0]])
REF_NUMID = 27
for key, bold, rest in ordered:
    base = '<w:szCs w:val="20"/>'
    emit(f'<w:p><w:pPr><w:pStyle w:val="Prrafodelista"/><w:numPr><w:ilvl w:val="0"/>'
             f'<w:numId w:val="{REF_NUMID}"/></w:numPr><w:spacing w:before="0" w:after="0"/>'
             f'<w:contextualSpacing w:val="0"/><w:jc w:val="both"/>'
             f'<w:rPr>{base}</w:rPr></w:pPr>'
             f'<w:r><w:rPr><w:b/><w:bCs/>{base}<w:lang w:val="en-US"/></w:rPr>'
             f'<w:t xml:space="preserve">{esc(bold)}</w:t></w:r>'
             f'<w:r><w:rPr>{base}<w:lang w:val="en-US"/></w:rPr>'
             f'<w:t xml:space="preserve">{esc(rest)}</w:t></w:r></w:p>')

# ---------------------------------------------------------------- package
if os.path.exists(OUT): shutil.rmtree(OUT)
shutil.copytree(TPL, OUT)
doc = open(f'{OUT}/word/document.xml').read()
sect = re.search(r'<w:sectPr.*?</w:sectPr>', doc, re.S).group(0)

PGSZ = '<w:pgSz w:w="12240" w:h="15840" w:code="1"/>'
PGMAR = ('<w:pgMar w:top="1985" w:right="1418" w:bottom="2268" w:left="1418" '
         'w:header="0" w:footer="1418" w:gutter="0"/>')
def sectpr(ncols, final=False):
    cols = ('<w:cols w:space="340"/>' if ncols == 1
            else '<w:cols w:num="2" w:space="340" w:equalWidth="1"/>')
    typ = '' if final else '<w:type w:val="continuous"/>'
    return f'<w:sectPr>{typ}{PGSZ}{PGMAR}{cols}<w:docGrid w:linePitch="360"/></w:sectPr>'

def with_sectpr(pxml, spr):
    """Append a sectPr as the LAST child of the paragraph's pPr (schema order)."""
    if pxml.startswith('<w:p><w:pPr>'):
        return pxml.replace('</w:pPr>', spr + '</w:pPr>', 1)
    return pxml.replace('<w:p>', '<w:p><w:pPr>' + spr + '</w:pPr>', 1)

SPACER = ('<w:p><w:pPr><w:spacing w:before="0" w:after="0" w:line="20" w:lineRule="exact"/>'
          '<w:ind w:firstLine="0"/><w:rPr><w:sz w:val="2"/></w:rPr>{spr}</w:pPr></w:p>')

out, k = [], 0
while k < len(P):
    span = SPAN[k]
    j = k
    while j < len(P) and SPAN[j] == span:
        j += 1
    run = P[k:j]                      # consecutive blocks sharing a span
    ncols = 1 if span == 'full' else 2
    last_section = (j >= len(P))
    if last_section:
        out.extend(run)               # final section uses the body-level sectPr
    else:
        spr = sectpr(ncols)
        if run[-1].startswith('<w:p>'):
            run[-1] = with_sectpr(run[-1], spr)
            out.extend(run)
        else:                         # ends on a table: needs a carrier paragraph
            out.extend(run)
            out.append(SPACER.format(spr=spr))
    k = j

body_sect = sectpr(2 if SPAN[-1] == 'col' else 1, final=True)
head = doc[:doc.index('<w:body>') + len('<w:body>')]
doc_new = head + ''.join(out) + body_sect + '</w:body></w:document>'
open(f'{OUT}/word/document.xml', 'w').write(doc_new)

# restarting numbering instance for the reference list
num = open(f'{OUT}/word/numbering.xml').read()
if 'w:numId="27"' not in num:
    num = num.replace('</w:numbering>',
        '<w:num w:numId="27"><w:abstractNumId w:val="1"/>'
        '<w:lvlOverride w:ilvl="0"><w:startOverride w:val="1"/></w:lvlOverride>'
        '</w:num></w:numbering>')
    open(f'{OUT}/word/numbering.xml', 'w').write(num)

# media + rels
os.makedirs(f'{OUT}/word/media', exist_ok=True)
rels = open(f'{OUT}/word/_rels/document.xml.rels').read()
add = ''
for path, rid in REL.items():
    name = rid + '_' + os.path.basename(path)
    shutil.copy(path, f'{OUT}/word/media/{name}')
    add += (f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
            f'relationships/image" Target="media/{name}"/>')
rels = rels.replace('</Relationships>', add + '</Relationships>')
open(f'{OUT}/word/_rels/document.xml.rels', 'w').write(rels)

ct = open(f'{OUT}/[Content_Types].xml').read()
if 'Extension="png"' not in ct:
    ct = ct.replace('<Types ', '<Types ', 1).replace(
        '</Types>', '<Default Extension="png" ContentType="image/png"/></Types>')
    open(f'{OUT}/[Content_Types].xml', 'w').write(ct)

outname = 'GAALI_CyS.docx'
if os.path.exists(outname): os.remove(outname)
zf = zipfile.ZipFile(outname, 'w', zipfile.ZIP_DEFLATED)
for root, _, files in os.walk(OUT):
    for f in files:
        full = os.path.join(root, f)
        zf.write(full, os.path.relpath(full, OUT))
zf.close()
print("citation order:", {k: v for k, v in sorted(CMAP.items(), key=lambda x: x[1])})
print("paragraphs:", len(P), "| images:", len(REL))
print("wrote", outname)
