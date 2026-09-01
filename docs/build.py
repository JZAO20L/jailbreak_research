#!/usr/bin/env python3
"""
哈尔滨工业大学硕士学位论文中期报告 - 构建脚本 (HTML版)

用法: python docs/build.py
从 docs/midterm_report.html 生成格式符合要求的 docs/midterm_report.docx

依赖: pip install pypandoc pypandoc_binary python-docx
"""

import sys
from pathlib import Path

try:
    import pypandoc
except ImportError:
    sys.exit("缺少 pypandoc，请运行: pip install pypandoc pypandoc_binary")

try:
    from docx import Document
    from docx.shared import Pt, Cm
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
    from docx.enum.style import WD_STYLE_TYPE
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
except ImportError:
    sys.exit("缺少 python-docx，请运行: pip install python-docx")

DIR = Path(__file__).parent
HTML_FILE = DIR / "midterm_report.html"
REF_DOCX = DIR / "reference.docx"
OUT_DOCX = DIR / "midterm_report.docx"


# ============================================================
#  Helpers
# ============================================================

def _set_style_font(style, east_asia, ascii_font="Times New Roman"):
    """Set East Asian + ASCII fonts on a Word style."""
    rPr = style.element.find(qn("w:rPr"))
    if rPr is None:
        rPr = OxmlElement("w:rPr")
        style.element.append(rPr)
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.insert(0, rFonts)
    rFonts.set(qn("w:eastAsia"), east_asia)
    rFonts.set(qn("w:ascii"), ascii_font)
    rFonts.set(qn("w:hAnsi"), ascii_font)


def _make_run(text, east_asia="黑体", size_pt=14, bold=True, underline=False):
    """Create a ``w:r`` XML element."""
    r = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")

    rFonts = OxmlElement("w:rFonts")
    rFonts.set(qn("w:eastAsia"), east_asia)
    rFonts.set(qn("w:ascii"), "Times New Roman")
    rFonts.set(qn("w:hAnsi"), "Times New Roman")
    rPr.append(rFonts)

    for tag in ("w:sz", "w:szCs"):
        el = OxmlElement(tag)
        el.set(qn("w:val"), str(int(size_pt * 2)))
        rPr.append(el)

    if bold:
        rPr.append(OxmlElement("w:b"))
    if underline:
        u = OxmlElement("w:u")
        u.set(qn("w:val"), "single")
        rPr.append(u)

    r.append(rPr)
    t = OxmlElement("w:t")
    t.set(qn("xml:space"), "preserve")
    t.text = text
    r.append(t)
    return r


def _make_para(runs, align=WD_ALIGN_PARAGRAPH.CENTER,
               space_before=0, space_after=0, left_indent_cm=0):
    """Create a ``w:p`` XML element."""
    p = OxmlElement("w:p")
    pPr = OxmlElement("w:pPr")

    if align == WD_ALIGN_PARAGRAPH.CENTER:
        jc = OxmlElement("w:jc")
        jc.set(qn("w:val"), "center")
        pPr.append(jc)

    spacing = OxmlElement("w:spacing")
    spacing.set(qn("w:before"), str(int(space_before * 20)))
    spacing.set(qn("w:after"), str(int(space_after * 20)))
    pPr.append(spacing)

    if left_indent_cm > 0:
        ind = OxmlElement("w:ind")
        ind.set(qn("w:left"), str(int(left_indent_cm * 567)))
        pPr.append(ind)

    p.append(pPr)

    if isinstance(runs, list):
        for r in runs:
            p.append(r)
    else:
        p.append(runs)
    return p


def _extract_cover_info():
    """Extract cover page info from HTML file."""
    import re
    html = HTML_FILE.read_text(encoding="utf-8")

    def extract(pattern, default=""):
        m = re.search(pattern, html)
        return m.group(1).strip() if m else default

    return {
        "title": extract(r'<span class="value">(.*?)</span>'),
        "department": extract(r'<td class="label">院\s*（系）</td>\s*<td class="value">(.*?)</td>'),
        "discipline": extract(r'<td class="label">学\s*科</td>\s*<td class="value">(.*?)</td>'),
        "advisor": extract(r'<td class="label">导\s*师</td>\s*<td class="value">(.*?)</td>'),
        "student": extract(r'<td class="label">研 究 生</td>\s*<td class="value">(.*?)</td>'),
        "student_id": extract(r'<td class="label">学\s*号</td>\s*<td class="value">(.*?)</td>'),
        "report_date": extract(r'<td class="label">中期报告日期</td>\s*<td class="value">(.*?)</td>'),
    }


# ============================================================
#  Step 1: Generate reference.docx
# ============================================================

def create_reference():
    """Generate reference.docx with styles matching the report requirements."""
    doc = Document()

    # Page margins (standard A4)
    for sec in doc.sections:
        sec.top_margin = Cm(2.54)
        sec.bottom_margin = Cm(2.54)
        sec.left_margin = Cm(3.18)
        sec.right_margin = Cm(3.18)

    # -- Normal (正文): 小4号(12pt), 宋体, ~33行/页 --
    s = doc.styles["Normal"]
    s.font.size = Pt(12)
    _set_style_font(s, "宋体")
    pf = s.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    pf.line_spacing = Pt(21)
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)

    # -- Heading 1 (节标题): 小3号(15pt), 黑体, 居中, 段前/后0.5行 --
    s = doc.styles["Heading 1"]
    s.font.size = Pt(15)
    s.font.bold = True
    _set_style_font(s, "黑体")
    pf = s.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    pf.line_spacing = Pt(28)
    pf.space_before = Pt(7)
    pf.space_after = Pt(7)

    # -- Heading 2 (条标题): 4号(14pt), 黑体, 段前/后0.5行 --
    s = doc.styles["Heading 2"]
    s.font.size = Pt(14)
    s.font.bold = True
    _set_style_font(s, "黑体")
    pf = s.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    pf.line_spacing = Pt(26)
    pf.space_before = Pt(7)
    pf.space_after = Pt(7)

    # -- Heading 3 (款/项标题): 小4号(12pt), 黑体, 段前/后0行 --
    s = doc.styles["Heading 3"]
    s.font.size = Pt(12)
    s.font.bold = True
    _set_style_font(s, "黑体")
    pf = s.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    pf.line_spacing = Pt(21)
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)

    # -- Body Text: 正文后续段落, 首行缩进~2字符 --
    s = doc.styles["Body Text"]
    s.font.size = Pt(12)
    _set_style_font(s, "宋体")
    pf = s.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    pf.line_spacing = Pt(21)
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.first_line_indent = Cm(0.85)

    # -- First Paragraph: 标题后首段 (may not exist, create if needed) --
    try:
        s = doc.styles["First Paragraph"]
    except KeyError:
        s = doc.styles.add_style("First Paragraph", WD_STYLE_TYPE.PARAGRAPH)
    s.font.size = Pt(12)
    _set_style_font(s, "宋体")
    pf = s.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    pf.line_spacing = Pt(21)
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.first_line_indent = Cm(0.85)

    # -- List styles --
    for name in ("List Paragraph", "List Bullet", "List Number"):
        try:
            s = doc.styles[name]
            s.font.size = Pt(12)
            _set_style_font(s, "宋体")
            pf = s.paragraph_format
            pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
            pf.line_spacing = Pt(21)
        except KeyError:
            pass

    doc.save(str(REF_DOCX))
    print(f"  -> {REF_DOCX.name}")


# ============================================================
#  Step 2: Convert HTML -> DOCX via pandoc
# ============================================================

def convert():
    """Convert HTML to DOCX using pandoc with reference styles."""
    pypandoc.convert_file(
        str(HTML_FILE),
        "docx",
        outputfile=str(OUT_DOCX),
        extra_args=[
            f"--reference-doc={REF_DOCX}",
        ],
    )
    print(f"  -> {HTML_FILE.name} => {OUT_DOCX.name}")


# ============================================================
#  Step 3: Fix cover page formatting
# ============================================================

def fix_cover_page(metadata):
    """Replace pandoc-generated cover with properly formatted cover page."""
    doc = Document(str(OUT_DOCX))
    body = doc.element.body

    # Remove the first few paragraphs that pandoc created from cover div
    # (Title, First Paragraph, Body Text x3)
    to_remove = []
    for i, p in enumerate(body):
        if p.tag == qn("w:p"):
            # Check if this is part of the cover (before first Heading 1)
            pPr = p.find(qn("w:pPr"))
            if pPr is not None:
                pStyle = pPr.find(qn("w:pStyle"))
                if pStyle is not None:
                    style_val = pStyle.get(qn("w:val"), "")
                    if "Heading1" in style_val:
                        break
            to_remove.append(p)
        if i > 10:  # Safety limit
            break

    for p in to_remove:
        body.remove(p)

    # Build proper cover page
    elems = []

    # "哈尔滨工业大学" - 小初号(36pt)
    elems.append(_make_para(
        _make_run("哈尔滨工业大学", size_pt=36),
        space_after=20,
    ))

    # blank spacer
    elems.append(_make_para(
        _make_run("", size_pt=12), space_after=10,
    ))

    # "硕士学位论文中期报告" - 二号(22pt)
    elems.append(_make_para(
        _make_run("硕士学位论文中期报告", size_pt=22),
        space_after=40,
    ))

    # 题目行
    title = metadata.get("title", "论文题目")
    elems.append(_make_para(
        [
            _make_run("题\u3000目：", east_asia="黑体", size_pt=15, bold=True),
            _make_run(title, east_asia="宋体", size_pt=15, bold=False, underline=True),
        ],
        space_before=20, space_after=40,
    ))

    # 信息字段
    fields = [
        ("院\u3000（系）", metadata.get("department", "")),
        ("学\u3000\u3000科", metadata.get("discipline", "")),
        ("导\u3000\u3000师", metadata.get("advisor", "")),
        ("研 究 生", metadata.get("student", "")),
        ("学\u3000\u3000号", metadata.get("student_id", "")),
        ("中期报告日期", metadata.get("report_date", "")),
    ]
    for label, value in fields:
        elems.append(_make_para(
            [
                _make_run(label + "\u3000", east_asia="黑体", size_pt=14, bold=False),
                _make_run(value or "\u3000" * 12, east_asia="宋体", size_pt=14, bold=False, underline=True),
            ],
            align=WD_ALIGN_PARAGRAPH.LEFT,
            space_after=14, left_indent_cm=3,
        ))

    # spacer
    elems.append(_make_para(
        _make_run("", size_pt=12), space_after=40,
    ))

    # "研究生院制"
    elems.append(_make_para(
        _make_run("研究生院制", size_pt=14),
        space_before=20,
    ))

    # page break
    pb = OxmlElement("w:p")
    r = OxmlElement("w:r")
    br = OxmlElement("w:br")
    br.set(qn("w:type"), "page")
    r.append(br)
    pb.append(r)
    elems.append(pb)

    # Insert at document start
    anchor = body[0] if len(body) > 0 else None
    if anchor is not None:
        for elem in elems:
            anchor.addprevious(elem)
    else:
        for elem in elems:
            body.append(elem)

    doc.save(str(OUT_DOCX))
    print("  -> Cover page fixed")


# ============================================================
#  Main
# ============================================================

def main():
    if not HTML_FILE.exists():
        sys.exit(f"Error: {HTML_FILE} not found")

    print("Building midterm report from HTML ...\n")

    print("[1/3] Generating reference.docx")
    create_reference()

    print("[2/3] Converting HTML -> docx")
    convert()

    print("[3/3] Fixing cover page")
    fix_cover_page(_extract_cover_info())

    print(f"\nDone -> {OUT_DOCX}")


if __name__ == "__main__":
    main()
