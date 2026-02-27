#!/usr/bin/env python3
"""
Convert FULL_PAPER_DRAFT.md to PDF using markdown + weasyprint.
Run: python3 paper/generate_pdf.py
"""
import markdown
import weasyprint
import re
import os

WORKDIR = os.path.dirname(os.path.abspath(__file__))
MD_FILE = os.path.join(WORKDIR, "FULL_PAPER_DRAFT.md")
HTML_FILE = os.path.join(WORKDIR, "bpfc_paper.html")
PDF_FILE = os.path.join(WORKDIR, "bpfc_paper.pdf")

CSS = """
@page {
    size: A4;
    margin: 2.5cm 2.5cm 2.5cm 2.5cm;
    @top-center {
        content: "Bayesian Posterior Factual Calibration in Diffusion LMs";
        font-size: 9pt;
        color: #555;
    }
    @bottom-center {
        content: counter(page);
        font-size: 9pt;
    }
}
body {
    font-family: "Times New Roman", Times, serif;
    font-size: 11pt;
    line-height: 1.5;
    color: #1a1a1a;
    max-width: none;
}
h1 {
    font-size: 17pt;
    font-weight: bold;
    text-align: center;
    margin-top: 0;
    margin-bottom: 0.3em;
}
h2 {
    font-size: 13pt;
    font-weight: bold;
    margin-top: 1.5em;
    margin-bottom: 0.4em;
    border-bottom: 1px solid #999;
    padding-bottom: 2px;
}
h3 {
    font-size: 11.5pt;
    font-weight: bold;
    margin-top: 1.2em;
    margin-bottom: 0.3em;
}
h4 {
    font-size: 11pt;
    font-weight: bold;
    font-style: italic;
    margin-top: 1em;
}
p {
    margin-top: 0.5em;
    margin-bottom: 0.5em;
    text-align: justify;
}
code {
    font-family: "Courier New", Courier, monospace;
    font-size: 9.5pt;
    background: #f0f0f0;
    padding: 1px 3px;
    border-radius: 2px;
}
pre {
    font-family: "Courier New", Courier, monospace;
    font-size: 9pt;
    background: #f5f5f5;
    border-left: 3px solid #aaa;
    padding: 0.5em 1em;
    overflow-x: auto;
    white-space: pre-wrap;
    word-break: break-all;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
    font-size: 10pt;
}
th, td {
    border: 1px solid #ccc;
    padding: 4px 8px;
    text-align: left;
}
th {
    background: #eaeaea;
    font-weight: bold;
}
tr:nth-child(even) { background: #f9f9f9; }
blockquote {
    border-left: 3px solid #2a7ae2;
    padding-left: 1em;
    color: #444;
    font-style: italic;
    margin: 1em 0;
}
.abstract-box {
    border: 1px solid #ccc;
    background: #fafafa;
    padding: 1em;
    margin: 1em 2em;
    font-size: 10.5pt;
}
hr {
    border: none;
    border-top: 1px solid #ddd;
    margin: 1.5em 0;
}
ul, ol {
    margin: 0.5em 0;
    padding-left: 2em;
}
li {
    margin-bottom: 0.3em;
}
.page-break {
    page-break-after: always;
}
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="author" content="Dr. Claw Research Agent"/>
<title>BPFC: Bayesian Posterior Factual Calibration in Diffusion Language Models</title>
<style>{css}</style>
</head>
<body>
{body}
</body>
</html>"""

def main():
    print(f"Reading {MD_FILE}...")
    with open(MD_FILE, "r") as f:
        md_content = f.read()
    
    # Pre-process: wrap abstract in a styled box
    # Replace the abstract section specially
    md_content = re.sub(
        r'(## Abstract\n)(.*?)(\n## [1-9])',
        r'<div class="abstract-box"><strong>Abstract</strong><br/>\2</div>\n\3',
        md_content, count=1, flags=re.DOTALL
    )
    
    print("Converting Markdown → HTML...")
    md_ext = ["tables", "fenced_code", "codehilite", "footnotes", "toc"]
    try:
        body = markdown.markdown(md_content, extensions=md_ext)
    except Exception:
        body = markdown.markdown(md_content, extensions=["tables", "fenced_code", "footnotes"])
    
    html = HTML_TEMPLATE.format(css=CSS, body=body)
    
    with open(HTML_FILE, "w") as f:
        f.write(html)
    print(f"HTML written to {HTML_FILE}")
    
    print("Converting HTML → PDF (weasyprint)...")
    doc = weasyprint.HTML(filename=HTML_FILE)
    doc.write_pdf(PDF_FILE)
    
    size_kb = os.path.getsize(PDF_FILE) / 1024
    print(f"✅ PDF written: {PDF_FILE} ({size_kb:.0f} KB)")
    return PDF_FILE

if __name__ == "__main__":
    main()
