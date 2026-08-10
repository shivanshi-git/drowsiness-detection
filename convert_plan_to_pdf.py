import os
import subprocess
import markdown

def convert_md_to_pdf(md_path, pdf_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        md_text = f.read()

    # Convert markdown to html with extra extensions (tables, code blocks, etc.)
    html_content = markdown.markdown(md_text, extensions=['tables', 'fenced_code', 'toc', 'nl2br'])

    # Wrap in a modern CSS document layout optimized for PDF print
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Driver Drowsiness Detection Implementation Plan</title>
<style>
    @page {{
        size: A4;
        margin: 18mm 14mm 18mm 14mm;
    }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #0f172a;
        line-height: 1.55;
        font-size: 12px;
        background: #ffffff;
        padding: 0;
        margin: 0;
    }}
    h1 {{
        color: #0f172a;
        font-size: 22px;
        border-bottom: 3px solid #2563eb;
        padding-bottom: 8px;
        margin-top: 0;
    }}
    h2 {{
        color: #1e3a8a;
        font-size: 16px;
        border-bottom: 1px solid #cbd5e1;
        padding-bottom: 5px;
        margin-top: 22px;
        page-break-after: avoid;
    }}
    h3 {{
        color: #2563eb;
        font-size: 14px;
        margin-top: 16px;
        page-break-after: avoid;
    }}
    h4 {{
        color: #334155;
        font-size: 12px;
        margin-top: 12px;
        page-break-after: avoid;
    }}
    p, li {{
        font-size: 12px;
    }}
    code {{
        background-color: #f1f5f9;
        color: #0f172a;
        padding: 2px 5px;
        border-radius: 4px;
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        font-size: 11px;
    }}
    pre {{
        background-color: #0f172a;
        color: #f8fafc;
        padding: 12px;
        border-radius: 6px;
        overflow-x: auto;
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        font-size: 10px;
        line-height: 1.4;
        page-break-inside: avoid;
    }}
    pre code {{
        background-color: transparent;
        color: inherit;
        padding: 0;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 14px 0;
        font-size: 11px;
        page-break-inside: avoid;
    }}
    th {{
        background-color: #1e293b;
        color: #ffffff;
        text-align: left;
        padding: 7px 9px;
        font-weight: 600;
    }}
    td {{
        padding: 6px 9px;
        border-bottom: 1px solid #e2e8f0;
    }}
    tr:nth-child(even) {{
        background-color: #f8fafc;
    }}
    blockquote {{
        border-left: 4px solid #2563eb;
        background-color: #eff6ff;
        margin: 14px 0;
        padding: 10px 14px;
        border-radius: 0 6px 6px 0;
        color: #1e40af;
        page-break-inside: avoid;
    }}
    blockquote p {{
        margin: 0;
    }}
    hr {{
        border: 0;
        height: 1px;
        background: #cbd5e1;
        margin: 20px 0;
    }}
    ul, ol {{
        padding-left: 18px;
    }}
    li {{
        margin-bottom: 3px;
    }}
</style>
</head>
<body>
{html_content}
</body>
</html>
"""

    html_path = md_path.replace('.md', '.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(full_html)

    print(f"HTML generated at: {html_path}")

    # Use Microsoft Edge headless printing to create PDF
    edge_cmd = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        html_path
    ]

    subprocess.run(edge_cmd, check=True)
    print(f"PDF successfully generated at: {pdf_path}")

if __name__ == "__main__":
    md_file = r"C:\Users\shiva\.gemini\antigravity-ide\brain\d37f02c4-0a14-4d77-bb6d-a2a5a77e2ff9\implementation_plan.md"
    pdf_out = r"d:\drowsiness detection\Driver_Drowsiness_Detection_Implementation_Plan.pdf"
    convert_md_to_pdf(md_file, pdf_out)
