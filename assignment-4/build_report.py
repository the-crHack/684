"""Build the report from Markdown plus measured results; render every page for QA."""
import json
import argparse
import re
from html import escape
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
import pymupdf as fitz

HERE = Path(__file__).resolve().parent

def inline(text):
    text = escape(text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'(https://[^\s]+)', r'<link href="\1" color="#245477">\1</link>', text)
    return text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', type=Path, default=HERE/'results/openrouter/summary.json')
    args = parser.parse_args()
    output = HERE/'output/pdf/assignment_4_report.pdf'
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='ReportBody', fontName='Helvetica', fontSize=10.2,
                              leading=14.1, spaceAfter=7, textColor=colors.HexColor('#172837')))
    styles.add(ParagraphStyle(name='ReportTitle', fontName='Helvetica-Bold', fontSize=21,
                              leading=25, spaceAfter=11, textColor=colors.HexColor('#153c50')))
    styles.add(ParagraphStyle(name='ReportSection', fontName='Helvetica-Bold', fontSize=12.5,
                              leading=16, spaceBefore=10, spaceAfter=6, keepWithNext=True))
    styles.add(ParagraphStyle(name='ReferenceText', fontName='Helvetica', fontSize=9,
                              leading=12.5, spaceAfter=10, splitLongWords=True))
    flow = []
    summary_path = args.summary
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {'status':'not_run','summary':[]}
    references = False
    for block in (HERE/'report.md').read_text().split('\n\n'):
        text = ' '.join(line.strip() for line in block.splitlines())
        if text.startswith('## References'):
            flow.append(PageBreak())
            references = True
        if text.startswith('## Task C:'):
            flow.append(PageBreak())
        if text.startswith('# '):
            flow.append(Paragraph(inline(text[2:]), styles['ReportTitle']))
        elif text.startswith('## '):
            flow.append(Paragraph(inline(text[3:]), styles['ReportSection']))
            if text.startswith('## Results') and summary['summary']:
                primary = summary.get('primary_study', {})
                caption = (f"Primary study: {primary.get('status','unknown')} "
                           f"({primary.get('recorded_requests',0)}/{primary.get('planned_requests',0)}). "
                           f"Optional grid: {summary['status']} ({summary.get('recorded_requests',0)}/{summary.get('planned_requests',0)}). "
                           'Table reports original scenarios only; high-effort rows are two-case probes.')
                flow.append(Paragraph(caption, styles['ReportBody']))
                data = [['Model / effort','Pipeline','Correct / tried','Strict accuracy']]
                for row in summary['summary']:
                    if row['kind'] == 'original':
                        names = {'liquid/lfm-2.5-2.6b:free':'LFM 2.6B', 'nvidia/nemotron-3.5-super-vl:free':'Nemotron 120B',
                                 'nvidia/nemotron-3-ultra-550b-a55b:free':'Nemotron 550B'}
                        name = names.get(row['model'],row['model'].split('/')[-1])
                        data.append([name+' / '+row['effort'], row['pipeline'], f"{row['correct']} / {row['attempted']}", f"{row['strict_accuracy']:.0%}"])
                table = Table(data, colWidths=[150,100,110,110], repeatRows=1, hAlign='LEFT')
                table.setStyle(TableStyle([('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'), ('FONTSIZE',(0,0),(-1,-1),9),
                    ('BOTTOMPADDING',(0,0),(-1,-1),6), ('TOPPADDING',(0,0),(-1,-1),6),
                    ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eaf0f3')),
                    ('LINEBELOW',(0,0),(-1,0),0.5,colors.HexColor('#78909a'))]))
                flow.extend([table, Spacer(1,8)])
        elif text:
            formatted = inline(text)
            if text.startswith('**Team 08'):
                formatted = formatted.replace('</b> Team members:', '</b><br/>Team members:').replace('Fnu Startup:', 'Fnu<br/>Startup:')
            flow.append(Paragraph(formatted, styles['ReferenceText' if references else 'ReportBody']))
    def footer(canvas, doc):
        canvas.setFont('Helvetica',8)
        canvas.setFillColor(colors.HexColor('#617582'))
        canvas.drawString(44,27,'Team 08 | SynthHealth | Assignment 4')
        canvas.drawRightString(568,27,str(doc.page))
    doc = SimpleDocTemplate(str(output), pagesize=letter, leftMargin=44, rightMargin=44,
                            topMargin=40, bottomMargin=43, title='Assignment 4: Contextual Integrity', author='Team 08')
    doc.build(flow, onFirstPage=footer, onLaterPages=footer)
    pdf = fitz.open(output)
    ref_page = next(i for i,p in enumerate(pdf) if 'References' in p.get_text().splitlines())
    if ref_page > 5:
        raise ValueError(f'{ref_page} content pages exceeds assignment limit')
    render = HERE/'tmp/report-preview'
    render.mkdir(parents=True, exist_ok=True)
    for i,page in enumerate(pdf):
        page.get_pixmap(matrix=fitz.Matrix(1.25,1.25)).save(render/f'page-{i+1}.png')
    print(f'{output}: {ref_page} content pages + {len(pdf)-ref_page} reference pages; {summary["status"]}')

if __name__ == '__main__':
    main()
