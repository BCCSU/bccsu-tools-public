# from docx import Document
# from docx.oxml.ns import nsdecls
# from docx.oxml import parse_xml
# import pandas as pd
# from docx.enum.style import WD_STYLE_TYPE
# from docx.shared import RGBColor, Pt
# from docx.enum.style import WD_STYLE_TYPE
# from docx.shared import Inches, Pt
# from docx.enum.text import WD_ALIGN_PARAGRAPH
# import docx
# import sys
# import bccsu
# from bccsu.reports.tools import write_table
# import numpy as np
# import os
# import importlib
#
# import re
# from pathlib import Path
# from datetime import date
#
# today = date.today()
#
# sas_date = today.strftime('%d%b%Y').upper()
#
# match = None
# report_in = None
# for i in Path().glob('*.docx'):
#     text = str(i)
#     match = re.match(r'[A-Z]{2}_[0-9]{4}\.docx', text)
#     if match:
#         report_in = i
#         report_out = f"{i.stem}_{sas_date}.docx"
#         break
#
# if match:
#     report = Document(str(report_in))
# else:
#     report = None
#     raise KeyError("Could not find a file of the style AA_0000.docx")
#
#
# #####STYLES#######
# style = report.styles.add_style('BCCSU_Header1', WD_STYLE_TYPE.PARAGRAPH)
# style.font.bold = True
# style.font.color.rgb = RGBColor(0x00, 0x78, 0xbf)
# style.font.size = Pt(16)
# style.font.name = 'Century Gothic'
# style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
#
# style = report.styles.add_style('BCCSU_Header2', WD_STYLE_TYPE.PARAGRAPH)
# style.font.bold = True
# style.font.color.rgb = RGBColor(0x00, 0x78, 0xbf)
# style.font.size = Pt(14)
# style.font.name = 'Century Gothic'
# style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
