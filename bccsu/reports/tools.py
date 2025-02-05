import docx
import pyperclip
from docx import Document
import pandas as pd
from bccsu.reports.excel_tools import *
from styleframe import StyleFrame


def dict_to_table(dictionary):
    def_table = []
    for key, value in dictionary.items():
        def_table.append([key, value])
    return def_table


def pd_to_table(table_in):
    labels = list(table_in)
    values = table_in.values.tolist()
    values.insert(0, labels)
    return values


def pretty_print_p_value(p):
    if float(p) < .0001:
        p_str = "<.0001"
    else:
        p_str = f"{p:.4f}"
    return p_str

def to_excel(df, save_as):
    # No columns can be named 'index' for some reason...
    writer = StyleFrame.ExcelWriter(save_as)

    cols = df.columns

    if isinstance(df.index, pd.MultiIndex):
        indices = df.index.names
    else:
        indices = [df.index.name]

    df = df.reset_index()

    sf = StyleFrame(df)

    # columns
    column_style = CStyler(border_color='B0B7BB',
                           border_type=utils.borders.thin,
                           bg_color=utils.colors.white,
                           bold=False,
                           font=utils.fonts.arial,
                           font_size=9.5)

    sf.apply_column_style(cols_to_style=cols,
                          styler_obj=column_style,
                          style_header=True)

    # Headers
    header_style = CStyler(border_color='B0B7BB',
                           border_type=utils.borders.thin,
                           bg_color='#EDF2F9',
                           bold=True,
                           font_size=9.5,
                           font_color='#112277',
                           number_format=utils.number_formats.general,
                           protection=False)

    sf.apply_headers_style(style_index_header=True, styler_obj=header_style)

    # Indices
    sf.apply_column_style(cols_to_style=indices,
                          styler_obj=header_style,
                          style_header=True)

    sf.to_excel(writer, sheet_name='Sheet1', index=False)

    sheet = writer.sheets['Sheet1']

    sheet.merge_range('A1:A5', 'Merged Range')

    writer.save()
