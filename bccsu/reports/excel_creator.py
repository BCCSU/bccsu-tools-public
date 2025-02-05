from pathlib import Path
import openpyxl
from openpyxl.styles import Alignment
import win32com.client
import tempfile
import json
from openpyxl.drawing.image import Image
from openpyxl.utils import get_column_letter
import pandas as pd
from openpyxl.styles import numbers


def load_workbook(filename, template=None, *args, **kwargs):
    if template is None:
        template = Path(__file__).parent / 'templates/template.xlsx'
    else:
        template = Path(template)
    destination = Path(filename)
    destination.write_bytes(template.read_bytes())
    return openpyxl.load_workbook(filename, *args, **kwargs)


def set_page_title(ws, title, pos=None):
    if pos is None:
        pos = [0, 0]
    cell = ws.cell(row=pos[0] + 1, column=pos[1] + 1, value=title)
    cell.style = 'SAS page title'


def add_note(ws, note, position=None, merge=None):
    if position is None:
        position = [1, 1]
    position[0] += 1
    position[1] += 1
    cell = ws.cell(row=position[0], column=position[1], value=note)
    if merge:
        ws.merge_cells(start_row=position[0],
                       start_column=position[1],
                       end_row=position[0] + merge[0],
                       end_column=position[1] + merge[1])
    cell.style = 'SAS note'


def get_contiguous_sequences(codes):
    previous_index = [0 for _ in codes[0]]
    final_position = len(codes[0]) - 1
    groups = []
    for col in codes:
        previous_code = col[0]
        start = 0
        sub_groups = []
        for pos, code in enumerate(col):
            if pos != 0:
                if previous_code != code or previous_index[pos] != previous_index[pos - 1]:
                    if pos - start > 1:
                        sub_groups.append([start, pos - 1])
                    start = pos
            previous_code = code
        if final_position - start > 0:
            sub_groups.append([start, final_position])
        groups.append(sub_groups)
        previous_index = col
    return groups


def get_index_codes(index):
    if index.nlevels > 1:
        return index.codes
    else:
        unique_values = index.unique().to_list()
        return [[unique_values.index(value) for value in index]]


def write_table(ws, df, table_start_pos=None, write_index=True, write_headers=True, title=None, percent=False):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = pd.MultiIndex.from_tuples(
            tuple(str(item) for item in level) for level in df.columns
        )
    else:
        df.columns = df.columns.astype(str)

    if table_start_pos is None:
        table_start_pos = [0, 0]

    table_original_start = table_start_pos

    if write_index:
        index_levels = df.index.nlevels
        index_groupings = get_contiguous_sequences(get_index_codes(df.index))
    else:
        index_levels = 0
        index_groupings = []

    if write_headers:
        column_levels = df.columns.nlevels
        header_groupings = get_contiguous_sequences(get_index_codes(df.columns))
    else:
        column_levels = 0
        header_groupings = []

    if title:
        cell = ws.cell(row=table_start_pos[0] + 1, column=table_start_pos[1] + 1, value=title)
        cell.style = 'SAS Title'
        cell.alignment = Alignment(horizontal='center', vertical='center')
        ws.merge_cells(start_row=table_start_pos[0] + 1,
                       start_column=table_start_pos[1] + 1,
                       end_row=table_start_pos[0] + 1,
                       end_column=table_start_pos[1] + index_levels + df.shape[1])
        table_start_pos[0] += 1

    if write_index:
        df_temp = df.reset_index()
    else:
        df_temp = df

    for column_n, (headers, column) in enumerate(df_temp.to_dict().items()):
        if isinstance(headers, str):
            headers = [headers]
        is_index = column_n < index_levels

        current_column = column_n + table_start_pos[1] + 1
        for row_n, value in enumerate(column.values()):
            if row_n == 0 and write_headers:
                for header_pos, header in enumerate(headers):
                    if is_index and (headers[0][:5] in ['level', 'index']):
                        header = ''
                    cell = ws.cell(row=table_start_pos[0] + 1 + header_pos, column=current_column, value=header)
                    cell.style = 'SAS Title'
                    cell.alignment = Alignment(horizontal='center', vertical='center')
            current_row = row_n + table_start_pos[0] + 1 + column_levels
            cell = ws.cell(row=current_row, column=current_column, value=value)
            if is_index:
                cell.style = 'SAS Title'
            else:
                cell.style = 'SAS Data'
                if percent:
                    cell.number_format = numbers.FORMAT_PERCENTAGE_00

    for i, index_grouping in enumerate(index_groupings):
        for group in index_grouping:
            ws.merge_cells(start_row=table_start_pos[0] + 1 + group[0] + column_levels,
                           start_column=table_start_pos[1] + 1 + i,
                           end_row=table_start_pos[0] + 1 + group[1] + column_levels,
                           end_column=table_start_pos[1] + 1 + i)

    for i, index_grouping in enumerate(header_groupings):
        for group in index_grouping:
            ws.merge_cells(start_row=table_start_pos[0] + 1,
                           start_column=table_start_pos[1] + 1 + i + group[0] + index_levels,
                           end_row=table_start_pos[0] + 1,
                           end_column=table_start_pos[1] + 1 + i + group[1] + index_levels)

    if column_levels > 1 or index_levels > 1:
        # Untested. But should work.
        ws.merge_cells(start_row=table_start_pos[0] + 1,
                       start_column=table_start_pos[1] + 1,
                       end_row=table_start_pos[0] + column_levels,
                       end_column=table_start_pos[1] + index_levels)

    # Makes cells background color change

    # for column in ws.columns:
    #     max_length = 0
    #     column = [cell for cell in column if cell.value]  # Checking if the cell has a value
    #     for cell in column:
    #         try:  # Necessary to avoid error on empty cells
    #             if len(str(cell.value)) > max_length:
    #                 max_length = len(cell.value)
    #         except:
    #             pass
    #     adjusted_width = (max_length + 2)  # Additional +2 for a bit of margin
    #     try:
    #         ws.column_dimensions[column[0].column_letter].width = min(adjusted_width, 100)
    #     except:
    #         pass

    return [table_original_start,
            [table_start_pos[0] + column_levels + (1 if title else 0) + df.shape[0],
             table_start_pos[1] + index_levels + df.shape[1]]]


def insert_image(ws, image_path, pos):
    img = Image(image_path)
    column_letter = get_column_letter(pos[1] + 1)
    row = pos[0] + 1
    ws.add_image(img, f'{column_letter}{row}')


def run_macro(macro_name, arguments):
    temp_file_path = Path(tempfile.gettempdir()) / 'workbookargs.json'
    with open(temp_file_path, 'w') as f:
        f.write(json.dumps(arguments))

    xl = win32com.client.Dispatch("Excel.Application")
    xl.Workbooks.Open(Filename=Path(Path(__file__).parent / 'templates/macros.xlsm'))
    xl.Application.Run(macro_name)
    xl.Application.Quit()
