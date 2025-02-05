import xlsxwriter
import pandas as pd
import numpy as np


def to_excel(df, save_as):
    df = df.fillna('N/A')
    workbook = xlsxwriter.Workbook(save_as, {'nan_inf_to_errors': True})

    all_cells_format_dict = {
        'font_name': 'Arial',
        'font_size': 9.5,
        'border': 1,
        'align': 'right',
        'border_color': '#B0B7BB'}

    good_cells_format_dict = all_cells_format_dict.copy()
    good_cells_format_dict.update({
        'fg_color': '#EDF2F9',
        'font_color': '#C6EFCE'})

    headers_indices_format_dict = all_cells_format_dict.copy()
    headers_indices_format_dict.update({
        'bold': True,
        'text_wrap': False,
        'valign': 'top',
        'align': 'left',
        'fg_color': '#EDF2F9',
        'font_color': '#112277'})

    all_cells_format = workbook.add_format(all_cells_format_dict)
    headers_indices_format = workbook.add_format(headers_indices_format_dict)
    worksheet = workbook.add_worksheet()

    if isinstance(df.index, pd.MultiIndex):
        indices = df.index.names
    else:
        indices = [df.index.name]

    indices_count = len(indices)

    if isinstance(list(df)[0], tuple):
        headers_count = len(list(df)[0])
        headers = np.array([list(i) for i in list(df)]).T
        padding = np.array([indices for i in range(headers_count)])
        header_row = np.concatenate([padding, headers], axis=1)
    else:
        header_row = np.array([indices + list(df)])
        headers_count = 1

    # All values in table
    full_table = np.concatenate([header_row, df.reset_index().values])
    for y_pos, y in enumerate(full_table[headers_count:], start=headers_count):
        for x_pos, val in enumerate(y[indices_count:], start=indices_count):
            worksheet.write(y_pos, x_pos, val, all_cells_format)


    # Find positions
    def find_positions(table, pos_count_1, pos_count_2):
        pos_h = np.array([pos_count_1 for i in range(pos_count_2)])
        positions = pos_h.copy()[np.newaxis, :]
        y_prev = table[pos_count_1, :pos_count_2]
        for y_pos, y in enumerate(table[pos_count_1:], start=pos_count_1):
            for x_pos, val in enumerate(y[:pos_count_2]):
                if (y_prev[x_pos] != y[x_pos]):
                    pos_h[x_pos:] = y_pos
                    positions = np.concatenate([positions, pos_h[np.newaxis, :]], axis=0)
                    break
            y_prev = y
        return positions

    def create_merged_cells(table, positions, headers=False):
        # Create merged indices
        for pos_x, i in enumerate(positions.T):
            uniq_vals = np.unique(i)
            last_filled_pos = 0
            table_size = table.shape[0] - 1
            for pos, j in enumerate(uniq_vals):
                try:
                    if j+1 == uniq_vals[pos+1]:
                        last_filled_pos = j
                        if headers:
                            worksheet.write(pos_x, j, table[j, pos_x], headers_indices_format)
                        else:
                            worksheet.write(j, pos_x, table[j, pos_x], headers_indices_format)
                    else:
                        last_filled_pos = uniq_vals[pos+1]-1
                        if headers:
                            worksheet.merge_range(pos_x, j, pos_x, last_filled_pos, table[j, pos_x],
                                                  headers_indices_format)
                        else:
                            worksheet.merge_range(j, pos_x, last_filled_pos, pos_x, table[j, pos_x],
                                                  headers_indices_format)
                except IndexError:
                    if last_filled_pos == table_size - 1:
                        if headers:
                            worksheet.write(pos_x, j, table[-1, pos_x], headers_indices_format)
                        else:
                            worksheet.write(j, pos_x, table[-1, pos_x], headers_indices_format)
                    else:
                        if headers:
                            worksheet.merge_range(pos_x, j, pos_x, table_size, table[-1, pos_x],
                                                  headers_indices_format)
                        else:
                            worksheet.merge_range(j, pos_x, table_size, pos_x, table[-1, pos_x],
                                                  headers_indices_format)

    positions = find_positions(full_table, headers_count, indices_count)

    positions_head = find_positions(full_table.T, indices_count, headers_count)

    create_merged_cells(full_table, positions)

    create_merged_cells(full_table.T, positions_head, headers=True)

    # Fill top left corner in.
    if (headers_count - 2) == 0 and (indices_count - 1) == 0:
        worksheet.write(0, 0, '', headers_indices_format)
    elif (headers_count - 2) != -1:
        worksheet.merge_range(0, 0, headers_count - 2, indices_count - 1, '', headers_indices_format)

    # Fill in index names.
    for pos, i in enumerate(full_table[0, :indices_count]):
        worksheet.write(headers_count - 1, pos, i, headers_indices_format)

    workbook.close()

    return df
