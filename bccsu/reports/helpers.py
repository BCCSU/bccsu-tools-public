import pandas as pd
import numpy as np
from bccsu.reports.excel_creator import add_note, write_table


class TableBuilder:
    def __init__(self, anal, report):
        self.data_dictionary = anal.data_dictionary
        self.default_df = anal.full_dataset
        self.anal = anal
        self.wb = report.wb
        self.ws = self.wb.active
        self.height = 1

    def reset_height(self):
        if self.wb.active != self.ws:
            self.ws = self.wb.active
            self.height = 1

    def build(self, lookup, num=None, dataset=None, custom=False, description=None, **kwargs):
        self.reset_height()
        if dataset is None:
            dataset = self.default_df
        if not custom:
            var_name = self.get_var_name(lookup)
            if num is not None:
                lookup = num
            try:
                self.height = write_table(self.ws, self.anal.build_table([var_name], dataset=dataset),
                                          table_start_pos=[self.height + 2, 0],
                                          title=f'{lookup} - {var_name}: {self.get_desc(var_name)}')[1][0]
            except:
                add_note(self.ws, f'No values for {var_name}', [self.height + 2, 0], [0, 5])
                self.height += 4
        else:
            self.height = write_table(self.ws, self.anal.build_table([lookup], dataset=dataset, custom=True, **kwargs),
                                      table_start_pos=[self.height + 2, 0],
                                      title=f'{lookup} - {description}')[1][0]

    def write_table(self, table, title):
        self.reset_height()
        self.height = write_table(self.ws, table,
                                  table_start_pos=[self.height + 2, 0],
                                  title=title)[1][0]

    def build_mean(self, lookup, num=None):
        self.reset_height()
        var_name = self.get_var_name(lookup)
        if num is not None:
            lookup = num
        self.height = write_table(self.ws, self.get_mean([var_name]),
                                  table_start_pos=[self.height + 2, 0],
                                  title=f'{lookup} - {var_name}: {self.get_desc(var_name)}')[1][0]

    def add_note(self, note):
        self.reset_height()
        add_note(self.ws, note, [self.height + 2, 0], [0, 5])
        self.height += 1

    def get_desc(self, variable):
        desc = self.data_dictionary[self.data_dictionary['name'] == variable]['description'].values[0]
        if desc is None:
            desc = "No description available"
        return desc

    def get_var_name(self, number):
        number = str(number)
        if number[-1] not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            return number
        row = self.data_dictionary[self.data_dictionary['adjusted_question_number'] == number]
        assert row['question_type'].values[0] != 'descriptive'
        return row['name'].values[0]

    def get_mean(self, columns, df=None):
        if df is None:
            df = self.default_df
        rows = []
        existing_columns = [c for c in columns if c in df.columns]
        for column in existing_columns:
            x = df[column].copy()
            x[x == 'D'] = np.nan
            x[x == 'R'] = np.nan
            x[x == 'N'] = np.nan

            x = x.astype(float)
            row = pd.DataFrame([{'Mean': x.mean(),
                                 'std': x.std(),
                                 'Median': np.nan if x.isna().all() else x.median(),
                                 '25 quartile': x.quantile(0.25),
                                 '75 quartile': x.quantile(0.75),
                                 'N': (~x.isna()).sum()}], index=[column])
            rows.append(row)
        return pd.concat(rows)

    def desc_table(self, table):
        table.index = (table.index + ': '
                       + table.index.map(self.data_dictionary.set_index('name')['description']).fillna(
                    'No Description'))
        return table
