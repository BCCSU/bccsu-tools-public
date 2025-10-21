import numpy as np
import pandas as pd

from bccsu.reports.excel_creator import add_note, write_table, set_page_title


class TableBuilder:
    yesno = {'1': 'Yes', '0': 'No'}

    def __init__(self, anal, report):
        self.data_dictionary = anal.data_dictionary
        self.default_df = anal.full_dataset
        self.anal = anal
        self.wb = report.wb
        self.ws = self.wb.active
        self.height = 1
        self.latest = None
        self.data_dictionary_question_numbers = self.data_dictionary['adjusted_question_number'].dropna().unique()

    def reset_height(self):
        if self.wb.active != self.ws:
            self.ws = self.wb.active
            self.height = 1

    def build(self, lookup, num=None, dataset=None, custom=False, description=None, y_pos=0, lookup_contains='',
              stack=False, transpose=False, collapse=False, **kwargs):
        self.reset_height()
        if dataset is None:
            dataset = self.default_df
        tables = []

        if isinstance(lookup, pd.core.series.Series):
            # If raw series is given.
            dataset = lookup.to_frame()
            lookup = lookup.name
            custom = True

        if collapse:
            temp_df = dataset[[]].copy()

            missing_mask = (dataset[[f'{lookup}___{i}' for i in ['r', 'n', 'd']]] == '1').any(axis=1)
            for key, value in collapse.items():
                columns = [f'{lookup}___{i}' for i in value]
                temp_df[key] = (dataset[columns] == '1').any(axis=1).astype(int)
                temp_df.loc[missing_mask, key] = np.nan
            custom = True
            dataset = temp_df
            kwargs['classes'] = self.yesno
            lookup = list(collapse.keys())

        if not custom:
            var_names = self.get_var_names(lookup, contains=lookup_contains)
            if num is not None:
                lookup = num
            for var_name in var_names:
                try:
                    if description is None:
                        table = self.anal.build_table([var_name], dataset=dataset)
                        if transpose:
                            table = table.T
                            table.rename(columns={table.columns[0]: 'count'}, inplace=True)
                        table.title = f'{lookup} - {var_name}: {self.get_desc(var_name)}'
                        tables.append(table)
                except:
                    table = f'No values for {var_name}'
                    add_note(self.ws, table, [self.height + 2, y_pos], [0, 5])
                    self.height += 4

            if stack:
                tables = pd.concat(tables)
                tables.fillna('0 (0.00%)', inplace=True)
                self.height = write_table(self.ws, tables,
                                          table_start_pos=[self.height + 2, y_pos],
                                          title=table.title)[1][0]
            else:
                for table in tables:
                    self.height = write_table(self.ws, table,
                                              table_start_pos=[self.height + 2, y_pos],
                                              title=table.title)[1][0]

        else:
            if not isinstance(lookup, list):
                lookup = [lookup]
            table = self.anal.build_table(lookup, dataset=dataset, custom=True, **kwargs)
            if transpose:
                table = table.T
            table.title = f'{lookup} - {description}'

            self.height = write_table(self.ws, self.anal.build_table(lookup, dataset=dataset, custom=True, **kwargs),
                                      table_start_pos=[self.height + 2, y_pos],
                                      title=table.title)[1][0]
            tables.append(table)
        self.latest = tables
        return tables

    def write_table(self, table, title, y_pos=0):
        self.reset_height()
        self.height = write_table(self.ws, table,
                                  table_start_pos=[self.height + 2, y_pos],
                                  title=title)[1][0]
        self.latest = table

    def build_mean(self, lookup, num=None, dataset=None, y_pos=0, custom=False, description=None, lookup_contains=''):
        self.reset_height()
        if dataset is None:
            dataset = self.default_df
        tables = []
        if isinstance(lookup, pd.core.series.Series):
            # If raw series is given.
            dataset = lookup.to_frame()
            lookup = lookup.name
            custom = True
        if not custom:
            var_names = self.get_var_names(lookup, contains=lookup_contains)
            if num is not None:
                lookup = num
            for var_name in var_names:
                table = self.get_mean([var_name], df=dataset)
                table.title = f'{lookup} - {var_name}: {self.get_desc(var_name)}'
                self.height = write_table(self.ws, table,
                                          table_start_pos=[self.height + 2, y_pos],
                                          title=table.title)[1][0]
                tables.append(table)
        else:
            table = self.get_mean([lookup], df=dataset)
            table.title = f'{lookup} - {description}'
            self.height = write_table(self.ws, self.get_mean([lookup], df=dataset),
                                      table_start_pos=[self.height + 2, y_pos],
                                      title=table.title)[1][0]
            tables.append(table)
        self.latest = tables
        return tables

    def add_note(self, note):
        self.reset_height()
        add_note(self.ws, note, [self.height + 2, 0], [0, 5])
        self.height += 1

    def add_title(self, title):
        self.reset_height()
        set_page_title(self.ws, title, [self.height + 2, 0])
        self.height += 1

    def get_desc(self, variable):
        desc = self.data_dictionary[self.data_dictionary['name'] == variable]['description'].values[0]
        if desc is None:
            desc = "No description available"
        return desc

    def get_var_names(self, number, contains=''):
        number = str(number)
        if number not in self.data_dictionary_question_numbers:
            return [number]
        contains_mask = True
        if contains:
            contains_mask = self.data_dictionary['name'].str.contains(contains)

        row = self.data_dictionary[(self.data_dictionary['adjusted_question_number'] == number)
                                   & (self.data_dictionary['question_type'] != 'descriptive')
                                   & contains_mask]
        assert len(row) >= 1
        return list(row['name'].values)

    def get_mean(self, columns, df=None, ):
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
        table = pd.concat(rows)
        self.latest = table
        return table

    def desc_table(self, table):
        table.index = (table.index + ': '
                       + table.index.map(self.data_dictionary.set_index('name')['description']).fillna(
                    'No Description'))
        return table

    def get_question_info(self, question_number):
        var_names = self.get_var_names(question_number)
        print(f'Question Number: {question_number}')
        for name in var_names:
            question = self.data_dictionary[(self.data_dictionary['name'] == name)].iloc[0]
            adj_question_number = question.get('adjusted_question_number')
            print(f"""Name: {name}
Description: {question.get('description')}
Question Type: {question.get('question_type')}
Instrument: {question.get('instrument')}
Question Table:""")
            for row in question.get('question_table') or []:
                print(f"\t{row['value']} - {row['description']}")
