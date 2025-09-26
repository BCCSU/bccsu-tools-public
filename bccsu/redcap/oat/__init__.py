import numpy as np
import pandas as pd
import re
from bccsu.bccsu.stats import counts, text_counts, value_counts_single
import ast
import warnings


class AnalyzeOAT:
    def __init__(self, full_dataset, data_dictionary):
        self.full_dataset = full_dataset
        self.data_dictionary = data_dictionary

    def create_restriction_mask(self, column, dataset):
        s = self.data_dictionary[self.data_dictionary['name'] == column]['restrictions'].values[0]
        if s is None:
            return np.ones(dataset.shape[0]).astype(bool)
        if 'arm-number' in s:
            # This is something new Rhea added. I just have to skip the restriction until I know how to handle it...
            warnings.warn(f'arm-number is in restriction: {column}')
            return np.ones(dataset.shape[0]).astype(bool)

        def replacer(match):
            m = match.group(0)
            if '=' in m:
                m = m.replace(' = ', '=')
                m = m.replace(' =', '=')
                column, value = m.split('=')
                relation = '=='
            elif '<>' in m:
                column, value = m.split('<>')
                relation = '!='
            elif '>' in m:
                column, value = m.split('>')
                relation = ''
            else:
                raise Exception('Relation not handled yet.')
            column = column.replace('(', '___')
            column = column.replace(')', '')
            column = column.replace('', '')
            if '"' not in value and "'" not in value:
                value = f"'{value}'"
            return f'(dataset[\'{column[1:-1]}\'] {relation} {value})'

        s = re.sub(r"\[(.+?)\]\s*(?:=|<>)\s*(?:(?:'([^']*)')|(?:\"([^\"]*)\")|(\d+))", replacer, s)

        s = s.replace(' and ', ' AND ')
        s = s.replace(' or ', ' OR ')

        s = s.replace(' AND ', '&')
        s = s.replace(' OR ', '|')
        return eval(compile(s, '<string>', 'eval'))

    def convert_to_float(self, array):
        return np.where(array.isin(['R', 'D', 'NA', None]), np.nan, array).astype('float')

    def _build_table(self, columns, dataset=None, custom=False, checkbox=False, use_mask=True):
        if dataset is None:
            dataset = self.full_dataset
        if isinstance(columns, str):
            columns = [columns]

        columns = [c for c in columns if c != '']

        checkbox = False
        if not custom:
            columns_corrected = []
            rename_rows = {}
            for column in columns:
                if column not in dataset.columns:
                    data_rec = self.data_dictionary[self.data_dictionary['name'] == column]
                    # levels = ast.literal_eval(data_rec['question_table'].iloc[0])
                    levels = data_rec['question_table'].iloc[0]
                    rename_rows.update({l['variable_name']: l['description'] for l in levels})
                    for l in levels:
                        columns_corrected.append(l['variable_name'])
                    checkbox = True
                else:
                    columns_corrected.append(column)
            columns = columns_corrected

        columns_names = list(set([column.split('___')[0] for column in columns]))
        text_columns = []
        masks = {}
        all_same = False
        classes = {}
        if not custom:
            for i, column in enumerate(columns_names):
                data_rec = self.data_dictionary[self.data_dictionary['name'] == column]
                data_type = data_rec['question_type'].values[0]
                mask = self.create_restriction_mask(column, dataset)
                mask_set = {column_full: mask for column_full in columns if column_full.startswith(column)}
                masks.update(mask_set)

                if i == 0:
                    # classes = ast.literal_eval(data_rec['question_table'].iloc[0]) # Used if converting string to dict
                    classes = data_rec['question_table'].iloc[0]
                    classes = {c['value']: c['description'] for c in classes}
                    all_same = True
                else:
                    # next_class = ast.literal_eval(data_rec['question_table'].iloc[0])
                    next_class = data_rec['question_table'].iloc[0]
                    next_class = {c['value']: c['description'] for c in next_class}
                    if classes != next_class:
                        all_same = False

                if data_type == 'text':
                    text_columns.append(column)

        counts_list = []
        for i, column in enumerate(columns):
            if not custom and use_mask:
                dataset_restricted = dataset[masks[column]]
            else:
                dataset_restricted = dataset

            table = value_counts_single(
                dataset_restricted[column].fillna('nan').astype(str).str.replace('.0', '', regex=False).replace('NA',
                                                                                                                'nan'))
            if column not in text_columns:
                table = table.T
                table.index = [table.columns.name]
                if all_same:
                    if checkbox:
                        classes = {'1': 'Yes', '0': 'No'}
                        table = table.rename(index=rename_rows)
            else:
                table['Text Response'] = table.index
                table = table.reset_index(drop=True)
                table['variable_name'] = column
                table = table.set_index('variable_name')
                table.index.name = None
                table = table.rename(columns={column: 'Total'})
                table = table[~table['Text Response'].isin(['nan', 'Total'])]
                try:
                    if table.shape[0] > 0:
                        table['Total'] = table['Total'].str.extract(r'^(\d+) .*?$').astype(int)
                    else:
                        table['Total'] = 0
                except KeyError:
                    pass
            table = table.drop(columns=['nan'], errors='ignore')
            counts_list.append(table)
        try:
            current_counts = pd.concat(counts_list)
        except ValueError:
            current_counts = pd.DataFrame(['-'], index=['No Data'], columns=[column])
        return current_counts, classes

    def build_table(self, *args, strat='', classes=None, **kwargs):
        classes_implicit = None
        try:
            dataset = kwargs.pop('dataset')
        except KeyError:
            dataset = self.full_dataset

        if strat:
            strat_var = np.sort(dataset[strat].unique())
            tables = []

            for s in strat_var:
                table, classes_implicit = self._build_table(*args, dataset=dataset[dataset[strat] == s], **kwargs)
                table[strat] = s
                tables.append(table)
            final = pd.concat(tables)
            final = final.reset_index().set_index(['index', strat])
        else:
            final, classes_implicit = self._build_table(*args, dataset=dataset, **kwargs)

        final = final.reindex(sorted(final.columns), axis=1)
        columns = [column for column in final.columns if column not in ['Total', 'Text Response']]
        if 'Text Response' in final.columns:
            # final.loc[final['Text Response'].isna(), columns] = final.loc[final['Text Response'].isna(),
            # columns].fillna('0 (0.00%)')
            final_old = final
            final = final.drop(columns=['Total', 'Text Response'], errors='ignore')
            if 'Total' in final_old.columns:
                final['Total'] = final_old['Total']
            final['Text Response'] = final_old['Text Response']
        else:
            # final.loc[:, columns] = final.loc[:, columns].fillna('0 (0.00%)')
            pass

        if classes:
            final = final.rename(columns=classes)
        elif classes_implicit:
            final = final.rename(columns=classes_implicit)

        column_order = final.columns.to_list()
        for col in ['Yes', 'No', 'Missing', 'R', 'D', 'NA', 'Total']:
            if col in column_order:
                column_order.remove(col)
                column_order.append(col)

        return final[column_order]

    def relabel_radial_data(self, name, variable, items, text_variables=None, text_responses=None):
        missing_labels = ['NA', 'R', 'D']
        df = self.full_dataset[self.create_restriction_mask(variable, self.full_dataset)].copy()
        all_text_values = []
        df[name] = None
        for item, category in items.items():
            df.loc[df[variable].isin([str(c) for c in category]) & df[name].isna(), name] = item
            if text_responses is not None:
                text_values = text_responses.get(item, [])
                if text_values:
                    df.loc[df[text_variables].isin(text_values).any(axis=1) & df[name].isna(), name] = item
                for text_value in text_values:
                    all_text_values.append(text_value)
        if text_variables is not None:
            for text_variable in text_variables:
                text_missing = df[~df[text_variable].isin(all_text_values + missing_labels)][[text_variable]].dropna()
                if text_missing.shape[0] > 0:
                    print(f"""{text_variable}:
--------------------------------
{'\n'.join(text_missing[text_variable].dropna().unique())}
--------------------------------""")
        missing_mask = self.full_dataset[variable].isin(missing_labels)
        df.loc[missing_mask, name] = self.full_dataset.loc[missing_mask, variable]

        df.loc[df[name].isna(), name] = 'nan'

        return self.build_table(name, custom=True, dataset=df)

    def relabel_checkbox_data(self, variable, items, text_variables=None, text_responses=None):
        df = self.full_dataset[self.create_restriction_mask(variable, self.full_dataset)].copy()
        all_text_values = []
        for item, category in items.items():
            df[item] = (df[[f'{variable}___{i}' for i in category]] == '1').any(axis=1).astype(int)
            if text_responses is not None:
                text_values = text_responses.get(item, [])
                if text_values:
                    df.loc[df[text_variables].isin(text_values).any(axis=1) & df[item].isna(), item] = 1
                for text_value in text_values:
                    all_text_values.append(text_value)
        if text_variables is not None:
            for text_variable in text_variables:
                text_missing = df[~df[text_variable].isin(all_text_values)][[text_variable]].dropna()
                if text_missing.shape[0] > 0:
                    print(f"""{text_variable}:
--------------------------------
{'\n'.join(text_missing[text_variable].dropna().unique())}
--------------------------------""")
        return self.build_table(list(items.keys()), custom=True, dataset=df, classes={'1': 'Yes', '0': 'No'})

    def search_columns(self, search_string):
        columns = pd.Series(self.full_dataset.columns)
        mask = columns.str.contains(search_string)
        return columns[mask].to_list()
