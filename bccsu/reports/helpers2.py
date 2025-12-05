import numpy as np
import pandas as pd

from bccsu.reports.excel_creator import add_note, write_table, set_page_title, add_title

class TableBuilder:
    yesno = {'1': 'Yes', '0': 'No'}

    def __init__(self, data, report, make_na=False):
        """

        :param data:
        :param report:
        :param make_na: Sets R,D,N to NA.
        """
        self.data_dictionary = data.meta
        self.default_df = data.df
        self.anal = data
        self.wb = report.wb
        self.ws = self.wb.active
        self.height = 1
        self.latest = None
        self.restriction = None
        self.data_dictionary_question_numbers = self.data_dictionary['adjusted_question_number'].dropna().unique()
        self.make_na = make_na

    def reset_height(self):
        if self.wb.active != self.ws:
            self.ws = self.wb.active
            self.height = 1

    def set_restriction(self, restriction, name):
        self.add_note(f'Restriction: {name}')
        self.restriction = restriction
        # todo check restriction whenever analyzing

    def compare(self, n1, n2, title=None, mask=None, mcnemar_test=False, write=True):
        if mask is not None:
            restriction = mask & self.restriction
        else:
            restriction = self.restriction
        c = self.anal.comp(n1, n2, restriction=restriction, set_missing_na=self.make_na, mcnemar_test=mcnemar_test)
        if write:
            self.write_table(c, title=c.title)
        return c

    def build(self, lookup, num=None, dataset=None, custom=False, description=None, y_pos=0, lookup_contains='',
              stack=False, transpose=None, collapse=False, title=None, write=True, mask=None, **kwargs):
        self.reset_height()
        if dataset is None:
            dataset = self.default_df
        tables = []

        if mask is not None:
            restriction = mask
            if self.restriction is not None:
                restriction &= self.restriction
        else:
            restriction = self.restriction

        if isinstance(lookup, pd.core.series.Series):
            # If raw series is given.
            dataset = lookup.to_frame()
            lookup = lookup.name
            custom = True
        if isinstance(lookup, str):
            cols = self.anal.question_lookup(lookup)
            results = []
            for col in cols:
                counts = self.anal.pretty_counts(col, restriction=restriction, set_missing_na=self.make_na)
                results.append(counts)
                meta = self.anal.meta.loc[col]
                if meta['question_category'] == 'checkbox':
                    title = f'{meta['question_number']} - {col}: {meta["description"]}'
                else:
                    title = None
                if transpose:
                    counts = counts.T
                else:
                    shape = counts.shape
                    if shape[0] < shape[1]:
                        counts = counts.T
                if write:
                    self.write_table(counts, title=title)
        elif isinstance(lookup, list):
            cols = lookup
            results = self.anal.pretty_counts(cols, restriction=restriction, set_missing_na=self.make_na)
            if transpose:
                results = results.T
            else:
                shape = results.shape
                if shape[0] < shape[1]:
                    results = results.T
            if write:
                self.write_table(results, title=title)
        else:
            raise ValueError('Invalid lookup')
        return results

    def write_table(self, table, title=None, y_pos=0):
        self.reset_height()
        self.height = write_table(self.ws, table,
                                  table_start_pos=[self.height + 2, y_pos],
                                  title=title)[1][0]
        self.latest = table
        return table

    def add_note(self, note):
        self.reset_height()
        add_note(self.ws, note, [self.height + 2, 0], [0, 5])
        self.height += 1

    def add_title(self, title):
        self.reset_height()
        add_title(self.ws, title, [self.height + 3, 0])
        self.height += 2

    def get_desc(self, variable):
        desc = self.data_dictionary[self.data_dictionary['name'] == variable]['description'].values[0]
        if desc is None:
            desc = "No description available"
        return desc

    def get_type(self, variable):
        qtype = self.data_dictionary[self.data_dictionary['name'] == variable]['question_type'].values[0]
        return qtype

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

    def get_question_table(self, question_number):
        var_names = self.get_var_names(question_number)
        name = var_names[0]
        question = self.data_dictionary[(self.data_dictionary['name'] == name)].iloc[0]
        return question.get('question_table')

    def get_response_order(self, question_number, rename=None):
        """
        I may have done this before somewhere else. This is only for radio questions.
        """
        question_table = self.get_question_table(question_number)
        sorted_data = sorted(question_table, key=lambda x: int(x['value']))
        if rename:
            for key, value in rename.items():
                for i in sorted_data:
                    if i['description'] == key:
                        i['description'] = value
                        break
        return [s['description'] for s in sorted_data] + ['D', 'N', 'R', 'Missing', 'Total']

    @staticmethod
    def sort_index(table, order):
        sort_order = {v: i for i, v in enumerate(order)}

        table['_sort_key'] = table.index.map(lambda x: sort_order.get(x, 99))
        return table.sort_values('_sort_key').drop('_sort_key', axis=1).fillna('0 (0.00%)')
