import ast
import logging
import re

import numpy as np
import pandas as pd

from bccsu.bccsu.stats import freqs


_SAFE_AST_NODES = (
    ast.Expression, ast.BoolOp, ast.And, ast.Or,
    ast.Compare, ast.Eq, ast.NotEq, ast.Gt, ast.Lt, ast.GtE, ast.LtE,
    ast.BinOp, ast.BitAnd, ast.BitOr, ast.UnaryOp, ast.Invert, ast.Not,
    ast.Attribute, ast.Subscript, ast.Name, ast.Constant,
    ast.Load,
)


def _validate_restriction_ast(source):
    """Validate that a restriction expression only contains safe pandas comparison nodes."""
    tree = ast.parse(source, mode='eval')
    for node in ast.walk(tree):
        if not isinstance(node, _SAFE_AST_NODES):
            raise ValueError(f"Unsafe node in restriction expression: {type(node).__name__}")


class RedCap:
    def __init__(self, df, meta):
        self.meta = meta
        self.df = df.reset_index(drop=True)
        self.restriction = None
        self.text_recategorization_started = False

    def create_restriction_mask(self, key):
        s = self.meta.loc[key]['restrictions']

        if pd.isna(s):
            return np.ones(self.df.shape[0]).astype(bool)

        # Remove commented lines
        executing_lines = []
        for line in s.split('\n'):
            line = line.strip()
            if not line.startswith('#'):
                executing_lines.append(line)
        s = '\n'.join(executing_lines)

        s = s.replace('\t\t\t\t', '')
        s = s.replace('\n', ' ')

        # Temporary. Removes multipl [] in restrictions and leaves last most one.
        # [y2_qi_staff_survey_arm_6][role_y2]='6' -> [role_y2]='6'
        pattern = re.compile(r'(?:\[[0-9a-z_]+\])*(\[[0-9a-z_]+\]\s*=\s*(?:\'[^\']*\'|"[^"]*"|\d+))', re.IGNORECASE)
        s = pattern.sub(r'\1', s)

        def replacer(match):
            m = match.group(0)
            if '=' in m:
                m = m.replace(' = ', '=')
                m = m.replace(' =', '=')
                key, value = m.split('=')
                relation = '=='
            elif '<>' in m:
                key, value = m.split('<>')
                relation = '!='
            elif '>' in m:
                key, value = m.split('>')
                relation = '>'
            else:
                raise Exception('Relation not handled yet.')
            key = key.replace('(', '___')
            key = key.replace(')', '')
            if '"' not in value and "'" not in value:
                value = f"'{value}'"
            return f'(self.df[\'{key[1:-1]}\'] {relation} {value})'

        s = re.sub(r"\[(.+?)\]\s*(?:=|<>)\s*(?:(?:'([^']*)')|(?:\"([^\"]*)\")|(\d+))", replacer, s)

        s = s.replace(' and ', ' AND ')
        s = s.replace(' or ', ' OR ')

        s = s.replace(' AND ', '&')
        s = s.replace(' OR ', '|')
        _validate_restriction_ast(s)
        return eval(compile(s, '<string>', 'eval'))

    def create_missingness_mask(self, key):
        question_category = self.meta.loc[key]['question_category']
        if question_category == 'checkbox':
            question_name = self.meta.loc[key]['name']
            mask = ~(self.df[[f'{question_name}___{i}' for i in ['r', 'n', 'd']]] == '1').any(axis=1)
        else:
            mask = ~self.df[key].str.lower().isin(['r', 'n', 'd', None])
        return mask

    def get_classes(self, key):
        classes = {}
        for row in self.meta.loc[key]['question_table']:
            classes[row['value']] = row['description']
        return classes

    def _clean_categorical(self, key, mask, set_missing_na=False):
        array = self.df[key].copy()
        array[mask] = np.nan
        classes = self.get_classes(key)
        if not set_missing_na:
            classes.update({'R': 'Refused', 'N': 'Not Applicable', 'D': 'Don\'t Know'})
        array = pd.Series(pd.Categorical(array.map(classes),
                                         categories=list(classes.values()),
                                         ordered=True), name=key, index=array.index)
        return array

    def _clean_numeric(self, key, mask):
        # Check if numeric
        array = self.df[key].copy()
        array[mask] = np.nan
        is_numeric = self.check_numeric(key)
        if not is_numeric:
            raise Exception(f'Not numeric: {key}')
        array = pd.to_numeric(array, errors='coerce')
        return array

    def check_numeric(self, key):
        if self.meta.loc[key]['question_category'] == 'numeric':
            return True
        if self.meta.loc[key]['question_category'] != 'text':
            return False
        array = self.df[key].copy()
        if array.isna().all():
            return False
        array = array.str.replace(',', '').str.replace('.', '')
        array[array.isin(['R', 'D', 'N'])] = np.nan
        numeric = array.str.isnumeric().all()

        if numeric:
            # update key on the fly if tested true for numeric.
            self.meta.loc[key, 'question_category'] = 'numeric'
        return numeric

    def _clean_text(self, key, mask):
        numeric = self.check_numeric(key)
        if numeric:
            return self._clean_numeric(key, mask)
        array = self.df[key].copy()
        array[mask] = np.nan
        return array

    def _clean_date(self, key, mask):
        array = self.df[key].copy()
        array[array.isin(['R', 'D', 'N'])] = np.nan
        array = pd.to_datetime(array)
        array[mask] = np.nan
        return array

    def clean(self, keys, set_missing_na=False, force_numeric=False):
        """
        If we know it is numeric, we can force it to stop from guessing. Applies to all columns given so be careful.
        """

        if isinstance(keys, str):
            keys = [keys]

        arrays = []
        for key in keys:
            if force_numeric and self.meta.loc[key]['question_category'] != 'numeric':
                self.meta.loc[key, 'question_category'] = 'numeric'
            question_category = self.meta.loc[key]['question_category']
            restrictions_mask = self.create_restriction_mask(key)
            if set_missing_na:
                restrictions_mask &= self.create_missingness_mask(key)
            mask = ~restrictions_mask
            if question_category == 'categorical':
                arrays.append(self._clean_categorical(key, mask, set_missing_na=set_missing_na))
            elif question_category == 'numeric':
                arrays.append(self._clean_numeric(key, mask))
            elif question_category == 'date':
                arrays.append(self._clean_date(key, mask))
            elif question_category == 'text':  # todo need to check if numeric.
                arrays.append(self._clean_text(key, mask))
            elif pd.isna(question_category):
                raise Exception('Not yet implemented.')
            elif question_category == 'checkbox':
                check_box_vars = self.get_checkbox_variables(key)
                for check_box_var in check_box_vars:
                    arrays.append(self._clean_categorical(check_box_var, mask, set_missing_na=set_missing_na))
            else:
                raise Exception('Question type not recognized.')

        if len(arrays) > 1:
            df = pd.concat(arrays, axis=1)
        else:
            df = arrays[0]
        df.index = self.df.index

        return df

    def get_checkbox_variables(self, key):
        return [q['variable_name'] for q in self.meta.loc[key]['question_table']]

    def __getitem__(self, keys):
        if isinstance(keys, str):
            keys = [keys]
        if isinstance(keys, list):
            cols = []
            for key in keys:
                row = self.meta.loc[key]
                if row['question_category'] == 'checkbox':
                    cols += self.get_checkbox_variables(key)
                else:
                    cols += [key]

            result = self.df[cols]
        else:
            result = self.df[keys]
        return result

    def _pretty_counts(self, key, numeric=False, dropna=False, restriction=None, set_missing_na=False):
        c = self.clean(key, set_missing_na=set_missing_na)
        if restriction is not None:
            c = c[restriction]
        counts = c.value_counts()
        if numeric:
            return counts
        pct = c.value_counts(normalize=True).mul(100)
        pct.fillna(0, inplace=True)
        if not dropna:
            counts['Missing'] = c.isna().sum()
            if c.shape[0] == 0:
                pct['Missing'] = 0
            else:
                pct['Missing'] = (c.isna().sum() / c.shape[0] * 100)
        counts_array = counts.map(str) + pct.map(lambda x: f" ({x:.2f}%)")
        counts_array['Total'] = str(c.dropna().shape[0])
        counts_array.name = (f"{self.meta.loc[key].get('question_number')} - {key}: "
                             f"{self.meta.loc[key].get('description')}")
        return counts_array

    def get_description_text(self, key):
        return f"{self.meta.loc[key].get('question_number')} - {key}: {self.meta.loc[key].get('description')}"

    def _pretty_iqr(self, key, restriction=None):
        c = self.clean(key)
        if restriction is not None:
            c = c[restriction]

        result = pd.DataFrame([{'Mean': c.mean(),
                                'std': c.std(),
                                'Median': np.nan if c.isna().all() else c.median(),
                                '25 quartile': c.quantile(0.25),
                                '75 quartile': c.quantile(0.75),
                                'N (Numeric Non-Missing)': (~c.isna()).sum(),
                                'Missing': c.isna().sum()}],
                              index=[f"{self.meta.loc[key].get('question_number')} - {key}: "
                                     f"{self.meta.loc[key].get('description')}"])
        return result

    def pretty_counts(self, keys, dropna=False, numeric=False, restriction=None, set_missing_na=False):
        # todo if multiple checkbox columns. Stack side-by-side.
        if isinstance(keys, str):
            keys = [keys]
        question_category = self.meta.loc[keys[0]]['question_category']
        if question_category == 'text':
            if self.check_numeric(keys[0]):
                question_category = 'numeric'

        if question_category == 'numeric':
            iqrs = []
            for i, key in enumerate(keys):
                if self.meta.loc[key]['question_category'] != 'numeric':
                    if self.meta.loc[key]['question_category'] == 'text':
                        continuous = self.check_numeric(key)
                        if not continuous:
                            raise Exception('Classes need to match.')
                    else:
                        raise Exception('Classes need to match.')
                iqr_data = self._pretty_iqr(key, restriction=restriction)
                if restriction is not None:
                    raw_data = self.df.loc[restriction, key]
                else:
                    raw_data = self.df[key]
                if not set_missing_na:
                    for item in ['R', 'N', 'D']:
                        count = (raw_data == item).sum()
                        if count > 0:
                            iqr_data[item] = (raw_data == item).sum()

                # iqr_data.loc[:, ['Mean', 'std']] = iqr_data.loc[:, ['Mean', 'std']].round(2)
                # percentiles = ['Median', '25 quartile', '75 quartile']
                # iqr_data.loc[:, percentiles] = iqr_data.loc[:, percentiles].round(0)
                # for c in percentiles:
                #     n_at_percentile = (self.df[keys].round(0) == iqr_data.loc[:, c].values[0]).sum().values[0]
                #     iqr_data.loc[:, c] = f"{iqr_data.loc[:, c].values[0]:.0f} ({n_at_percentile})"
                # iqr_data.rename(columns={c: f"{c} (N)" for c in percentiles}, inplace=True)

                iqrs.append(iqr_data)
            return pd.concat(iqrs)

        yesno = {'1': 'Yes', '0': 'No'}
        classes = self.get_classes(keys[0])
        if question_category == 'checkbox':
            classes = yesno
        expanded_keys = []
        for i, key in enumerate(keys):
            if question_category == 'checkbox':
                if classes != yesno:
                    raise Exception('Classes need to match.')
                for check_var in self.get_checkbox_variables(key):
                    expanded_keys.append(check_var)
            else:
                if classes != self.get_classes(key):
                    raise Exception('Classes need to match.')
                expanded_keys.append(key)

        counts = []
        for key in expanded_keys:
            counts.append(
                self._pretty_counts(key, numeric=numeric, restriction=restriction, set_missing_na=set_missing_na))
        counts = pd.concat(counts, axis=1)

        if question_category != 'text':
            order = list(classes.values())
            if not set_missing_na:
                order += ['Refused', 'Not Applicable', 'Don\'t Know']
            order += ['Total']
            if not dropna:
                order += ['Missing']

            if len(order) < len(counts.index):
                raise Exception('Not all classes represented.')

            counts = counts.loc[order]
            if (counts.shape[0] < 5 and counts.shape[1] > 5 and not numeric):
                counts = counts.T

        # PI's only want to see these counts if they exist.
        empty = (counts[counts.index.isin(
            ['Refused', 'Not Applicable', 'Don\'t Know', 'Missing'])] == '0 (0.00%)').all(axis=1)
        empty = empty[empty]
        remove = empty.index.to_list()
        if len(remove) > 0:
            counts.drop(index=remove, inplace=True)
        return counts

    def comp(self, strat, columns, restriction=None, set_missing_na=False, mcnemar_test=False):
        if self.meta.loc[strat]['question_category'] != 'categorical':
            raise Exception('Stratification must be categorical.')

        if isinstance(columns, str):
            columns = [columns]

        df = self.clean([strat] + columns, set_missing_na=set_missing_na)
        if restriction is not None:
            df = df[restriction]
        classes = {}
        new_var_names = {}
        new_columns = []
        for i in [strat] + columns:
            if self.meta.loc[i]['question_category'] == 'checkbox':
                i = self.get_checkbox_variables(i)
            else:
                i = [i]
            for j in i:
                if self.meta.loc[j]['question_category'] != 'numeric' or not self.check_numeric(j):
                    curr_class = self.get_classes(j)
                    classes[j] = {int(key): item for key, item in curr_class.items()}
                    df[j] = df[j].astype(str).map({item: key for key, item in curr_class.items()})
                new_var_names[
                    j] = f"{self.meta.loc[j].get('question_number')} - {j}: {self.meta.loc[j].get('description')}"
                if j != strat:
                    new_columns.append(j)
        f = freqs(df, new_columns, strat=strat, labels=classes, mcnemar_test=mcnemar_test)
        f.index = f.index.set_levels([pd.Index([new_var_names.get(x) for x in level])
                                      if i == 0 else level
                                      for i, level in enumerate(f.index.levels)])
        f.title = f.title.replace(strat, new_var_names.get(strat))
        return f

    def collapse(self, key, new_name, collapse_dict):
        # todo allow for multiple keys if all are using the same collapse dict.
        if new_name in self.df.columns:  # todo only look at columns that are not custom.
            raise Exception('Name already exists.')
        if key not in self.meta.index:
            raise Exception('Key not found.')

        question_category = self.meta.loc[key]['question_category']
        if question_category == 'categorical':
            qt = self.meta.loc[key]['question_table']
            new_qt = []
            array = self.df[key].copy()
            array[:] = None
            desc = []

            for i, (dict_key, value) in enumerate(collapse_dict.items()):
                new_key = str(i + 1)
                new_qt.append({'value': new_key, 'description': dict_key})
                array[self.df[key].isin(value)] = new_key
                desc.append(f"{dict_key}: {key}={','.join(value)}")

            # Create a dedicated method for this.
            self.df[new_name] = array
            self.meta.loc[new_name] = self.meta.loc[key]
            self.meta.at[new_name, 'name'] = new_name
            self.meta.at[new_name, 'description'] = '; '.join(desc)
            self.meta.at[new_name, 'question_table'] = new_qt
            self.meta.at[new_name, 'question_category'] = 'categorical'
            self.meta.at[new_name, 'question_number'] = 'Derived'
            self.meta.at[new_name, 'source_variables'] = [key]
            return self.df[new_name]

        elif question_category == 'checkbox':
            qt = self.meta.loc[key]['question_table']
            yesno = [{'value': '1', 'description': 'Yes'}, {'value': '0', 'description': 'No'}]

            new_qt = []
            template_array = pd.Series([None] * self.df.shape[0], dtype="object")
            mask = self.df[[f"{key}___{q['value']}" for q in qt]].isin(['0', '1']).all(axis=1)
            template_array.index = mask.index
            template_array[mask] = '0'
            desc = []
            for i, (dict_key, value) in enumerate(collapse_dict.items()):
                new_key = str(i + 1)
                new_name_cat = f'{new_name}___{new_key}'
                new_qt.append({'value': new_key, 'variable_name': new_name_cat, 'description': dict_key})

                array = template_array.copy()

                columns = [f'{key}___{i}' for i in value]
                array[(self.df[columns] == '1').any(axis=1)] = '1'

                description = f"{dict_key}: {key}={','.join(value)}"

                self.meta.loc[new_name_cat] = self.meta.loc[key]
                self.meta.at[new_name_cat, 'name'] = new_name_cat
                self.meta.at[new_name_cat, 'description'] = description
                self.meta.at[new_name_cat, 'question_table'] = yesno
                self.meta.at[new_name_cat, 'question_category'] = 'categorical'
                self.meta.at[new_name_cat, 'question_number'] = 'Derived'
                self.meta.at[new_name_cat, 'source_variables'] = [key]
                self.df.loc[:, new_name_cat] = array
                desc.append(description)

            self.meta.loc[new_name] = self.meta.loc[key]
            self.meta.at[new_name, 'name'] = new_name
            self.meta.at[new_name, 'description'] = '; '.join(desc)
            self.meta.at[new_name, 'question_table'] = new_qt
            self.meta.at[new_name, 'question_category'] = 'checkbox'
            self.meta.at[new_name, 'question_number'] = 'Derived'
            self.meta.at[new_name, 'source_variables'] = [key]

            for value in ['n', 'd', 'r']:
                self.df[f'{new_name}___{value}'] = self.df[f'{key}___{value}']

            self.df = self.df.copy()

        else:
            raise Exception('Stratification must be categorical or checkbox.')

    def create(self, array, source_variables=None, **kwargs):
        if kwargs.get('name') is None:
            kwargs['name'] = array.name
        else:
            try:
                array.name = kwargs['name']
            except AttributeError:
                array = pd.array(array)
                array.name = kwargs['name']
        # todo add a notes section for each variable.
        if kwargs['name'] in self.df.columns:
            raise Exception('Name already exists.')
        # todo We need a number of checks here.

        kwargs['question_number'] = 'Derived'
        kwargs['source_variables'] = source_variables or []
        if isinstance(kwargs.get('question_table'), dict):
            kwargs['question_table'] = [{'value': str(key), 'description': value}
                                        for key, value in kwargs['question_table'].items()]
        assert self.df.shape[0] == array.shape[0]
        array.index = self.df.index
        self.df[array.name] = array
        self.meta.loc[array.name] = kwargs
        self.meta.at[array.name, 'name'] = array.name
        return self.df[array.name]

    def create_classes(self, varname, classes):
        meta = self.meta.loc[varname]
        qtype = meta['question_type']
        question_table = meta['question_table']
        values = [v['value'] for v in question_table]
        dup_values = [str(key) for key in classes.keys() if str(key) in values]
        first_in_q_table = question_table[0]

        if len(dup_values) > 0:
            raise Exception(f'Duplicate values found: {dup_values}')
        for key, value in classes.items():
            new_rec = {'value': str(key), 'description': value}

            if qtype == 'checkbox':
                vname = first_in_q_table['variable_name']
                new_rec['variable_name'] = f'{varname}___{key}'
                self.df[f'{varname}___{key}'] = '0'
                self.df.loc[self.df[vname].isna(), f'{varname}___{key}'] = np.nan
                self.meta.loc[f'{varname}___{key}'] = self.meta.loc[vname]
                self.meta.loc[f'{varname}___{key}', 'description'] = value

            question_table.append(new_rec)
        self.meta.at[varname, 'question_table'] = question_table

    @staticmethod
    def clean_text(item):
        if isinstance(item, str):
            return item.lower().strip().replace('"', "").replace("'", "")
        else:
            return item

    def distribute_other(self, varname, text_vars, distribution=None, ignore=None, clean=True):
        """
        99.9% of the time, we will want clean=True. I can't think of why we wouldn't but I left the option just in case.
        """
        if distribution is None:
            return
        qtype = self.meta.loc[varname, 'question_category']

        if clean:
            distribution_clean = {}
            if distribution:
                for key, item in distribution.items():
                    if key not in distribution_clean.keys():
                        distribution_clean[key] = []
                    for sub_item in item:
                        distribution_clean[key].append(self.clean_text(sub_item))
            distribution = distribution_clean

            text_vars = self.df[text_vars].map(self.clean_text)
            if ignore:
                ignore = [self.clean_text(item) for item in ignore]
        else:
            text_vars = self.df[text_vars]

        logged_items = {}
        if distribution:
            for key, item in distribution.items():
                for sub_item in item:
                    if sub_item not in logged_items.keys():
                        logged_items[sub_item] = {'text_value': sub_item,
                                                  'category': key,
                                                  'explicitly_defined': 'Yes'}
                    else:
                        logged_items[sub_item]['category'] = f'{logged_items[sub_item]["category"]}, {key}'

        if ignore:
            for item in ignore:
                if item not in logged_items.keys():
                    logged_items[item] = {'text_value': item,
                                          'category': 'Ignore',
                                          'explicitly_defined': 'Yes'}
                else:
                    logged_items[item]['category'] = f'{logged_items[item]["category"]}, Ignore'

        unhandled = text_vars[~text_vars.isin(list(logged_items.keys()))].dropna()

        for item in unhandled:
            logged_items[item] = {'text_value': item, 'category': '', 'explicitly_defined': "No"}

        if not self.text_recategorization_started:
            """
            Make a csv tracking all recategorizations, if file exists, overwrite and create new
            """
            self.text_recategorization_started = True
            method = 'w'
        else:
            method = 'a'

        with open('recategorization_log.csv', method) as f:
            f.write(f'{varname}\n')

        pd.DataFrame(list(logged_items.values())).to_csv(
            'recategorization_log.csv',
            mode="a",
            index=False,
        )

        with open('recategorization_log.csv', 'a') as f:
            f.write('\n')

        if unhandled.shape[0] > 0:
            logging.warning(f'Unhandled items found in "{varname}": \n{unhandled.to_list()}')
            # raise Exception(f'Unhandled items found: \n{unhandled.to_list()}')

        if qtype == 'checkbox':
            for key, value in distribution.items():
                values_lower = [v.strip().lower() for v in value]
                self.df.loc[text_vars.isin(values_lower), f'{varname}___{key}'] = '1'
        elif qtype == 'categorical':
            for key, value in distribution.items():
                values_lower = [v.strip().lower() for v in value]
                self.df.loc[text_vars.isin(values_lower), varname] = str(key)
        else:
            raise Exception('Variable must be checkbox or categorical.')

    def question_lookup(self, qn):
        if qn in self.meta.index:
            return [qn]
        else:
            return self.meta[((self.meta['question_number'] == qn) &
                              ~self.meta['checkbox'].astype(bool) &
                              (self.meta['question_type'] != 'descriptive'))]['name'].index.tolist()

    def search_variables(self, query, limit=20):
        """Search variable names and descriptions for a query string (case-insensitive)."""
        q = query.lower()
        mask = (self.meta['name'].str.lower().str.contains(q, na=False) |
                self.meta['description'].str.lower().str.contains(q, na=False))
        result = self.meta.loc[mask, ['name', 'description', 'question_category', 'instrument']].head(limit)
        return result

    def describe(self, key):
        """Return a structured dict with all metadata and basic stats for a variable."""
        row = self.meta.loc[key]
        info = {
            'name': key,
            'description': row.get('description'),
            'question_category': row.get('question_category'),
            'question_type': row.get('question_type'),
            'instrument': row.get('instrument'),
            'event': row.get('event'),
            'restrictions': row.get('restrictions'),
            'section': row.get('section'),
        }
        if row.get('question_category') in ('categorical', 'checkbox'):
            qt = row.get('question_table')
            if qt and isinstance(qt, list):
                info['categories'] = {item['value']: item.get('description', '') for item in qt}
        if row.get('question_category') == 'checkbox':
            info['sub_variables'] = self.get_checkbox_variables(key)
        if key in self.df.columns:
            col = self.df[key]
            info['n_rows'] = len(col)
            info['n_missing'] = col.isna().sum() + col.isin(['', None]).sum()
            info['sample_values'] = col.dropna().head(5).tolist()
        return info

    def list_variables(self, instrument=None, category=None, event=None):
        """List available variables with optional filtering by instrument, category, or event."""
        mask = pd.Series(True, index=self.meta.index)
        if instrument:
            mask &= self.meta['instrument'].str.lower().str.contains(instrument.lower(), na=False)
        if category:
            mask &= self.meta['question_category'] == category
        if event:
            mask &= self.meta['event'].apply(
                lambda x: event.lower() in [e.lower() for e in x] if isinstance(x, list) else False)
        return self.meta.loc[mask, ['name', 'description', 'question_category', 'instrument']]

    def summary(self):
        """Return a high-level overview of the dataset."""
        cat_counts = self.meta['question_category'].value_counts().to_dict()
        instruments = self.meta['instrument'].dropna().unique().tolist()
        events = []
        for e in self.meta['event'].dropna():
            if isinstance(e, list):
                events.extend(e)
        events = list(set(events))
        return {
            'n_rows': self.df.shape[0],
            'n_variables': len(self.meta),
            'instruments': instruments,
            'category_counts': cat_counts,
            'events': sorted(events),
        }

    def to_sas_df(self, variables, id_col='participant_id', classes=None):
        """Prepare a DataFrame for SAS procedures.

        Converts RedCap string-stored data to numeric types suitable for SAS,
        auto-generates labels and classes dicts from metadata.

        Args:
            variables: list of variable names (dependent + independent)
            id_col: subject identifier column name (becomes 'CODE' for SAS)
            classes: dict of {var: reference_value} overrides. If a categorical
                     variable is not in this dict, its first category value is
                     used as the reference.

        Returns:
            dict with keys:
                'df': numeric DataFrame with CODE column
                'labels': {var: {raw_value: description}} for categorical vars
                'classes': {var: reference_value} for categorical vars
        """
        if classes is None:
            classes = {}

        out_labels = {}
        out_classes = {}
        cols = {}

        for var in variables:
            cat = self.meta.loc[var]['question_category']
            qt = self.meta.loc[var].get('question_table')

            if cat == 'numeric':
                arr = self.df[var].copy()
                arr[arr.isin(['R', 'N', 'D'])] = np.nan
                cols[var] = pd.to_numeric(arr, errors='coerce')

            elif cat in ('categorical', 'checkbox'):
                arr = self.df[var].copy()
                arr[arr.isin(['R', 'N', 'D'])] = np.nan
                cols[var] = pd.to_numeric(arr, errors='coerce')

                if qt:
                    label_map = {}
                    for entry in qt:
                        val = entry.get('value', entry.get('variable_name', ''))
                        label_map[val] = entry['description']
                    out_labels[var] = label_map

                if var in classes:
                    out_classes[var] = classes[var]
                elif qt:
                    out_classes[var] = qt[0]['value']

            elif cat == 'text':
                arr = self.df[var].copy()
                arr[arr.isin(['R', 'N', 'D'])] = np.nan
                cols[var] = pd.to_numeric(arr, errors='coerce')

            else:
                arr = self.df[var].copy()
                arr[arr.isin(['R', 'N', 'D'])] = np.nan
                cols[var] = pd.to_numeric(arr, errors='coerce')

        df = pd.DataFrame(cols, index=self.df.index)
        if id_col in self.df.columns:
            df['CODE'] = pd.to_numeric(self.df[id_col], errors='coerce')

        return {
            'df': df,
            'labels': out_labels,
            'classes': out_classes,
        }


class R2R(RedCap):
    def __init__(self, df_raw, meta_dict):
        super().__init__(
            df_raw,
            meta_dict
        )
        dict_keys = list(self.meta.keys())
        self.meta['question_number'] = self.meta['adjusted_question_number']

    def __getitem__(self, keys):
        if isinstance(keys, str):
            keys = [keys]

        if isinstance(keys, list):
            for key in keys:
                if key not in self.meta.index:
                    columns = self.meta[self.meta['question_number'] == key].index.tolist()
                    if columns:
                        return self.df[columns]
                    raise Exception(f'Key not found: {key}')

        return super().__getitem__(keys)

    def pretty_counts(self, keys, dropna=False, numeric=False, restriction=None, set_missing_na=False):
        counts = []
        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            mask = restriction
            events = self.meta.loc[key, 'event']
            fu = False
            if isinstance(events, list):
                event = self.meta.loc[key, 'event'][0]
                fu = event.startswith('12_months')
            if fu:
                new_mask = (self.df['nature_of_study_completion___11'] == '1')
                if mask is not None:
                    mask = new_mask & mask
                else:
                    mask = new_mask
            counts.append(pd.DataFrame(super().pretty_counts(key, dropna=dropna, numeric=numeric, restriction=mask,
                                                             set_missing_na=set_missing_na)))
        cat = self.meta.loc[keys[0]]['question_category']
        if cat == 'numeric':
            return pd.concat(counts, axis=0)
        else:
            final_counts = pd.concat(counts, axis=1)
            rows = final_counts.index.to_list()
            optional_rows = ['Refused', 'Not Applicable', 'Don\'t Know', 'Missing', 'Total']
            new_order = [r for r in rows if r not in optional_rows]
            for i in optional_rows:
                if i in rows:
                    new_order.append(i)
            final_counts = final_counts.loc[new_order]
            final_counts.fillna('0 (0.00%)', inplace=True)
            return final_counts
