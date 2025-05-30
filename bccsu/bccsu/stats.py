import pandas as pd
import numpy as np


def get_iqr(df, col=None):
    if col is None and isinstance(df, pd.core.series.Series):
        col = df.name
        df = pd.DataFrame(df)
    iqr = pd.DataFrame(df[col].quantile([.25, .5, .75]))[col]
    return f'{iqr[.5]:0.1f} ({iqr[.25]:0.1f} - {iqr[.75]:0.1f})'


def counts(df, cols=None):
    is_array = False
    if isinstance(df, pd.core.series.Series):
        cols = [df.name]
        df = pd.DataFrame(df)
        is_array = True

    def round_perc(col):
        series = df[col]
        if len(df[i].unique()) > 3:
            print(len(df[i].unique()))
            yes_col = get_iqr(df, i)
            no_col = ''
            total = df[i].notna().sum()
        else:
            yes_count = (series == 1).sum()
            no_count = (series == 0).sum()
            total = yes_count + no_count
            yes_col = f'{yes_count} ({yes_count / total * 100:.1f}%)'
            no_col = f'{no_count} ({no_count / total * 100:.1f}%)'

        values = [{
                      'Yes': yes_col,
                      'No': no_col,
                      'Total': total,
                      'Missing': series.isna().sum()
                      }]

        return pd.DataFrame(values, index=[col])

    rows = []
    for i in cols:
        rows.append(round_perc(i))

    out = pd.concat(rows)

    if is_array:
        return out.T
    else:
        return out


def value_counts_single(arr, labels=None):
    arr = arr.copy()
    arr[arr == 'nan'] = np.nan
    pref = arr.value_counts()
    if labels:
        pref.index = pref.index.map(labels)
    total = pref.sum()
    pref_count = pref.map(lambda x: f'{x} ({x / total * 100:0.2f}%)')
    if labels:
        pref_count.index = pd.Categorical(pref_count.index, categories=list(labels.values()), ordered=True)
    pref_count = pref_count.sort_index()
    pref_count.loc['Total'] = total
    if arr.shape[0] == 0:
        denominator = 1
    else:
        denominator = arr.shape[0]
    pref_count.loc['Missing'] = f'{arr.isna().sum()} ({arr.isna().sum() / denominator * 100:0.2f}%)'
    return pd.DataFrame(pref_count)


def text_counts(arr):
    return pd.DataFrame(arr.value_counts())


def set_int(df):
    if isinstance(df.index, pd.MultiIndex):
        for i, level in enumerate(df.index.levels):
            if level.dtype == 'float64':
                df.index = df.index.set_levels(level.astype(int, copy=False), level=i)
    else:
        if df.index.dtype == 'float64':
            df.index = df.index.astype(int)

    if isinstance(df.columns, pd.MultiIndex):
        for i, level in enumerate(df.columns.levels):
            if level.dtype == 'float64':
                df.columns = df.columns.set_levels(level.astype(int, copy=False), level=i)
    else:
        if df.columns.dtype == 'float64':
            df.columns = df.columns.astype(int)


def n_perc(n):
    def inner(x):
        return f'{x:.0f} ({x / n * 100:.2f}%)'

    return inner


def _cross_tab_single(df, rows, col, row_totals=True, col_totals=True):
    if isinstance(rows, str):
        rows = [rows]
    t_counts = df.value_counts(rows + [col]).unstack().fillna(0)
    set_int(t_counts)

    final_sum = None

    totals = t_counts.copy()

    t_counts = t_counts.map(n_perc(t_counts.values.sum()))

    if row_totals:
        if isinstance(t_counts.index, pd.MultiIndex):
            indices = tuple(['Total'] + ['' for i in t_counts.index.names[1:]])
        else:
            indices = 'Total'

        total_row_raw = totals.sum(axis=0)
        total_row_sum = total_row_raw.sum()
        total_row = total_row_raw.map(n_perc(total_row_sum))
        total_row.name = 'Total'
        t_counts.loc[indices, :] = total_row
        final_sum = total_row_sum

    if col_totals:
        total_col_raw = totals.sum(axis=1)
        total_col_sum = total_col_raw.sum()
        total_col = total_col_raw.map(n_perc(total_col_sum))

        t_counts['Total'] = total_col
        final_sum = total_col_sum

    try:
        # Just temporary while we test this new function.
        assert total_row_sum == total_col_sum
    except NameError:
        pass

    t_counts.columns = pd.MultiIndex.from_tuples(zip([t_counts.columns.name for i in t_counts], list(t_counts)))

    if final_sum > 0:
        t_counts = t_counts.fillna(f'{final_sum} ({100:.2f}%)')

    return t_counts


def cross_tab(df, rows, cols, **kwargs):
    # Use transpose in some cases. It makes things look nicer.
    if isinstance(cols, str):
        cols = [cols]
    table = None
    for num, i in enumerate(cols):
        if num == 0:
            table = _cross_tab_single(df, rows, i, **kwargs)
        else:
            table = pd.merge(table, _cross_tab_single(df, rows, i, **kwargs), how='outer', left_index=True,
                             right_index=True)
    return table


def strat_table(df, strat, cols):
    """
    Creates a stratified table.
    :param df:
    :param strat:
    :param cols:
    :return:
    """
    return cross_tab(df, [strat], cols).T


def iqr_strat(df, strat, cols):
    """
    Creates a table of stratified continuous vars showing iqr.
    :param df:
    :param strat:
    :param cols:
    :return:
    """

    vals = np.sort(df[strat].dropna().unique())

    roundit = False
    try:
        if (vals == vals.round()).all():
            roundit = True
    except TypeError:
        pass

    table = []
    index = []
    for col in cols:
        strats = {'Total': get_iqr(df, col)}
        for i in vals:
            if roundit:
                label = round(i)
            else:
                label = i
            strats[label] = get_iqr(df[df[strat] == i], col)
        table.append(strats)
        index.append(col)
    out = pd.DataFrame(table, index=index)
    out.columns.name = strat


    return out


def freqs(df, cols, strat=None, labels=None, continuous_vars=None, classes=None):
    # get order
    # split variables cat and cont

    stratify = True
    if strat is None:
        df = df[cols].copy()
        stratify = False
        # temp fix for now.
        strat = 'Category'
        df[strat] = [i % 2 for i in range(df.shape[0])]
    else:
        df = df[cols + [strat]].copy()

    strat_tables = []
    if labels:
        cat_vars = [col for col in cols if col in labels.keys()]
        cont_vars = [col for col in cols if col not in cat_vars]
        cat_strat = strat_table(df, strat, cat_vars).reset_index()

        strat_tables.append(cat_strat)
    else:
        cont_vars = cols

    cont_strat = iqr_strat(df, strat, cont_vars).reset_index()
    cont_strat = cont_strat.rename(columns={'index': 'level_0'})
    cont_strat['level_1'] = ''

    strat_tables.append(cont_strat)

    table = pd.concat(strat_tables)

    table['level_0'] = pd.Categorical(table['level_0'], categories=cols, ordered=True)
    table = table.sort_values(['level_0', 'level_1'])

    if labels is not None:
        for key, variable in table.groupby('level_0', observed=False):
            if labels.get(key):
                table.loc[(table['level_0'] == key) & (table['level_1'] != 'Total'), 'level_1'] = variable['level_1'].map(labels.get(key)).fillna('')

    out = table.set_index(['level_0', 'level_1'])

    if labels:
        if labels.get(strat):
            out = out.rename(columns=labels.get(strat))

    out.index.names = [None, None]

    if not stratify:
        out = out[['Total']]
    else:
        n_missing = df[strat].isna().sum()
        out.title = f'{strat} n={df.shape[0] - n_missing} (missing {n_missing})'

    return out
