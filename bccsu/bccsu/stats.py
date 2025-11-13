import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact
from scipy.stats import ttest_ind, levene

def assoc_test_auto(x, y):
    """
    Decide between Fisher's exact (2x2 with small expected) and chi-square otherwise.
    Returns NaN if a category total is zero (row/column sum == 0).

    Parameters
    ----------
    x, y : array-like (categorical)

    Returns
    -------
    dict with keys:
      - test: 'fisher' or 'chi2' (or None if NaN)
      - stat: test statistic (odds ratio for fisher; chi2 for chi-square)
      - p_value: p-value (np.nan if not estimable)
      - dof: degrees of freedom (chi-square only)
      - expected: expected counts (chi-square only)
      - observed: observed contingency table (numpy array)
      - odds_ratio: for 2x2; NaN if any cell is zero
    """
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if df.empty:
        return {"test": None, "stat": np.nan, "p_value": np.nan,
                "dof": None, "expected": None, "observed": np.array([[]]),
                "odds_ratio": np.nan}

    ct = pd.crosstab(df["x"], df["y"])
    observed = ct.values

    # If any row or column sum is zero, not estimable
    if (observed.sum(axis=1) == 0).any() or (observed.sum(axis=0) == 0).any() or len(observed) == 1:
        return {"test": None, "stat": np.nan, "p_value": np.nan,
                "dof": None, "expected": None, "observed": observed,
                "odds_ratio": np.nan}

    r, c = observed.shape

    # Chi-square (to get expected counts and for non-2x2 cases)
    chi2, p_chi2, dof, expected = chi2_contingency(observed, correction=False)

    if r == 2 and c == 2:
        a, b, c2, d = observed.ravel()

        # Odds ratio is undefined if any cell is zero -> NaN per your requirement
        odds_ratio = np.nan if 0 in (a, b, c2, d) else (a * d) / (b * c2)

        # Use Fisher if any expected cell < 5, else chi-square
        if (expected < 5).any():
            odds, p = fisher_exact(observed, alternative="two-sided")
            return {"test": "fisher", "stat": odds, "p_value": p,
                    "dof": None, "expected": None, "observed": observed,
                    "odds_ratio": odds_ratio}
        else:
            return {"test": "chi2", "stat": chi2, "p_value": p_chi2,
                    "dof": dof, "expected": expected, "observed": observed,
                    "odds_ratio": odds_ratio}

    # Larger than 2x2 -> chi-square
    return {"test": "chi2", "stat": chi2, "p_value": p_chi2,
            "dof": dof, "expected": expected, "observed": observed,
            "odds_ratio": np.nan}


def t_test_auto(x, group, alternative="two-sided", use_levene=True):
    """
    Two-sample t-test for a continuous variable across exactly two groups.

    Parameters
    ----------
    x : array-like (continuous)
    group : array-like (categorical/indicator)
    alternative : {'two-sided', 'less', 'greater'}
    use_levene : bool, default True
        If True, perform Levene’s test to choose equal_var. If Levene p<0.05 -> Welch (equal_var=False).

    Returns
    -------
    dict with keys:
      - test: 't'
      - stat: t statistic (np.nan if not estimable)
      - p_value: p-value (np.nan if not estimable)
      - dof: degrees of freedom (Welch-Satterthwaite if unequal variances; else n1+n2-2)
      - n1, n2: sample sizes used
      - equal_var: bool or None
      - groups: the sorted unique group labels used
    """
    df = pd.DataFrame({"x": x, "g": group}).dropna()
    if df.empty:
        return {"test": None, "stat": np.nan, "p_value": np.nan,
                "dof": None, "n1": 0, "n2": 0, "equal_var": None, "groups": []}

    groups = df["g"].unique()
    if len(groups) != 2:
        # Only defined for two groups
        return {"test": None, "stat": np.nan, "p_value": np.nan,
                "dof": None, "n1": 0, "n2": 0, "equal_var": None, "groups": list(np.sort(groups))}

    g_sorted = np.sort(groups)
    a = df.loc[df["g"] == g_sorted[0], "x"].astype(float).to_numpy()
    b = df.loc[df["g"] == g_sorted[1], "x"].astype(float).to_numpy()

    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    n1, n2 = len(a), len(b)

    if n1 < 2 or n2 < 2:
        return {"test": None, "stat": np.nan, "p_value": np.nan,
                "dof": None, "n1": n1, "n2": n2, "equal_var": None, "groups": list(g_sorted)}

    equal_var = False  # default to Welch for robustness
    if use_levene:
        try:
            _, p_var = levene(a, b, center="median")
            equal_var = p_var >= 0.05
        except Exception:
            # Fall back to Welch if Levene fails
            equal_var = False

    res = ttest_ind(a, b, equal_var=equal_var, alternative=alternative)
    t_stat = float(res.statistic)
    p_val = float(res.pvalue)

    if equal_var:
        dof = n1 + n2 - 2
    else:
        # Welch-Satterthwaite approximation
        va = np.var(a, ddof=1)
        vb = np.var(b, ddof=1)
        num = (va / n1 + vb / n2) ** 2
        den = (va**2) / (n1**2 * (n1 - 1)) + (vb**2) / (n2**2 * (n2 - 1))
        dof = float(num / den) if den > 0 else np.nan

    return {"test": "t", "stat": t_stat, "p_value": p_val,
            "dof": dof, "n1": n1, "n2": n2, "equal_var": equal_var, "groups": list(g_sorted)}


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


def freqs(df, cols, strat=None, labels=None):
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
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    strat_tables = []
    tests = []
    if labels:
        cat_vars = [col for col in cols if col in labels.keys()]
        cont_vars = [col for col in cols if col not in cat_vars]
        if cat_vars:
            cat_strat = strat_table(df, strat, cat_vars).reset_index()
            strat_tables.append(cat_strat)

        for c in cat_vars:
            result = assoc_test_auto(df[c], df[strat])
            tests.append({'level_0': c, 'P Value': result['p_value'],
                           'Test Type': result['test']})
    else:
        cont_vars = cols

    cont_strat = iqr_strat(df, strat, cont_vars).reset_index()
    cont_strat = cont_strat.rename(columns={'index': 'level_0'})
    cont_strat['level_1'] = ''
    for c in cont_vars:
        # t-test for continuous vars across strat groups (only if exactly two groups with data)
        t_res = t_test_auto(df[c], df[strat])
        tests.append({'level_0': c,
                      'P Value': t_res['p_value'],
                      'Test Type': ('student_t' if t_res.get('equal_var') else 'welch_t') if t_res.get('test') == 't' else np.nan})
    tests = pd.DataFrame(tests)
    tests['group'] = 1

    strat_tables.append(cont_strat)

    table = pd.concat(strat_tables)
    table['group'] = 1
    table['group'] = table.groupby('level_0')['group'].cumsum()

    table = pd.merge(table, tests, how='left', on=['level_0', 'group'])

    table['level_0'] = pd.Categorical(table['level_0'], categories=cols, ordered=True)
    table = table.sort_values(['level_0', 'level_1'])
    table.drop(columns=['group'], inplace=True)

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
