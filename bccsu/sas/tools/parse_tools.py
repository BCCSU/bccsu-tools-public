import numpy as np
from bccsu.bccsu.tools import pretty_print_p_value
import pandas as pd


def prettify(df, index, round_col, pval_col):
    df = df.drop(columns=['varNum', '_BREAK_', 'outcomeNum'], errors='ignore').fillna('')
    df = df.set_index(index)

    def clean_x(x):
        if x == "":
            return ""
        elif float(x) > 1000:
            return 'Infinity'
        else:
            return f'{float(x):.2f}'

    df[round_col] = df[round_col].map(clean_x)
    df[pval_col] = df[pval_col].map(lambda x: pretty_print_p_value(x) if x != "" else x)
    return df


def get_classes(*args, **kwargs):
    classes = kwargs.get('classes')
    if not classes:
        classes = {}
    return classes


def rename_columns(df, outcome, parameter, estimate, lower, upper, pvalue, category, exponentiate=False):
    out = df[[]].copy()

    out['Outcome'] = outcome
    try:
        out['Category'] = df[category]
    except KeyError:
        out['Category'] = np.nan
        out['Category'] = out['Category'].astype(object)

    out[['Parameter', 'Estimate', 'Lower', 'Upper', 'P-Value']] = df[[parameter, estimate, lower, upper, pvalue]]

    if exponentiate:
        for col in ['Estimate', 'Lower', 'Upper']:
            out[col] = pd.to_numeric(out[col], errors='coerce')
            out[col] = np.where(out[col].notna(), np.exp(out[col]), out[col])

    return out[['Outcome', 'Parameter', 'Category', 'Estimate', 'Lower', 'Upper', 'P-Value']].reset_index(drop=True)


def add_labels(df, classes, labels):
    indices = ['Outcome', 'Parameter', 'Category']

    df_clean = df.copy()
    df_clean['Category'] = df_clean['Category'].astype(object)
    try:
        mask = ~df_clean['Category'].isna()
        df_clean.loc[mask, 'Category'] = df_clean[mask]['Category'].astype(int)
    except ValueError:
        pass
    df_clean['Category'] = df_clean['Category'].astype(str)

    df_clean['Parameter'] = df_clean['Parameter'].str.lower()

    labels = {key.lower(): value for key, value in labels.items()}

    if not classes:
        indices = ['Outcome', 'Parameter']
    else:
        if labels:
            for label in labels:
                if label in df_clean['Parameter'].unique():
                    labels_str = {str(key): value for key, value in labels[label].items()}
                    df_clean.loc[df_clean['Parameter'] == label, 'Category'] = df_clean.loc[df_clean['Parameter'] == label, 'Category'].map(labels_str)
    df_clean.loc[df['Category'].isna(), 'Category'] = ''
    return df_clean, indices


def clean_model_output(df, classes, labels, nobs=None):
    df, indices = add_labels(df, classes, labels)

    df = prettify(df, indices, ['Estimate', 'Lower', 'Upper'], ['P-Value'])

    if 'N' not in list(df):
        df['N'] = ''
        df.iloc[0, df.columns.get_loc('N')] = nobs
    return df


def clean_model_wrapper(function):
    def wrapper(*args, **kwargs):
        df, nobs, classes = function(*args, **kwargs)
        try:
            labels = kwargs.get('labels')
        except KeyError:
            labels = []
        return clean_model_output(df, classes, labels, nobs)
    return wrapper

