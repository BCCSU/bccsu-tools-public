"""Pure Python implementations of SAS statistical procedures.

Uses statsmodels to replicate SAS PROC LOGISTIC output format,
producing identical column structure so results can be used
interchangeably with the SAS-based functions.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2

from bccsu.sas.tools.parse_tools import clean_model_output


def _fit_logistic(df: pd.DataFrame, dependent: str, independent: list[str],
                  classes: dict[str, str] | None = None) -> dict[str, Any]:
    """Fit a logistic regression and return raw results matching SAS output structure.

    Args:
        df: numeric DataFrame (as produced by to_sas_df)
        dependent: outcome variable name
        independent: list of predictor variable names
        classes: {var: reference_value} for categorical predictors

    Returns:
        dict with 'df' (results DataFrame), 'nobs', 'fit' (AIC), 'classes'
    """
    if classes is None:
        classes = {}

    work = df[[dependent] + independent].dropna().copy()
    y = work[dependent]

    # SAS `descending` models P(Y = max_value). Binarize: max → 1, all else → 0.
    y_max = y.max()
    y = (y == y_max).astype(float)

    # Build design matrix with dummy encoding for class variables
    x_parts = []
    param_info = []  # (parameter_name, category_label_or_nan)

    for var in independent:
        if var in classes:
            ref = float(classes[var])
            levels = sorted(work[var].dropna().unique())
            for level in levels:
                if level == ref:
                    continue
                dummy = (work[var] == level).astype(float)
                col_name = f'{var}_{int(level)}'
                x_parts.append(dummy.rename(col_name))
                param_info.append((var, str(int(level))))
        else:
            x_parts.append(work[var])
            param_info.append((var, np.nan))

    X = pd.concat(x_parts, axis=1)
    X = sm.add_constant(X)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        try:
            model = sm.Logit(y, X)
            result = model.fit(disp=0, maxiter=100, method='newton')
        except Exception:
            # Fall back to bfgs if newton fails
            model = sm.Logit(y, X)
            result = model.fit(disp=0, maxiter=100, method='bfgs')

    # Extract parameter estimates (skip intercept)
    params = result.params[1:]
    conf = result.conf_int().iloc[1:]
    wald = result.tvalues[1:] ** 2  # Wald chi-square = z^2
    pvalues = result.pvalues[1:]

    # Build output matching SAS rename_columns format
    rows = []
    for i, (param_name, category) in enumerate(param_info):
        coef = params.iloc[i]
        odds_ratio = np.exp(coef)
        lower = np.exp(conf.iloc[i, 0])
        upper = np.exp(conf.iloc[i, 1])

        rows.append({
            'Outcome': dependent,
            'Parameter': param_name,
            'Category': category,
            'Estimate': odds_ratio,
            'Lower': lower,
            'Upper': upper,
            'P-Value': pvalues.iloc[i],
            'WaldChiSq': wald.iloc[i],
        })

    out = pd.DataFrame(rows)
    nobs = int(result.nobs)
    fit = result.aic

    return {'df': out, 'nobs': nobs, 'fit': fit, 'classes': classes}


def proc_logistic_python(df: pd.DataFrame, dependent: str, independent: list[str],
                         classes: dict[str, str] | None = None,
                         labels: dict[str, dict] | None = None,
                         **kwargs) -> pd.DataFrame:
    """Python replacement for proc_logistic. Fits a multivariate logistic regression.

    Returns DataFrame with identical structure to the SAS version:
    Index: [Outcome, Parameter, Category]
    Columns: Estimate, Lower, Upper, P-Value, WaldChiSq, N
    """
    result = _fit_logistic(df, dependent, independent, classes)
    out = result['df']
    out['N'] = ''
    out.iloc[0, out.columns.get_loc('N')] = result['nobs']

    return clean_model_output(out, result['classes'], labels, result['nobs'])


def bivar_logistic_python(df: pd.DataFrame, dependent: str, independent: list[str],
                          classes: dict[str, str] | None = None,
                          labels: dict[str, dict] | None = None,
                          **kwargs) -> pd.DataFrame:
    """Python replacement for bivar_logistic. Runs separate univariate logistic regressions.

    Returns DataFrame with identical structure to the SAS version:
    Index: [Outcome, Parameter, Category]
    Columns: Estimate, Lower, Upper, P-Value, WaldChiSq, N, P-Value < .1
    """
    if classes is None:
        classes = {}

    outputs = []
    for variable in independent:
        class_val = {}
        if variable in classes:
            class_val[variable] = classes[variable]
        outcome_class = classes.get(dependent)
        if outcome_class:
            class_val[dependent] = outcome_class

        try:
            result = _fit_logistic(df, dependent, [variable], class_val)
            row = result['df']
            row['N'] = result['nobs']
            outputs.append(row)
        except Exception:
            warnings.warn(f'{variable} - Could not be estimated.')

    out = pd.concat(outputs, ignore_index=True)
    out['P-Value'] = pd.to_numeric(out['P-Value'], errors='coerce')
    out['P-Value < .1'] = np.where(out['P-Value'] < 0.1, 'x', '')

    return clean_model_output(out, classes, labels)