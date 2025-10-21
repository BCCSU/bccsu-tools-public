from bccsu.sas.tools.parse_tools import get_classes, clean_model_output, clean_model_wrapper
from bccsu.sas.exceptions import InconsistencyError
import pandas as pd
import numpy as np
import warnings


def bivar(function):
    def wrapper(*args, **kwargs):
        outputs = []
        try:
            classes = kwargs.pop('classes')
        except KeyError:
            classes = {}
        labels = kwargs.get('labels')
        variables = args[2]

        prefix = kwargs.get('prefix')

        if not kwargs.get('show_sas_code'):
            kwargs['show_sas_code'] = False
        for variable in variables:
            class_var = classes.get(variable)
            if class_var is not None:
                class_val = {variable: class_var}
            else:
                class_val = {}
            outcome_class = classes.get(args[1])
            if outcome_class:
                class_val[args[1]] = outcome_class
            if kwargs.get('table_key'):
                table_key = kwargs['table_key'][variable]
                table_prefix = prefix
                table_prefix += f'_{table_key}'
                kwargs['prefix'] = table_prefix
            else:
                table_prefix = prefix
            try:
                row, nobs, _, _ = function(args[0], args[1], [variable], classes=class_val, **kwargs)
                row['N'] = nobs
                outputs.append(row)
            except TypeError:
                warnings.warn(f'{variable} - Couldn\'t be estimated.')

        df = pd.concat(outputs)
        # if pvalue is of type string
        df['P-Value'] = df['P-Value'].astype(str)
        df['P-Value'] = np.where(df['P-Value'] == '<.0001', '0.00001',
                                 np.where(df['P-Value'] == '.', np.nan, df['P-Value'])).astype(float)
        df['P-Value < .1'] = np.where(df['P-Value'] < .1, 'x', '')

        return clean_model_output(df, classes, labels)
    return wrapper


def explanatory(function):
    def wrapper(*args, **kwargs):
        df = args[0]
        dependent = args[1]
        independent = args[2]
        try:
            classes = kwargs.pop('classes')
        except KeyError:
            classes = {}
        labels = kwargs.get('labels')

        if not kwargs.get('show_sas_code'):
            kwargs['show_sas_code'] = False

        df = df[~df.isna()[[dependent] + independent].any(axis=1)].copy()
        nobs = df.shape[0]
        dropped_parm = None
        var_set = independent.copy()
        class_set = classes.copy()
        fit = np.inf
        final_df = None

        runs = []
        for i in range(len(independent)):
            if dropped_parm:
                var_set.remove(dropped_parm)
                try:
                    del class_set[dropped_parm]
                except KeyError:
                    pass

            prev_fit = fit

            build_df, _, fit, classes = function(df, dependent, var_set, classes=class_set, **kwargs)

            if _ != nobs:
                warnings.warn(f'Base model has {nobs} observations. Current submodel has {_}')

            if fit < prev_fit:
                final_df = build_df

            runs.append({'Iteration': i, 'Dropped Parameter': dropped_parm, 'Fit Statistic': fit})

            drop_index = build_df['P-Value'].argmax()
            dropped_parm = build_df.loc[drop_index, 'Parameter']

        runs = pd.DataFrame(runs).set_index('Iteration')

        return clean_model_output(final_df, class_set, labels, nobs), runs
    return wrapper


def confounding(inverse_exponent=False):
    '''
    For categorical variables, the maximum of the deltas in that category is used.
    :param inverse_exponent:
    :return:
    '''

    def innerfunc(function):
        def exclude(ls, item):
            l2 = ls.copy()
            l2.remove(item)
            return l2

        def wrapper(*args, **kwargs):
            df = args[0]
            dependent = args[1]
            independent = args[2]
            main_independent = args[3]
            try:
                classes = kwargs.pop('classes')
            except KeyError:
                classes = {}

            if not kwargs.get('show_sas_code'):
                kwargs['show_sas_code'] = False

            labels = kwargs.get('labels')

            df = df[~df.isna()[[dependent] + independent].any(axis=1)].copy()
            nobs = df.shape[0]
            dropped_parm = None
            var_set = independent.copy()
            var_set.remove(main_independent)
            class_set = classes.copy()
            final_df = None
            steps = {}
            for i in range(len(independent)):
                if dropped_parm:
                    var_set.remove(dropped_parm)
                    try:
                        del class_set[dropped_parm]
                    except KeyError:
                        pass

                model_vars = var_set + [main_independent]
                full_df, nobs, fit, classes = function(df, dependent, model_vars, classes=class_set, **kwargs)

                estimates = []
                for variable in var_set:
                    tmp_class_set = class_set.copy()
                    try:
                        del tmp_class_set[variable]
                    except KeyError:
                        pass

                    sub_df, nobs, fit, classes = function(df, dependent, exclude(model_vars, variable),
                                                          classes=tmp_class_set, **kwargs)
                    estimate = sub_df.loc[sub_df['Parameter'] == main_independent, ['Parameter', 'Category', 'Estimate']]
                    estimate['Dropped Var'] = variable
                    estimates.append(estimate)

                if len(estimates) > 0:
                    estimates_df = pd.concat(estimates)

                full_comparison = pd.merge(estimates_df,
                                           full_df[['Parameter', 'Category', 'Estimate']], how='left',
                                           left_on=['Parameter', 'Category'], right_on=['Parameter', 'Category'])

                if inverse_exponent:
                    full_comparison[['Estimate_x', 'Estimate_y']] = np.log(full_comparison[['Estimate_x', 'Estimate_y']])

                full_comparison['Absolute Delta'] = abs(100 * (
                            full_comparison['Estimate_x'] / full_comparison['Estimate_y'] - 1))

                custom_function = kwargs.get('custom_function')
                if custom_function:
                    dropped_parm, min_delta = custom_function(full_comparison)
                else:
                    maxes = full_comparison.groupby('Dropped Var')['Absolute Delta'].transform('max')
                    min_delta = maxes.min()
                    dropped_parm = full_comparison.iloc[maxes.argmin(), full_comparison.columns.get_loc('Dropped Var')]

                steps[dropped_parm] = full_comparison

                if kwargs.get('show_steps'):
                    print(f'Dropped: {dropped_parm}, Absolute Delta: {min_delta}')

                if min_delta > 5 or i == len(independent) - 1:
                    print(f'Absolute Delta: {min_delta}')
                    final_df = full_df
                    break

            return clean_model_output(final_df, classes, labels, nobs), steps
        return wrapper
    return innerfunc
