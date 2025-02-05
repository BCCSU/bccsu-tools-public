import numpy as np
from bccsu.sas.commands import *
from bccsu.sas.tools.parse_tools import get_classes, rename_columns, clean_model_output
import pandas as pd
from bccsu.bccsu.tools import pretty_print_p_value
import warnings


def proc_logistic_parse(*args, **kwargs):
    results = proc_logistic_command(*args, **kwargs, outputs=['fitstatistics', 'parameterestimates', 'nobs',
                                                              'oddsratios', 'modelanova'], v=True)

    df = results['parameterestimates']

    df = df[df['Variable'] != 'Intercept'].copy().reset_index()
    df = df.join(results['oddsratios'].drop(columns=['Response'], errors='ignore'))

    nobs = results['nobs']['NObsUsed'][0]
    try:
        fit = (results['fitstatistics']
        .loc[results['fitstatistics']['Criterion'] == 'AIC', 'InterceptAndCovariates'].values[0])
    except AttributeError:
        fit = None
    classes = get_classes(*args, **kwargs)
    stat = df['WaldChiSq']
    df = rename_columns(df, args[1], 'Variable', 'OddsRatioEst', 'LowerCL', 'UpperCL', 'ProbChiSq', 'ClassVal0')

    df['WaldChiSq'] = stat

    if kwargs.get('link', '').lower() == 'glogit':
        type_3 = results['modelanova']['ProbChiSq'][0]
        df['Type 3 Estimate'] = type_3

    return df, nobs, fit, classes


def proc_phreg_parse(*args, **kwargs):
    results = proc_phreg_command(*args, **kwargs, outputs=['fitstatistics', 'parameterestimates', 'nobs'])
    df = results['parameterestimates']
    nobs = results['nobs']['NObsUsed'][0]
    fit = results['fitstatistics'].loc[results['fitstatistics']['Criterion'] == 'AIC', 'WithCovariates'].values[0]
    classes = get_classes(*args, **kwargs)

    df = rename_columns(df, args[1], 'Parameter', 'HazardRatio', 'HRLowerCL', 'HRUpperCL', 'ProbChiSq', 'ClassVal0')
    return df, nobs, fit, classes


def proc_glimmix_parse(*args, **kwargs):
    results = proc_glimmix_command(*args, **kwargs, outputs=['parameterestimates', 'nobs', 'fitstatistics'])
    df = results['parameterestimates']
    nobs = results['nobs']['NobsUsed'][0]
    try:
        fit = \
            results['fitstatistics'].loc[
                results['fitstatistics']['Descr'] == 'AIC  (smaller is better)', 'Value'].values[
                0]
    except AttributeError:
        fit = None
    df = df[df['Effect'] != 'Intercept'].copy()
    classes = get_classes(*args, **kwargs)

    df['Category'] = np.nan
    df['Category'] = df['Category'].astype(object)
    for key in classes:
        try:
            df['Category'] = df['Category'].fillna(df[key])
        except KeyError:
            pass

    link = kwargs.get('link')
    if link == 'identity':
        exponentiate = False
    else:
        exponentiate = True
    stat = df.reset_index()['tValue']
    df = rename_columns(df, args[1], 'Effect', 'Estimate', 'Lower', 'Upper', 'Probt', 'Category',
                        exponentiate=exponentiate)
    df['tValue'] = stat
    return df, nobs, fit, classes


def proc_genmod_parse(*args, **kwargs):
    if not kwargs.get('cross'):
        # Cross-sectional
        outputs = ['geeemppest', 'nobs', 'geefitcriteria']
        results = proc_genmod_command(*args, **kwargs, outputs=outputs)
        df = results['geeemppest']
        df = df[~df['Parm'].str.match(r'^Intercept\d*?$')].copy()
        nobs = results['nobs']['NObsUsed'][0]
        fit = results['geefitcriteria'].loc[results['geefitcriteria']['Criterion'] == 'QIC', 'Value'].values[0]
        classes = get_classes(*args, **kwargs)

        df = rename_columns(df, args[1], 'Parm', 'Estimate', 'LowerCL', 'UpperCL', 'ProbZ', 'Level1',
                            exponentiate=True)
    else:
        outputs = ['parameterestimates', 'nobs', 'modelfit']
        results = proc_genmod_command(*args, **kwargs, outputs=outputs)
        df = results['parameterestimates']
        fit = results['modelfit'][results['modelfit']['Criterion'].str.startswith('AIC ')]['Value'].values[0]
        df = df[~df['Parameter'].str.match(r'^Intercept\d*?$')].copy()
        df = df[df['Parameter'] != 'Scale'].copy()
        nobs = results['nobs']['NObsUsed'][0]
        classes = get_classes(*args, **kwargs)

        df = rename_columns(df, args[1], 'Parameter', 'Estimate', 'LowerLRCL', 'UpperLRCL', 'ProbChiSq', 'Level1',
                            exponentiate=True)

    return df, nobs, fit, classes


def vif_parse(*args, **kwargs):
    def remove_decimal(x):
        if x.dtype == 'float64':
            return x.astype(int).astype(str)
        else:
            return x

    df = args[0]
    classes = kwargs.get('classes')
    independent = args[1]
    if classes:
        independent = [i for i in independent if i not in list(classes.keys())]
    dummies = []

    if classes:
        drop_vars = [f'{key}_{item}' for key, item in classes.items()]
        dummies = pd.get_dummies(df[list(classes.keys())].apply(remove_decimal)).drop(columns=drop_vars)

    results = vif_command(df[independent].join(dummies), independent + list(dummies), *args[2:], **kwargs, outputs=[
        'parameterestimates'])['parameterestimates']
    results = results[results['Variable'] != 'Intercept'][['Variable', 'VarianceInflation']].set_index('Variable')

    return results.rename(columns={'VarianceInflation': 'VIF'})


def ancova_parse(*args, **kwargs):
    results = ancova_command(*args, **kwargs, outputs=['LSMeans', 'LSMeanCL', 'LSMeanDiffCL'])
    lsmeanvar = args[2]
    lsmeans = results['LSMeans'][[lsmeanvar, 'ProbtDiff']]
    mean_table = pd.merge(results['LSMeanCL'], lsmeans, how='left', left_on=[lsmeanvar], right_on=[lsmeanvar])
    mean_table = mean_table.rename(columns={'Dependent': 'Outcome',
                                            'Effect': 'Parameter',
                                            lsmeanvar: 'Category',
                                            'LowerCL': 'Lower',
                                            'UpperCL': 'Upper',
                                            'ProbtDiff': 'P-Value',
                                            'LSMean': 'Estimate'})
    mean_table = mean_table[['Outcome', 'Parameter', 'Category', 'Estimate', 'Lower', 'Upper', 'P-Value']]
    mean_table = mean_table.set_index(['Outcome', 'Parameter', 'Category'])
    return mean_table


def prevalence_ratio_parse(*args, **kwargs):
    results = prevalence_ratio_command(*args, **kwargs, outputs=['nobs', 'DIFFS', 'geefitcriteria'])
    nobs = results['nobs']['NObsUsed'][0]
    diffs = results['DIFFS']

    tables = []

    classes = get_classes(*args, **kwargs)

    for i in diffs['Effect'].unique():
        try:
            current_class = str(classes[i])
        except KeyError:
            current_class = 0

        diff_table = diffs[diffs[i].notna()]
        diff_table = diff_table[diff_table['_' + i] == current_class]
        diff_table['Category'] = diff_table[i]
        tables.append(diff_table)
    df = pd.concat(tables)

    fit = results['geefitcriteria'].loc[results['geefitcriteria']['Criterion'] == 'QIC', 'Value'].values[0]

    df = rename_columns(df, args[1], 'Effect', 'ExpEstimate', 'LowerExp', 'UpperExp', 'Probz', 'Category',
                        exponentiate=False)

    return df, nobs, fit, classes


def proc_reg_parse(*args, **kwargs):
    results = proc_reg_command(*args, **kwargs, outputs=['parameterestimates', 'fitstatistics', 'nobs'])
    df = results['parameterestimates']
    df = df[~df['Parameter'].str.match(r'^Intercept\d*?$')].copy()
    df['Category'] = df['Parameter'].str.extract(r'^.*? (.*?)$')
    df['Parameter'] = df['Parameter'].str.extract(r'^([\w]+)')

    nobs = results['nobs']['NObsUsed'][0]
    fit = results['fitstatistics']['RSquare'][0]
    classes = get_classes(*args, **kwargs)

    df = rename_columns(df, args[1], 'Parameter', 'Estimate', 'LowerCL', 'UpperCL', 'Probt', 'Category',
                        exponentiate=True)

    return df, nobs, fit, classes


def select_command_parse(*args, **kwargs):
    results = select_command(*args, **kwargs, outputs=['parameterestimates'])['parameterestimates']
    kept_variables = list(results['Effect'])
    kept_variables.remove('Intercept')
    return kept_variables


def proc_causaltrt_parse(*args, out_name='PREDICTION', **kwargs):
    df = args[0]
    treatment = args[2]
    if 'classes' in kwargs:
        classes = kwargs.get('classes')
        dummies = pd.get_dummies(df[classes.keys()], prefix_sep='_')
        df = df.join(dummies)
    else:
        classes = {}
    if 'category_labels' in kwargs:
        category_labels = kwargs.get('category_labels')
    else:
        category_labels = {}

    results = proc_causaltrt_command(df, *args[1:], **kwargs, outputs=['PSMODELESTIMATES', 'nobs', 'CAUSALEFFECTS'])
    est = results['PSMODELESTIMATES']
    mask = est['ChiSq'].notna()
    est = est.loc[mask].copy()

    beta = est['Estimate']
    std = est['StdErr']

    if 'Level1' in list(est):
        labels = (est['Parameter'] + '_' + est['Level1']).combine_first(
            est['Parameter']).to_list()  # Account for categorical names.
    else:
        est['Level1'] = np.nan
        labels = est['Parameter'].to_list()

    df['Intercept'] = 1
    x = np.array(df[labels])
    df[out_name] = np.matmul(x, beta)
    nobs = results['nobs']['NObsUsed'][0]
    tables_output = est.copy()[['Parameter', 'Level1']]
    tables_output['Estimate'] = beta
    tables_output['Lower'] = est['LowerWaldCL']
    tables_output['Upper'] = est['UpperWaldCL']
    tables_output['P-Value'] = est['ProbChiSq']
    tables_output = tables_output[tables_output['Parameter'] != 'Intercept']
    tables_output = rename_columns(tables_output, args[1], 'Parameter', 'Estimate', 'Lower', 'Upper', 'P-Value',
                                   'Level1')
    if 'classes' in kwargs:
        df = df.drop(columns=dummies)
    ce = results['CAUSALEFFECTS']
    ce = ce.rename(columns={'LowerWaldCL': 'Lower', 'UpperWaldCL': 'Upper', 'ProbZ': 'P-Value', 'Level': 'Category'})
    ce.loc[ce['Category'] == '1', 'N'] = df[df[treatment] == 1].shape[0]
    ce.loc[ce['Category'] == '0', 'N'] = df[df[treatment] == 0].shape[0]
    ce.loc[ce['Parameter'] == 'ATE', 'N'] = df[df[treatment].notna()].shape[0]

    ce['Outcome'] = 'DummyColumn'

    ce = clean_model_output(ce[['Outcome', 'Parameter', 'Category', 'Estimate', 'Lower', 'Upper', 'P-Value', 'N']],
                            {'Level': '0'}, None)
    ce = ce.reset_index().drop(columns=['Outcome'])
    ce = ce.rename(columns={'Category': 'Level'})
    ce = ce.set_index(['Parameter', 'Level'])

    tables_output = clean_model_output(tables_output, classes, category_labels, nobs)
    return df, tables_output, nobs, ce


def get_predictions_parse(*args, **kwargs, ):
    predicted = predictions_command(*args, **kwargs, get_predictions=True, outputs=['PREDICTIONS'])['PREDICTIONS']
    return predicted


def frequencies_parse(*args, **kwargs):
    """If this function hangs, don't get Fisher's exact test."""

    df, independent, dependent = args[:3]
    continuous_vars = kwargs.pop('continuous_vars', [])
    categorical_vars = [i for i in dependent if i not in continuous_vars]
    results = frequencies_command(df, independent, categorical_vars, continuous_vars, **kwargs,
                                  outputs=['crosstabfreqs', 'univariate_out', 'univariate_out_total',
                                           'chisq', 'wilcoxontest', 'onewayfreqs', 'fishersexact'])
    final = pd.DataFrame()

    if categorical_vars:
        freqs = results['crosstabfreqs'].copy()
        freqs['VarName'] = freqs['VarName'].str.lower()
        freqs['_TYPE_'] = freqs['_TYPE_'].astype(str).str.zfill(2)

        freqs['Value'] = freqs['Value'].astype(object)
        freqs['outcomeValue'] = freqs['outcomeValue'].astype(object)
        freqs.loc[freqs['outcomeValue'] == '.', 'outcomeValue'] = np.nan
        freqs.loc[freqs['Value'] == '.', 'Value'] = np.nan

        freqs.loc[~freqs['Value'].isna(), 'Value'] = freqs.loc[~freqs['Value'].isna(), 'Value'].astype(int).astype(str)
        freqs.loc[~freqs['outcomeValue'].isna(), 'outcomeValue'] = (
            freqs.loc[~freqs['outcomeValue'].isna(), 'outcomeValue']
            .astype(int).astype(str))

        def clean_freqs(row_type, percent_type):
            data = freqs[freqs['_TYPE_'].isin([row_type])].copy()
            if row_type == '11':
                data['outcomeValue'] = data['outcomeValue'].fillna('Missing')
            if row_type == '01':
                data['outcomeValue'] = data['outcomeValue'].fillna('Total')
            data['Value'] = data['Value'].fillna('Missing')

            strat_counts = pd.pivot(data, index=['VarName', 'Value'],
                                    columns='outcomeValue',
                                    values=['Frequency', percent_type])
            strat_counts['Frequency'] = strat_counts['Frequency'].astype(int).astype(str)
            for col in strat_counts['Frequency'].columns:
                strat_counts.loc[:, (percent_type, col)] = strat_counts[percent_type][col].apply(lambda x: f'{x:.2f}%')

            return strat_counts['Frequency'] + ' (' + strat_counts[percent_type] + ')'

        freqs_cleaned = (clean_freqs('01', 'Percent')
                         .join(clean_freqs('11', 'RowPercent')))
        chisq = results['chisq'].copy()
        chisq['VarName'] = chisq['VarName'].str.lower()
        chisq = chisq[chisq['Statistic'] == 'Chi-Square'].copy()
        top_rows = freqs_cleaned.groupby('VarName').head(1).copy()
        top_rows['P-Value'] = top_rows.join(chisq.set_index('VarName')['Prob'], on='VarName')['Prob']

        fish = results['fishersexact']
        fish = fish[fish['Label1'].isin(['Pr <= P', 'Two-sided Pr <= P'])].copy()
        top_rows['Fisher\'s Exact'] = top_rows.join(fish.set_index('VarName')['cValue1'], on='VarName')['cValue1']

        freqs_cleaned = freqs_cleaned.join(top_rows[['P-Value', 'Fisher\'s Exact']])
        freqs_cleaned['P-Value'] = freqs_cleaned['P-Value'].apply(pretty_print_p_value).str.replace('nan', '')

        final = pd.concat([final, freqs_cleaned])

    if continuous_vars:
        iqr = results['univariate_out'].copy()
        iqr['VarName'] = iqr['VarName'].str.lower()
        iqr['outcomeValue'] = iqr['outcomeValue'].astype(object)
        iqr.loc[~iqr['outcomeValue'].isna(), 'outcomeValue'] = (iqr.loc[(~iqr['outcomeValue'].isna()), 'outcomeValue']
                                                                .astype(int).astype(str))
        iqr_tot = results['univariate_out_total'].copy()
        iqr_tot['outcomeValue'] = 'Total'
        iqr = pd.concat([iqr_tot, iqr], ignore_index=True, sort=False)

        iqr['outcomeValue'] = iqr['outcomeValue'].fillna('Missing')
        strat_counts = pd.pivot(iqr, index=['VarName'],
                                columns='outcomeValue',
                                values=['median', 'q1', 'q3'])
        for col in strat_counts['median'].columns:
            strat_counts.loc[:, ('median', col)] = strat_counts['median'][col].apply(lambda x: f'{x:.2f}')
            strat_counts.loc[:, ('q1', col)] = strat_counts['q1'][col].apply(lambda x: f'{x:.2f}')
            strat_counts.loc[:, ('q3', col)] = strat_counts['q3'][col].apply(lambda x: f'{x:.2f}')
        iqr_cleaned = strat_counts['median'] + ' (' + strat_counts['q1'] + ' - ' + strat_counts['q3'] + ')'
        iqr_cleaned['Value'] = 'median (iqr)'
        iqr_cleaned.set_index('Value', append=True, inplace=True)

        if results.get('wilcoxontest') is not None:
            wilcoxin = results['wilcoxontest'].copy()
            wilcoxin.set_index(['VarName'], inplace=True)
            try:
                wilcoxin['P-Value'] = wilcoxin['Prob2'].apply(pretty_print_p_value)
            except KeyError:
                wilcoxin = wilcoxin[(wilcoxin['Label1'] == 'Two-Sided Pr > |Z|') & (wilcoxin['Name1'] == 'PT2_WIL')].copy()
                wilcoxin['P-Value'] = wilcoxin['cValue1'].apply(pretty_print_p_value)

            wilcoxin = wilcoxin[['P-Value']]
            iqr_cleaned = iqr_cleaned.join(wilcoxin)
        else:
            # create warning
            warnings.warn('Wilcoxon test not run.')
            iqr_cleaned['P-Value'] = ""

        final = pd.concat([final, iqr_cleaned])

    final.reset_index(inplace=True)
    final['VarName'] = pd.Categorical(final['VarName'], categories=dependent, ordered=True)
    final.sort_values(['VarName', 'Value'], inplace=True)
    labels = kwargs.get('labels', {})
    for key, label in labels.items():
        label_w_str_keys = {str(k): v for k, v in label.items()}
        final.loc[final['VarName'] == key, 'Value'] = (final.loc[final['VarName'] == key, 'Value']
                                                       .replace(label_w_str_keys))

    outcome_freqs = results['onewayfreqs'].copy()
    outcome_freqs[independent] = outcome_freqs[independent].astype(object)
    outcome_freqs.loc[outcome_freqs[independent] == '.', independent] = np.nan
    outcome_freqs.loc[~outcome_freqs[independent].isna(),
    independent] = (outcome_freqs.loc[(~outcome_freqs[independent].isna()), independent]
                    .astype(int).astype(str))
    outcome_freqs[independent] = outcome_freqs[independent].fillna('Missing')
    outcome_freqs['label_match'] = outcome_freqs[independent]
    if independent in labels:
        labels['label_match'] = {str(k): v for k, v in labels[independent].items()}
        outcome_freqs['label_match'] = outcome_freqs['label_match'].replace(labels['label_match'])
    outcome_freqs['Percent'] = outcome_freqs['Percent'].apply(lambda x: f'{x:.2f}')
    outcome_freqs['clean'] = outcome_freqs['label_match'] + ', N=' + outcome_freqs['Frequency'].astype('str') + ' (' + outcome_freqs['Percent'] + '%)'

    outcome_freqs.set_index(independent, inplace=True)
    new_columns = outcome_freqs['clean'].to_dict()

    final = final.rename(columns=new_columns)

    final.rename(columns={'VarName': 'Variable'}, inplace=True)
    final.set_index(['Variable', 'Value'], inplace=True)

    try:
        total_missing = outcome_freqs.loc['Missing']['Frequency'].sum()
    except KeyError:
        total_missing = 0
    total_n = outcome_freqs[outcome_freqs.index != 'Missing']['Frequency'].sum()
    final.title = f'Frequencies for {independent}, Total={total_n}, Missing={total_missing}'
    final.notes = (f"Pearson's Chi-squared test was used to test for differences between groups and "
                   f"Wilcoxon test was used to test for differences between groups for continuous variables.")

    return final
