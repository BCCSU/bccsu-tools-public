from bccsu.sas import proc_logistic
import pandas as pd
import numpy as np


def frequencies(df, strat, ind_vars, classes=None, category_labels=None, cut_off_continuous=5,
                columns_100_percent=False,
                legacy_mode=False,
                baseline=True,
                show_sas_code=False):

    def _get_first(df):
        return df.sort_values(['CODE', 'SURVEY']).groupby('CODE').nth(0).reset_index()

    def _setup_strat(strat_value=None):
        if strat_value is None:
            strat_string = 'All'
            baseline = df_base
        else:
            strat_string = f'{strat}={strat_value}'
            mask = df_base[strat] == strat_value
            baseline = df_base[mask]
        dataset_size = baseline.shape[0]
        if legacy_mode:
            dataset_size = baseline[strat].notna().sum()

        column_name = f'{strat_string}, Total n={dataset_size} ({dataset_size / df_base.shape[0] * 100:.1f}%), N(%)'

        return baseline, strat_string, dataset_size, column_name

    def _freq_table_counts(strat_value=None):
        baseline, strat_string, dataset_size, column_name = _setup_strat(strat_value=strat_value)

        counts = baseline[current_var].value_counts()
        if columns_100_percent:
            col_total = counts.sum()
        else:
            col_total = dataset_size
        counts = counts.apply(lambda x: f"{x} ({x / col_total * 100:.1f})")
        counts.name = column_name
        return counts

    def _freq_table_iqr(strat_value=None):
        baseline, strat_string, dataset_size, column_name = _setup_strat(strat_value=strat_value)

        q = baseline[current_var].quantile([.25, .5, .75])
        counts = pd.DataFrame([f"med(IQR) = {q[.5]:0.1f}({q[.25]:0.1f} - {q[.75]:0.1f})"], index=[1.0])[0]
        counts.name = column_name
        return counts

    def _get_model_data():
        results = proc_logistic(df_base, strat, [current_var], classes=ref_level, show_sas_code=show_sas_code).reset_index()
        results['Category'] = results['Category'].replace('', np.nan)
        if results['Category'].isna().all():
            results['Value'] = 1.0
        else:
            results['Value'] = results['Category'].astype(float)
        results = results.set_index('Value')
        p_value = pd.DataFrame(results['P-Value'])
        p_value.name = 'P-Value'

        or_cli = results.apply(lambda x: f"{x['Estimate']} ({x['Lower']} - {x['Upper']})", axis=1)
        or_cli.name = 'Odds Ratio (95% CL)'

        return p_value.join(or_cli).join(results['WaldChiSq'])

    results = []
    for current_var in ind_vars:
        try:
            try:
                if classes:
                    ref_level = {current_var: classes[current_var]}
                else:
                    ref_level = None
            except KeyError:
                ref_level = None
            try:
                if category_labels:
                    category_label = category_labels[current_var]
                else:
                    category_label = None
            except KeyError:
                category_label = None

            if baseline:
                df_base = _get_first(df)
            else:
                df_base = df

            values = df[current_var].dropna().unique()

            if (len(values) < cut_off_continuous) | (current_var in (classes if classes else [])):
                base_results = pd.DataFrame(values, columns=['Value']).sort_index(ascending=False).set_index('Value')

                base_results = base_results.join(_freq_table_counts())
                base_results = base_results.join(_freq_table_counts(1))
                base_results = base_results.join(_freq_table_counts(0))

                base_results = base_results.join(_get_model_data())

                base_results['Variable'] = current_var
                base_results = base_results.reset_index(drop=False)
                base_results['Value'] = base_results['Value'].astype(int)

                if category_label:
                    base_results['Value'] = base_results['Value'].map(category_label)

                base_results = base_results.set_index(['Variable', 'Value'])
                base_results = base_results.sort_index(ascending=False)
                results.append(base_results)
            else:
                base_results = pd.DataFrame([1.0], columns=['Value']).sort_index(ascending=False).set_index('Value')

                base_results = base_results.join(_freq_table_iqr())
                base_results = base_results.join(_freq_table_iqr(1))
                base_results = base_results.join(_freq_table_iqr(0))

                base_results = base_results.join(_get_model_data())

                base_results['Variable'] = current_var
                base_results = base_results.reset_index(drop=False)
                base_results['Value'] = ""
                base_results = base_results.set_index(['Variable', 'Value'])
                results.append(base_results)
        except Exception as e:
            print(f"Failed on {current_var}")
            print(e)

    results_df = pd.concat(results)
    results_df.iloc[:, 0:3] = results_df.iloc[:, 0:3].fillna(f'0 ({0:0.1f})')
    return results_df.fillna('')
