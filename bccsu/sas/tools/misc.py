import pandas as pd
from bccsu.sas.core import session
import numpy as np


def get_kept(df):
    mask = (df['P-Value < .1'] == 'x').groupby(level=[0, 1], sort=False).any()
    return mask.index[mask].get_level_values(1).to_list()


def read_table_file(path):
    df = pd.read_excel(path)
    df[['Outcome', 'Parameter']] = df[['Outcome', 'Parameter']].ffill()
    df = df.fillna('')
    df = df.set_index(['Outcome', 'Parameter', 'Category'])
    return df


def get_ods_trace(sas_command):
    """
    Finds the tables output by a sas command.
    :param sas_command:
    :return:
    """
    import re
    get_names_re = re.compile(r'Output Added:\n-------------\nName:\s*?([A-Za-z]*?)\n')
    l = session.submit('ods trace on;' + sas_command)['LOG']
    return get_names_re.findall(l)


def get_output(sas_command):
    """
    Grabs all tables from a sas command.
    :param sas_command:
    :return:
    """
    # Needs to run the sas command twice to work properly.
    import re
    get_names_re = re.compile(r'Output Added:\n-------------\nName:\s*?([A-Za-z]*?)\n')
    l = session.submit('ods trace on;' + sas_command)['LOG']
    tables = get_names_re.findall(l)
    get_tables = 'ods output ' + ' '.join([f'{table} = {table}' for table in tables]) + ';'
    session.submit(f"proc delete {' '.join(tables)};")
    session.submit(get_tables + sas_command)
    table_dict = {}
    for table in tables:
        table_dict[table] = session.sd2df(table)
    return table_dict


def get_cox_dataset(data, event, time_var='INT_DATE', remove_records_after_na=True):
    count_array = data.groupby(['CODE']).count()['SURVEY'] >= 1
    restricted_data = data[data['CODE'].isin(count_array[count_array].index)].copy()

    restricted_data['EVENT'] = restricted_data[event]
    restricted_data['last_event'] = restricted_data['EVENT'].shift(1)
    restricted_data['last_date'] = restricted_data[time_var].shift(1)
    restricted_data['time_delta'] = (restricted_data[time_var] - restricted_data['last_date']).dt.days

    restricted_data.loc[(restricted_data['last_event'] == 1) | (restricted_data['EVENT'].isna()), 'time_delta'] = 0
    restricted_data.loc[restricted_data.groupby('CODE').head(1).index, 'time_delta'] = 0

    restricted_data.loc[restricted_data['time_delta'] == 0, 'group'] = 1
    restricted_data.loc[restricted_data['time_delta'] == 0, 'group'] = restricted_data[restricted_data['time_delta'] == 0][
        'group'].cumsum()
    restricted_data['group'] = restricted_data['group'].fillna(method='ffill')

    restricted_data['STOP_DAY'] = restricted_data.groupby('group')['time_delta'].transform(pd.Series.cumsum)

    restricted_data['START_DAY'] = np.where(restricted_data['STOP_DAY'] != 0, restricted_data['STOP_DAY'].shift(1), 0)

    event_day = restricted_data[restricted_data['EVENT'] == 1]
    restricted_data.loc[restricted_data['EVENT'] == 1, 'STOP_DAY'] = event_day['START_DAY'] + event_day['time_delta'] / 2

    if remove_records_after_na:
        for name, group in restricted_data.groupby('group'):
            mask = False
            for index, row in group.iterrows():
                if np.isnan(row[event]):
                    mask = True
                if mask:
                    restricted_data.loc[index, 'START_DAY'] = np.nan
                    restricted_data.loc[index, 'STOP_DAY'] = np.nan
                    restricted_data.loc[index, 'EVENT'] = np.nan

    final = restricted_data[~((restricted_data['STOP_DAY'] == 0) | restricted_data['EVENT'].isna())].copy()
    return final
