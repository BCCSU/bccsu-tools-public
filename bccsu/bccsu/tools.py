import pandas as pd
import numpy as np
import os
from pathlib import Path

home_dir = Path(os.getenv("USERPROFILE"))


def check_for_bad_values(data, value, remove=True):
    """
    Check for bad data and remove it if it exists.
    :param data:
    :param value:
    :param remove:
    :return:
    """
    printable_val = value

    if isinstance(value, str):
        printable_val = f'"{value}"'

    check = (data == value).sum()
    if check[check > 0].any():
        print(f'Records with {printable_val}:')
        print(check)
        if remove:
            data.replace(value, np.nan, inplace=True)
            print(f'Records with {printable_val} set to na.')
    else:
        print(f'There are no {printable_val} records.')


def create_missing_table(df):
    missing = df.isna().sum()
    missing_table = pd.DataFrame(missing, columns=['N Missing'])
    missing_table['Percent'] = ((missing / df.shape[0]) * 100).round(2)
    return missing_table


def check_other_issues(df):
    cols = []
    for column in df:
        val_counts = df[column].value_counts()
        unique = val_counts.count()
        val_dict = {'Column Name': column,
                    'Unique Values': unique,
                    'Proportion of largest': val_counts.max() / df.shape[0]}
        if (unique < 10) and column:
            val_dict.update(pd.DataFrame(val_counts).to_dict()[column])
        cols.append(val_dict)
    return pd.DataFrame(cols).sort_values('Proportion of largest', ascending=False)


def convert_from_binary(df):
    for column in df:
        if df[column].dtype == object:
            df[column] = df[column].str.decode('utf-8')


def labels_upper(df):
    df.rename(columns={i: i.upper() for i in df if not i.isupper()}, inplace=True)


def remove_bad_values(df):
    check_for_bad_values(df, '-77')
    check_for_bad_values(df, -77)
    check_for_bad_values(df, '-99')
    check_for_bad_values(df, -99)
    check_for_bad_values(df, '')


def import_data(path):
    if Path(path).suffix == '.sas7bdat':
        df = pd.read_sas(path)
        # SAS character type converts to binary type.
        convert_from_binary(df)

    else:
        df = pd.read_csv(path)
        mask = df.astype(str).apply(lambda x: x.str.match(r'\d{2}[A-Z]{3}\d{4}').all())
        df.loc[:, mask] = df.loc[:, mask].apply(pd.to_datetime)
        if 'CODE' in df.columns:
            if df['CODE'].dtype == np.int64:
                df['CODE'] = df['CODE'].astype(str).str.pad(width=4, fillchar='0')

    labels_upper(df)
    remove_bad_values(df)
    missing_table = create_missing_table(df)
    other_issues = check_other_issues(df)
    return df, missing_table, other_issues


def pretty_print_p_value(p):
    if isinstance(p, str):
        return p
    else:
        if float(p) < .0001:
            p_str = "<.0001"
        else:
            p_str = f"{p:.4f}"
        return p_str


def get_survey_period(date_array):
    def survey_period(x):
        """
        Return the survey period
        :param x: datetime pandas array.
        :return: integer pandas array with study period.
        """
        # Survey 22 was in the first half of 2016.
        reference_survey = 22
        # should work for everything. But should double check at each use.
        if x.month in [12, 1, 2, 3, 4, 5]:
            month_offset = 0
            if x.month == 12:
                month_offset = 2
        else:
            month_offset = 1

        year_offset = 2 * (x.year - 2017)

        return reference_survey + year_offset + month_offset

    return date_array.map(survey_period)


def get_person_years_100(data, followups_per_year=2):
    # Not per 100.
    # person-years = data.shape[0] / followups_per_year
    data = data.dropna()
    a = data.sum()
    n = data.shape[0] / followups_per_year
    r = a / n
    se = np.sqrt((1 - r) / a)
    lower = r / np.exp(1.96 * se)
    upper = r * np.exp(1.96 * se)
    return pd.DataFrame([r, lower, upper], index=['Rate', 'Lower', 'Upper'], columns=['Person Years'])


def create_encoding(dataframe, labels, classes, variable, column_name=None):
    """
    This function encodes categorical variables in a dataframe to numerical values. It also updates the labels and
    classes dictionaries with the new mappings.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe containing the variable to be encoded.
    labels (dict): A dictionary where the keys are the variable names and the values are dictionaries mapping the
    original values to the new numerical values.
    classes (dict): A dictionary where the keys are the variable names and the values are the class labels.
    variable (str): The name of the variable to be encoded.
    column_name (str, optional): The new name for the variable after encoding. If not provided, the original variable
    name is used.

    Returns:
    None
    """
    # If a new column name is provided, create a new column with the same values as the original variable
    if column_name is not None:
        dataframe[column_name] = dataframe[variable]
        variable = column_name

    # Get the unique values of the variable
    var_array = dataframe[variable].drop_duplicates(ignore_index=True)
    # Set the index of the series to its own values
    var_array.index = var_array
    # Rank the unique values and drop any NaN values
    rank = var_array.rank().dropna().astype(int)
    rank.name = variable

    # Copy the rank series and convert the index to string
    rank_labels = rank.copy()
    rank_labels.index = rank_labels.index.astype(str)
    # Update the labels dictionary with the mapping from the original values to the new numerical values
    labels[variable] = {y: k for k, y in rank_labels.to_dict().items()}
    classes[variable] = 0
    # Replace the original values in the dataframe with the new numerical values
    dataframe[variable] = dataframe[[variable]].merge(rank, left_on=variable, right_index=True, suffixes=('_old', ''))[
        variable]
