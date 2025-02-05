from bccsu.sas.commands import *
from bccsu.sas.table_parse import *
from bccsu.sas.tools.parse_tools import prettify, clean_model_wrapper
from bccsu.sas.stat_wrappers import *


def freqs(*args, **kwargs):
    df = freqs_command(*args, outputs=['report'], **kwargs)['report']
    df = prettify(df, ['VarName', 'Value'], ['Chi', 'wilcox'], ['cpval', 'fpval'])
    return df


def backward_gee(*args, **kwargs):
    df = backward_gee_command(*args, **kwargs, outputs=['report'])['report']
    df = prettify(df, ['outcomeName', 'EFFECT'], ['Estimate', 'LOWER', 'UPPER'], ['PROBT'])
    return df


def rrisk(*args, **kwargs):
    df = rrisk_command(*args, **kwargs, outputs=['report'])['report']
    df = prettify(df, ['outcomeName', 'Parameter'], ['Estimate', 'Lower', 'Upper'], ['Prob'])
    return df


def convert_cox_parse(*args, **kwargs):
    return convert_cox(*args, **kwargs, outputs=['OUTPUT_COX'])['OUTPUT_COX']


@clean_model_wrapper
def proc_logistic(*args, **kwargs):
    df, nobs, _, classes = proc_logistic_parse(*args, **kwargs)
    return df, nobs, classes


@clean_model_wrapper
def proc_genmod(*args, **kwargs):
    df, nobs, _, classes = proc_genmod_parse(*args, **kwargs)
    return df, nobs, classes


@clean_model_wrapper
def prevalence_ratio(*args, **kwargs):
    df, nobs, _, classes = prevalence_ratio_parse(*args, **kwargs)
    return df, nobs, classes


@clean_model_wrapper
def proc_glimmix(*args, **kwargs):
    df, nobs, _, classes = proc_glimmix_parse(*args, **kwargs)
    return df, nobs, classes


@clean_model_wrapper
def proc_phreg(*args, **kwargs):
    df, nobs, _, classes = proc_phreg_parse(*args, **kwargs)
    return df, nobs, classes


@clean_model_wrapper
def proc_reg(*args, **kwargs):
    df, nobs, _, classes = proc_reg_parse(*args, **kwargs)
    return df, nobs, classes


@bivar
def bivar_reg(*args, **kwargs):
    return proc_reg_parse(*args, **kwargs)


@bivar
def bivar_logistic(*args, **kwargs):
    return proc_logistic_parse(*args, **kwargs)


@bivar
def bivar_glmm(*args, **kwargs):
    return proc_glimmix_parse(*args, **kwargs)


@bivar
def bivar_gee(*args, **kwargs):
    return proc_genmod_parse(*args, **kwargs)


@bivar
def bivar_cox(*args, **kwargs):
    return proc_phreg_parse(*args, **kwargs)


@explanatory
def exp_logistic(*args, **kwargs):
    return proc_logistic_parse(*args, **kwargs)


@explanatory
def exp_glmm(*args, **kwargs):
    return proc_glimmix_parse(*args, **kwargs)


@explanatory
def exp_gee(*args, **kwargs):
    return proc_genmod_parse(*args, **kwargs)


@explanatory
def exp_cox(*args, **kwargs):
    return proc_phreg_parse(*args, **kwargs)


@confounding(inverse_exponent=True)
def conf_logistic(*args, **kwargs):
    return proc_logistic_parse(*args, **kwargs)


@confounding(inverse_exponent=True)
def conf_glmm(*args, **kwargs):
    return proc_glimmix_parse(*args, **kwargs)


@confounding(inverse_exponent=True)
def conf_gee(*args, **kwargs):
    return proc_genmod_parse(*args, **kwargs)


@confounding(inverse_exponent=True)
def conf_cox(*args, **kwargs):
    return proc_phreg_parse(*args, **kwargs)


def get_vif(*args, **kwargs):
    return vif_parse(*args, **kwargs)


def select(*args, **kwargs):
    return select_command_parse(*args, **kwargs)


def causaltreat(*args, **kwargs):
    """

    :param args: df, dependent, treatment, independent
    :param kwargs: classes
    :return: df with column for predicted values, nobs
    """
    return proc_causaltrt_parse(*args, **kwargs)


def get_predictions(*args, **kwargs):
    return get_predictions_parse(*args, **kwargs)


def frequencies(*args, **kwargs):
    """Returns the standard CSU frequencies table.
    Currently it uses wilcoxon for continuous variables and chi-square for categorical variables.
    We can later expand this to Kruksal-Wallis if we need to examine categorical outcomes."""
    return frequencies_parse(*args, **kwargs)
