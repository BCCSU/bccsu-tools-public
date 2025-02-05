from bccsu.sas.core import base_command_pythonize, output_serializer


@base_command_pythonize
def freqs_command(dependent, independent, **kwargs):
    command = rf"""
%freqs(TEMP_DATA, CODE, {dependent}, {' '.join(independent)}, survey=survey, base=1)
"""
    return command


@base_command_pythonize
def bivar_command(dependent, independent, model_type, **kwargs):
    command = rf"""
%bivar({model_type}, TEMP_DATA,{dependent}, {' '.join(independent)})
"""
    return command


@base_command_pythonize
def backward_gee_command(dependent, primary_independent, independent, show_steps=False, binomial=True, **kwargs):
    command = rf"""
%backward_gee(TEMP_DATA, {dependent}, {primary_independent}, {' '.join(independent)}, showsteps=
{int(show_steps)}, binomial={int(binomial)});
"""
    return command


@base_command_pythonize
def rrisk_command(dependent, independent, **kwargs):
    command = rf"""
%rrisk(TEMP_DATA, {dependent}, {' '.join(independent)});
"""
    return command


@base_command_pythonize
def proc_logistic_command(dependent, independent, classes=None, outputs=None, link='logit', options=None, **kwargs):
    if not classes:
        classes = {}
    if options is not None:
        options = ' '.join(options)
    else:
        options = ''

    # Apparently SAS uses effect coding rather than dummy coding.
    class_variables = ' '.join(
        [f'{key}(ref="{item}")' for key, item in classes.items() if key.lower() in independent + [dependent]])
    if class_variables:
        class_variables += ' / param=ref'

    command = rf"""
proc logistic data=TEMP_DATA NAMELEN=32 descending;
    class {class_variables};
    model {dependent} = {' '.join(independent)} / {options} link={link};
    {output_serializer(outputs)}
run;
"""
    return command


@base_command_pythonize
def proc_reg_command(dependent, independent, classes=None, outputs=None, **kwargs):
    if not classes:
        classes = {}

    command = rf"""
proc glm data=TEMP_DATA;
    class code {' '.join([f'{key}(ref="{item}")' for key, item in classes.items() if key.lower() in independent + [dependent]])};
    model {dependent} = {' '.join(independent)} / solution clparm;
    {output_serializer(outputs)}
run;
"""
    return command


@base_command_pythonize
def proc_genmod_command(dependent, independent, classes=None, outputs=None, dist='binomial', link='logit',
                        descending=True, **kwargs):
    if not classes:
        classes = {}
    try:
        main_ref = classes[dependent]
    except KeyError:
        main_ref = 0

    if dist == 'multinomial':
        corr_type = 'ind'
        link = 'clogit'
    else:
        corr_type = 'exch'
    independent = [i.lower() for i in independent]

    classes = {key: item for key, item in classes.items() if key.lower() in independent + [dependent]}

    command = rf"""
proc genmod data=TEMP_DATA NAMELEN=32 {'descending' if descending else ''};
    class {dependent}(ref='{main_ref}') {'CODE' if not kwargs.get('cross') else ''} 
{' '.join([f'{key}(ref="{item}")' for key, item in classes.items()])};
    model {dependent} = {' '.join(independent)} / CL dist={dist} link={link} lrci;
    {f'repeated subject=CODE / type={corr_type};' if not kwargs.get('cross') else ''}
   
    ods exclude ClassLevels ObStats;
    {output_serializer(outputs)}
run;
"""
    return command


@base_command_pythonize
def proc_glimmix_command(dependent, independent, classes=None, outputs=None, qpoints=100, dist='binomial',
                         link='logit', outcome_arg='', **kwargs):
    if not classes:
        classes = {}
    # Descending option shouldn't break anything. But keep an eye on it.
    command = rf"""
proc glimmix data=TEMP_DATA method=quad(qpoints={qpoints}) NAMELEN=32;
    class code {' '.join([f'{key}(ref="{item}")' for key, item in classes.items() if key.lower() in independent + [dependent]])};
    model {dependent}{outcome_arg} = {' '.join(independent)}/ solution CL dist={dist} link={link} oddsratio;
    random intercept / subject=code;
    covtest glm / wald;
    {output_serializer(outputs)}
run;
"""
    return command


@base_command_pythonize
def proc_phreg_command(dependent, independent, classes=None, outputs=None, **kwargs):
    if not classes:
        classes = {}

    command = rf"""
proc phreg data=TEMP_DATA covs(aggregate) namelen=32;
    class code {' '.join([f'{key}(ref="{item}")' for key, item in classes.items() if key.lower() in independent + [dependent]])};
    model (start_day, stop_day)*{dependent}(0)= {' '.join(independent)} / rl;
    id code;
    {output_serializer(outputs)}
run;
"""
    return command


@base_command_pythonize
def vif_command(independent, outputs=None, **kwargs):
    command = rf"""
DATA TEMP;
    SET TEMP_DATA;
    temp_outcome_var = 1;
RUN;

proc reg data=TEMP;
    model temp_outcome_var= {' '.join(independent)} / vif;
    {output_serializer(outputs)}
run;

PROC DELETE DATA=TEMP;
RUN;
"""
    return command


@base_command_pythonize
def ancova_command(independent, dependent, outputs=None, **kwargs):
    command = rf"""
proc glm data=TEMP_DATA;
   class {dependent}(ref='0');
   model {independent} = {dependent} / solution;
   lsmeans {dependent} / stderr pdiff cov cl out=adjmeans;
   {output_serializer(outputs)}
run;

"""
    # ods output LSMeans=LSMeans LSMeanCL=LSMeanCL LSMeanDiffCL=LSMeanDiffCL ;
    return command


@base_command_pythonize
def prevalence_ratio_command(dependent,
                             independent,
                             identifier=None,
                             classes=None,
                             classes_values=None,
                             outputs=None,
                             table_name='temp_data',
                             **kwargs):
    # Using Zou's method
    # https://support.sas.com/kb/23/003.html
    # https://academic.oup.com/aje/article/162/3/199/171116
    # https://stats.oarc.ucla.edu/sas/faq/how-can-i-estimate-relative-risk-in-sas-using-proc-genmod-for-common
    # -outcomes-in-cohort-studies/
    if not classes:
        classes = {}
    try:
        main_ref = classes[dependent]
    except:
        main_ref = 0
    independent = [i.lower() for i in independent]
    lsmeans = '\n'.join([f"lsmeans {i} / diff exp cl;" for i in independent])

    classes = {key: item for key, item in classes.items() if key.lower() in independent}

    command = rf"""
proc genmod data={table_name};
    class {identifier} {dependent}(ref='{main_ref}') {' '.join([f'{key}(ref="{item}")' for key, item in classes.items() if key.lower() in independent + [dependent]])};
    model {dependent}={' '.join(independent)} /dist=poisson link=log;
    repeated subject={identifier}/type=ind;
    /*{lsmeans}*/
    {output_serializer(outputs, prefix=kwargs.get('prefix', ''))}
run;
"""
    return command


@base_command_pythonize
def select_command(dependent, independent, classes=None, outputs=None, steps=0, split=10, selection='lasso', stop=None,
                   **kwargs):
    if not classes:
        classes = {}
    try:
        main_ref = classes[dependent]
    except KeyError:
        main_ref = 0

    extra_args = []
    if selection == 'lasso':
        extra_args.append('ADAPTIVE')
    if stop:
        extra_args.append(f'stop={stop + 1}')  # 1 will only give you the intercept.
    if steps:
        extra_args.append(f'steps={steps}')

    extra_args = ' '.join(extra_args)

    command = rf"""
    proc glmselect data=temp_data plots=coefficients;
        class {dependent}(ref='{main_ref}') {' '.join([f'{key}(ref="{item}")' for key, item in classes.items() if key.lower() in independent + [dependent]])}/split;
        model {dependent} = {' '.join(independent)}/ selection={selection}(choose=cv {extra_args})
        stats=all cvmethod=split({split});
        {output_serializer(outputs)}
    run;
    """
    return command


@base_command_pythonize
def proc_causaltrt_command(dependent, treatment, independent, classes=None, outputs=None, **kwargs):
    if not classes:
        classes = {}

    # They use regular old linear regression in the paper.
    command = rf"""
proc causaltrt data=TEMP_DATA method=ipwr ppsmodel namelen=32;
   class {treatment}  {' '.join([f'{key}(ref="{item}")' for key, item in classes.items() if key.lower() in independent + [dependent]])}/desc;
   psmodel {treatment}(ref='0') = {' '.join(independent)};
   model {dependent}(ref='0') / dist = bin;
   {output_serializer(outputs)}
run;
"""
    return command


@base_command_pythonize
def predictions_command(dependent, independent, classes=None, classes_values=None, outputs=None, **kwargs):
    if not classes:
        classes = {}

    command = rf"""
proc genmod data=temp_data descending;
    class CODE {dependent} {' '.join([f'{key}(ref="{item}")' for key, item in classes.items() if key.lower() in independent + [dependent]])};
    model {dependent}={' '.join(independent)} /dist=normal link=identity;
    /*repeated subject=CODE/type=ind;*/ 
    output out=predictions pred=PREDICTED;
run;
"""
    return command


@base_command_pythonize
def frequencies_command(dependent, categorical_vars, continuous_vars, **kwargs):
    command = (
        rf"%get_baseline_characteristics(TEMP_DATA, {dependent}, categorical_vars={' '.join(categorical_vars)}, continuous_vars={' '.join(continuous_vars)});")
    # if len(categorical_vars) > 0:
    #     command += rf"%get_frequencies(TEMP_DATA, {dependent}, {' '.join(categorical_vars)});"
    # if len(continuous_vars) > 0:
    #     command += rf"%get_med_iqr(TEMP_DATA, {dependent}, {' '.join(continuous_vars)});"
    return command
