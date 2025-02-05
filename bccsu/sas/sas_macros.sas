%let out_tables=;

%macro join_strings(delimiter, values);
    %local i count result value;
    %let count = %sysfunc(countw(&values));
    %let result =;

    %do i = 1 %to &count;
        %let value = %scan(&values, &i);
        %if &i = 1 %then %let result = &value;
        %else %let result = &result.&delimiter.&value;
    %end;

    &result;
%mend join_strings;

%macro add_tables(tables);
    /*Add output tables to list*/
    %let out_tables = &out_tables &tables;
%mend;

%macro prepend_name(names, prefix);
    /*Prepend prefix to each name in list*/
    %let out = ;
    %do i = 1 %to %sysfunc(countw(&names));
        %let name = %scan(&names, &i);
        %let out = &out &prefix._&name;
    %end;
    &out
%mend;

%macro long(in,out,dep,ind,keep=);
	data &out;
		set &in;
		array y [*] &dep;
		array x [*] &ind;
		do outcomeNum = 1 to dim(y);
			outcomeName=vname(y[outcomeNum]);
			outcomeValue=y[outcomeNum];
			do varNum = 1 to dim(x);
				VarName=vname(x[varNum]);
				Value = x[varNum];
				output;
			end;
		end;
		keep VarName Value VarNum outcomeNum outcomeValue outcomeName &keep;
	run;

	proc sort data=&out; by outcomeNum VarNum; run;
%mend;

%macro export_tables(path=);
	%if &path ne %then %do;
        ods excel file=&path;

        %do i = 1 %to %sysfunc(countw(&out_tables));
            %let table_name = %scan(&out_tables, &i);

            %if %sysfunc(exist(&table_name)) %then %do;
                ods excel options(sheet_name="&table_name");
                proc print data=&table_name;
                run;
            %end;
            %else %do;
                %put NOTE: The table &table_name does not exist.;
            %end;
        %end;
        %let out_tables=;
        ods excel close;
    %end;
%mend;

%macro get_frequencies(dataset, stratify_by, variables, name=);

    data &dataset._temp;
        set &dataset;
    run;

    %long(&dataset._temp, &dataset._LONG, &stratify_by, &variables);

    proc sort data=&dataset._LONG; by VarNum VarName; run;


    /* Careful, it treats missing as another level. */
    proc freq data=&dataset._LONG;
        by VarNum VarName;
        table OutcomeValue*Value / missing;
        ods output CrossTabFreqs=&name._CrossTabFreqs;
    run;

    proc freq data=&dataset._LONG;
        by VarNum VarName;
        table OutcomeValue*Value / chisq fisher;
        ods output chisq=&name._chisq FishersExact=&name._FishersExact;
    run;
%mend;

%macro get_med_iqr(dataset, stratify_by, variables, name=);
	%long(&dataset, &dataset._long, &stratify_by, &variables)

	proc sort data=&dataset._long;
		by outcomeNum outcomeName outcomeValue varNum VarName;
	run;

	proc univariate data=&dataset._long;
		by outcomeNum outcomeName outcomeValue varNum VarName;
		var Value;
		OUTPUT OUT=&name._univariate_out MEDIAN=median Q1=q1 Q3=q3;
	run;

	proc sort data=&dataset._long;
		by outcomeNum outcomeName varNum VarName;
	run;

	proc univariate data=&dataset._long;
		by outcomeNum outcomeName varNum VarName;
		var Value;
		OUTPUT OUT=&name._univariate_out_total MEDIAN=median Q1=q1 Q3=q3;
	run;

	proc npar1way data=&dataset._long wilcoxon;
		by outcomeNum outcomeName varNum VarName;
	    class outcomeValue;
	    var value;
		ods output WilcoxonTest=&name._WilcoxonTest;
	run;
%mend;

%macro get_baseline_characteristics(dataset, stratify_by, categorical_vars=, continuous_vars=, name=);
    proc freq data=&dataset;
        table &stratify_by / missing;
        ods output OneWayFreqs=&name._OneWayFreqs;
    run;

    %if &categorical_vars ne %then %do;
        %get_frequencies(&dataset, &stratify_by, &categorical_vars, name=&name);
    %end;
    %if &continuous_vars ne %then %do;
        %get_med_iqr(&dataset, &stratify_by, &continuous_vars, name=&name);
    %end;

    %add_tables(%prepend_name(univariate_out univariate_out_total CrossTabFreqs chisq WilcoxonTest OneWayFreqs FishersExact, &name));
%mend;