/*added ability to put multiple outcomes*/
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

/*Make it work for 3 level variables. also make it use labels instead of "0" "1"
	Need to fix problem where the table title is on multiple lines.
	If all counts above 5, don't use Fisher Exact test at all.*/
%macro freqs(in,id,strat,vars,survey=survey,base=1);

    ods exclude all;

    %let cont_test = 0;

    proc sort data=&in; by &id &survey; run;

    data &in._BASE(where=(&strat^=.));
        set &in;
        %if &base=1 %then %do;
            by &id;
            if first.&id;
        %end;
    run;

    %long(&in._BASE,&in._BASE_LONG,&strat,&vars);

    proc sort data=&in._BASE_LONG; by VarNum VarName; run;

    /*Check if continuous*/
    %long(&in._BASE,&in._BASE_LONG,&strat,&vars);
    proc freq data=&in._BASE_LONG nlevels;
        by VarNum;
        table Value;
        ods output nlevels=n_levels;
    run;

    data cont_vars;
        set n_levels;
        if NLevels>3 then continuous=1;
        if continuous=1 then call symput("cont_test",1);
    run;

    data &in._BASE_LONG;
        merge &in._BASE_LONG cont_vars(keep=VarNum continuous);
        by VarNum;
    run;

    /**********/

    proc freq data=&in._BASE_LONG(where=(continuous=.));
        by VarNum VarName;
        table OutcomeValue*Value /cl;
        exact fisher or;
        ods output OddsRatioExactCL=FishExact RelativeRisks=OR FishersExact=FishP ChiSq=Chi CrossTabFreqs=Freqs;
    run;

    data stats;
        merge 	chi(where=(Statistic="Chi-Square") rename=(Value=Chi Prob=cpval))
                fishp(where=(Name1="XP2_FISH") drop=cValue1 rename=(nValue1=fpval))
                OR(where=(statistic="Odds Ratio") rename=(Value=OR))
                FishExact(where=(Name1="XL_RROR")drop=cValue1 rename=(nValue1=XLower))
                FishExact(where=(Name1="XU_RROR") drop=cValue1 rename=(nValue1=XUpper));
        by VarNum VarName;
        Value=1;
        keep VarNum VarName Chi cpval fpval OR LowerCL UpperCL XLower XUpper Value DF;
    run;

    proc sql;
        select count(*) into :CountTot from &in._BASE;
        select count(*) into :Count0 from &in._BASE(where=(&strat=0));
        select count(*) into :Count1 from &in._BASE(where=(&strat=1));
    quit;

    %put &CountTot;
    %put &Count0;
    %put &Count1;

    %if &cont_test=1 %then %do;
        /*For Continuous Variables only*/
        data range_find;
            set &in._BASE_LONG(where=(continuous=1));
            group=0;
            output;
            if OutcomeValue = 0 then do;
                group=1;
                output;
            end;
            if OutcomeValue = 1 then do;
                group=2;
                output;
            end;
        run;

        proc sort data=range_find;
            by VarNum VarName group;
        run;

        proc means data=range_find p25 p50 p75;
            by VarNum VarName group;
            var Value;
            ods output summary=ranges;
        run;

        proc glimmix data=&in._BASE_LONG(where=(Continuous=1));
            by VarNum VarName;
            model OutcomeValue=Value / dist=binomial or;
            ods output OddsRatios=est;
        run;

        proc npar1way data=&in._BASE_LONG(where=(Continuous=1)) wilcoxon;
            by VarNum VarName;
            class OutcomeValue;
            var Value;
            ods output WilcoxonTest=pval;
        run;

        data continuous_stats;
            merge est pval(keep=VarNum VarName tProb2);
            by VarNum VarName;
            array med medTot med0 med1;
            array medl medlTot medl0 medl1;
            array medu meduTot medu0 medu1;
            if first.VarNum then do;
                i=1;
                do until(last.VarNum);
                    set ranges;
                    by VarNum;
                    med[i]=Value_P50;
                    medl[i]=Value_P25;
                    medu[i]=Value_P75;
                    i+1;
                end;
            end;
            continuous=1;
            rename tProb2=wilcox;
            Value=1;
            keep VarNum VarName Estimate Lower Upper tProb2 Value med: continuous;
        run;
    %end;
    /******************************/



    proc sort data=stats; by VarNum VarName Value; run;

    data freq_merge;
        merge 	Freqs(where=(OutcomeValue=0 and _TYPE_="11") rename=(Frequency=strat0))
                Freqs(where=(OutcomeValue=1 and _TYPE_="11") rename=(Frequency=strat1))
                Freqs(where=(OutcomeValue=. and _TYPE_="01") rename=(Frequency=stratTot));
        by VarNum VarName Value;
        keep VarNum VarName Value strat0 strat1 stratTot;
    run;


    data Frequency_Out;
        merge stats freq_merge
        %if &cont_test=1 %then %str(continuous_stats);
        ;
        by VarNum VarName Value;
        array Totals $100 Total Total0 Total1;
        array Strat StratTot Strat0 Strat1;
        array Med	MedTot Med0 Med1;
        array medl medlTot medl0 medl1;
        array medu meduTot medu0 medu1;
        array counts {3} _temporary_ (&CountTot &Count0 &Count1);
        if not continuous then do;
            do i=1 to dim(Totals);
                Totals[i]=cat(Strat[i],' (',round(Strat[i]/counts[i]*100,.1),')');
            end;
            if Value=1 then do;
                ORcl=cat(round(OR,.01),' (',round(LowerCL,.01),' - ',round(UpperCL,.01),')');
                fcl=cat(round(OR,.01),' (',round(XLower,.01),' - ',round(XUpper,.01),')');
            end;
        end;
        else do;
            do i=1 to dim(Totals);
                Totals[i]=cat('med(IQR) = ',round(Med[i],.1),' (',round(medl[i],.1),'-',round(Medu[i],.1),')');
            end;
            if Value=1 then ORcl=cat(round(Estimate,.01),' (',round(Lower,.01),' - ',round(Upper,.01),')');
            Value=.;
        end;
    run;

    ods exclude none;

    title "Baseline Descriptive Statistics Stratified by &strat";

    title3 justify=left font=arial color=black italic height=9.5pt "* Use P-Value Fisher for counts<5.";
    title4 justify=left font=arial color=black italic height=9.5pt "** OR for continuous variables obtained using regression.";


    proc report headline data=Frequency_Out ls=256 spanrows;
        column ("Descr; Stratified: &strat" VarNum VarName Value Total Total1 Total0 Chi cpval ORcl fpval fcl %if &cont_test=1 %then %str(wilcox););
        define VarNum / order order=internal noprint;
        define VarName / display group order=data 'Variable';
        define Value / display  'Value' missing order descending;
        define Total / display "All, Total n=%sysfunc(trim(&CountTot)),(100%), N(%)" style(column)={tagattr='wraptext:no' width=100%};
        define Total1 / display "&strat=1, Total n=%sysfunc(trim(&count1)) (%sysfunc(round(%sysevalf(&count1/&CountTot*100),.01))%), N(%)";
        define Total0 / display "&strat=0, Total n=%sysfunc(trim(&count0)) (%sysfunc(round(%sysevalf(&count0/&CountTot*100),.01))%), N(%)";
        define Chi / display 'Chi-Square' format=8.2;
        define cpval / display 'P-Value' format=PVALUE6.;
        define ORcl / display 'Odds Ratio (95% CL)';
        define fpval / display 'P-Value Fisher' format=PVALUE6.;
        define fcl / display 'Odds Ratio (95% CL)';
        define wilcox / display 'P-Value Mann-Whitney' format=PVALUE6.;
    run;

    title;
    title3;
    title4;

    proc delete data=CHI CONTINUOUS_STATS EST FISHEXACT FISHP FREQS FREQ_MERGE OR PVAL RANGES RANGE_FIND STATS;
    run;
%mend;

/*Combined most of my Bivariate functions.
types=GEE,GLMM,Logistic(for cross-sectional data)
Set binomial=0 for continuous var.
*/
%macro bivar(type,in,outcome,vars,id=code,binomial=1,keepnames=,moretitles=,ref='0',cutoff=.1);
	%long(&in,&in._LONG,
			&outcome,
			&vars,
			keep=&id);

	%if &type=LINEAR %then %let binomial=0;

	ods exclude all;
	%if %upcase(&type)=GEE %then %do;
		proc genmod data=&in._LONG;
			by outcomenum outcomename VARNUM VARNAME;
			class &id %if &binomial=1 %then %str(OutcomeValue(ref=&ref));;
			model OutcomeValue = VALUE / CL %if &binomial=1 %then %str(dist=binomial link=logit); lrci;
			repeated subject=&id %if &binomial=1 %then %str(/ type=exch); ;
			ods output
				GEEEmpPEst=PE NObs=NObs;
		run;
		data PE;
			merge PE(where=(Parm^="Intercept")) NObs(keep=outcomenum varnum outcomeName varName Label N where=(Label="Number of Observations Used"));
			by outcomenum outcomeName varnum varName;
		run;
		%let title="Bivariate GEE";
	%end;

	%if %upcase(&type)=LINEAR %then %do;
		proc reg data=&in._LONG;
			by outcomenum outcomename VARNUM VARNAME;
			model OutcomeValue = VALUE / clb;
			ods output ParameterEstimates=PE Nobs=NObs;
		run;

		data PE;
			merge PE(where=(Variable^="Intercept")) NObs(keep=Label N outcomenum outcomename VARNUM VARNAME where=(Label="Number of Observations Used"));
			rename probt=probz;
		run;

		%let title="Linear Regression";
	%end;

	%if %upcase(&type)=GLMM %then %do;
		proc glimmix data=&in._LONG method=quad(qpoints=100);
			by outcomenum outcomename VARNUM VARNAME;
			class &id;
			model OutcomeValue = VALUE / solution CL %if &binomial=1 %then %str(dist=binomial oddsratio); ;
			random intercept / subject=&id;
			covtest glm / wald;
			ods output  %if &binomial=1 %then %str(OddsRatios=odds); %else %str(ParameterEstimates=odds); Tests3=prob ConvergenceStatus=Convergence NObs=NObs;
		run;

		data PE;
			merge odds(keep=outcomenum varnum outcomeName varName Estimate Lower Upper %if &binomial=0 %then %str(Effect where=(Effect^="Intercept"));)
				  prob(keep=outcomenum varnum outcomeName varName ProbF)
				  NObs(keep=outcomenum varnum outcomeName varName Label N where=(Label="Number of Observations Used"));
			drop label Effect;
			by outcomenum outcomeName varnum varName;
			rename Lower=LowerCL Upper=UpperCL ProbF=ProbZ;
		run;
		%let title="Bivariate GLMM";
	%end;

	%if %upcase(&type)=LOGISTIC %then %do;
		proc genmod data=&in._LONG;
			by outcomenum outcomename VARNUM VARNAME;
			class OutcomeValue(ref=&ref);
			model OutcomeValue = VALUE / CL dist=binomial link=logit lrci;
			ods output ParameterEstimates=PE NObs=NObs;
		run;

		data PE;
			merge PE(where=(upcase(parameter)="VALUE")) NObs(keep=outcomenum varnum outcomeName varName Label N where=(Label="Number of Observations Used"));
			by outcomenum outcomeName varnum varName;
			rename LowerLRCL=LowerCl UpperLRCL=UpperCL ProbChiSq=ProbZ Parameter=Parm;
		run;
		%let title="Bivariate Logistic";
	%end;

	ods exclude none;


	title1 &title;
	title2 justify=left font=arial color=black italic height=9.5pt "* P-Values < &cutoff are colored green.";
	&moretitles;
	proc report data=PE ls=256 spanrows;
		column (&title outcomenum outcomename VarNum VarName estimate LowerCL UpperCL ProbZ N);
		define outcomenum / order order=internal noprint;
		define outcomename / display group order=data 'Outcome' style(column)=[background=#EDF2F9 color=#112277 fontweight=bold vjust=top];
		define VarNum / order order=internal noprint;
		define VarName / display group order=data 'Exposure';
		define estimate / display %if &binomial=1 %then %str("Odds Ratio"); %else %str("Beta"); format=8.2;
		define LowerCL / display "Lower" format=8.2;
		define UpperCL / display "Upper" format=8.2;
		define ProbZ / display "P Value" format=PValue6.;
		define N / display "N";

		compute N;
		%if %upcase(&type)^=GLMM and &binomial=1 %then %do;
			estimate=exp(estimate);
			LowerCL=exp(LowerCL);
			UpperCL=exp(UpperCL);
		%end;
			if ProbZ < &cutoff and ProbZ^=. then do;
				do i=3 to 9;
					call define(i ,"style","style={foreground=#006100 background=#C6EFCE}");
				end;
			end;
			else if ProbZ=. then do;
				do i=3 to 9;
					call define(i ,"style","style={foreground=#9C0006 background=#FFC7CE}");
				end;
				estimate=.;
				LowerCL=.;
				UpperCL=.;
			end;
		endcomp;
	run;
	title1;
	title2;

	%if &keepnames^= %then %do;
	%global &keepnames;
		proc sql noprint;
			select VarName into :tempnames separated by ' '
			from PE having ProbZ < &cutoff and not ProbZ=.;
		quit;
	%let &keepnames=&tempnames;
	%end;

	proc delete data=PE &in._LONG;
	run;
%mend;

