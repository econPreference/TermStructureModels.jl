
---
title: Readme file for replication code and data
author: Michael Bauer
date: 2016/03/07
geometry: margin=3cm
---

This ZIP file contains all the code and data required for reproducing
the results in the paper and in the online appendix.

# Paper "Restrictions on Risk Prices in Dynamic Term Structure Models"
- Cite as:

	Bauer, Michael D., "Restrictions on Risk Prices in Dynamic Term
    Structure Model," forthcoming in Journal of Business & Economic
    Statistics.

- FRBSF Working Paper 2011-03 (updated with newest version) available
at

	\url{http://www.frbsf.org/economic-research/publications/working-papers/2011/wp11-03bk.pdf}

- Online Appendix available on my website

	\url{http://www.frbsf.org/economic-research/economists/michael-bauer/}

# Data

The paper uses monthly Treasury yields that are described in
Section 4. The data is stored in data/le_data_monthly.RData. This file
contains a date vector "dates" and a matrix "Y" with the yields. These
are end-of-month observations, from June 1961 to December 2012, for
monthly maturities from 1 to 120 months. The paper uses only a sample
from January 1990 to December 2007, and only maturities of 1 through
5, 7, and 10 years.

If you would like to use this data in your own work, you need to first ask
Anh Le for permission.

# Code

The code is written in R. You can download and install R for free at
\url{https://www.r-project.org/}. I used version 3.2.0.

You will need to install the following R packages from CRAN:

- MCMCpack
- mvtnorm

## Reproduce simulation study (Section 3)
First run the scripts for the simulations as follows (from the main directory)---Note that each of these each take several hours to run.

	source("R/sim_mcmc.r")
	source("R/sim_gvs.r")
	source("R/sim_ssvs.r")
	source("R/sim_rjmcmc.r")

To analyze the results using `results_simulation_sec_3.r`.

## Reproduce empirical analysis (Sections 4 and 5)
First run the following estimation scripts:

   - `est_mcmc.r` for estimation of unrestricted model (use "M0")
   - `est_gvs.r`, `est_ssvs.r`, `est_rjmcmc.r` for joint model-parameter sampling

Then carry out sensitivity analysis using `est_gvs.r`

   - set `g` to 10, 1000, and 10000
   - for each setting change string from "gvs" to, for example,
     "gvs_10" in call to `getResultsFileName`

Now you can analyze the results for estimation, model selection, and sensitivity analysis using
`results_estimation_sec_4.r`. This produces tables 3-6.

Before you can analyze the economic implications of the different
models, you'll need to estimate the restricted models individually
using script `est_mcmc.r`. Change `model` to "M1", "M2", "M3",
respectively.

You are now able to analyze volatilities, term premia, and return predictability using
`results_economics_sec_5.r`. This produces tables 7-9 and figures 1-2

## Questions, feedback, suggestions?
Contact me at michael.bauer@sf.frb.org
