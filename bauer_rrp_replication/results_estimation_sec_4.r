## analyze results of model estimation and selection
## Tables 3, 4, 5, 6
rm(list=ls())
source("R/rrp_functions.r")
source("R/estimation_fns.r") # for estAllModels()
source("R/analysis_fns.r")
source("R/jsz_fns.r")

to.latex <- FALSE

filename.mcmc <- "estimates/mcmc_M0_N3_20160306.RData"
filename.ssvs <- "estimates/ssvs_N3_20160306.RData"
filename.gvs <- "estimates/gvs_N3_20160306.RData"
filename.rjmcmc <- "estimates/rjmcmc_N3_20160306.RData"

cat("# Table 3 - Estimates for unrestricted model\n")
printParameterEstimates(filename.mcmc, to.latex)

cat("# Table 4 - Risk price restrictions\n")
## summary statistics for gamma
printGammaStats(c(filename.ssvs, filename.gvs, filename.rjmcmc), to.latex)

cat("# Table 5 - Posterior model probabilities\n")
all.models <- estAllModels() ## for AIC/BIC
printModels(c(filename.ssvs, filename.gvs, filename.rjmcmc), all.models, to.latex)

## posterior probability that level/slope/curve risk is priced
pricedRisks(filename.gvs)

cat("# Table 6 - Model selection and prior dispersion\n")
## sensitivity to prior dispersion
printModelFreq("estimates/gvs_10000_N3_20160306.RData")
printModelFreq("estimates/gvs_1000_N3_20160306.RData")
printModelFreq("estimates/gvs_N3_20160306.RData")
printModelFreq("estimates/gvs_10_N3_20160306.RData")
