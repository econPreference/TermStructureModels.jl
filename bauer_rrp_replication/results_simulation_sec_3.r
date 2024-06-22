## analyze results of simulation study
## Tables 1, 2

rm(list=ls())
source("R/rrp_functions.r")
source("R/analysis_fns.r")
source("R/jsz_fns.r")

to.file <- FALSE

## DGP: Restricted Risk Prices
file.sim.mcmc <- "estimates/sim_rrp_mcmc_N2_20160206.RData"
file.sim.gvs <- "estimates/sim_rrp_gvs_N2_20160206.RData"
file.sim.ssvs <- "estimates/sim_rrp_ssvs_N2_20160206.RData"
file.sim.rjmcmc <- "estimates/sim_rrp_rjmcmc_N2_20160206.RData"
filename1 <- "tables/sim.tex"
filename2 <- "tables/sim_pers.tex"

## DGP: Unrestricted Risk Prices -- results in appendix
## file.sim.mcmc <- "estimates/sim_urp_mcmc_N2_20160228.RData"
## file.sim.gvs <- "estimates/sim_urp_gvs_N2_20160228.RData"
## file.sim.ssvs <- "estimates/sim_urp_ssvs_N2_20160228.RData"
## file.sim.rjmcmc <- "estimates/sim_urp_rjmcmc_N2_20160228.RData"
## filename1 <- "tables/sim_urp.tex"
## filename2 <- "tables/sim_pers_urp.tex"

files <- c(file.sim.mcmc, file.sim.ssvs, file.sim.gvs, file.sim.rjmcmc)

for (file in files) {
    print(file)
    load(file)
    print(true.pars$gamma)
}
print(true.pars$kinfQ)
print(true.pars$lamQ+1)
print(sqrt(true.pars$sige2)*1200)
print(round(true.pars$Sigma, digi=5))
print(max(abs(eigen(true.pars$Phi)$values)))
print(true.pars$lambda)

cat("# Table 1: simulation study - risk-price parameters\n")
tblSel <- getAllSimResults(file.sim.mcmc, file.sim.ssvs, file.sim.gvs, file.sim.rjmcmc)
printAllSimResultsToLatex(tblSel, to.file, filename1)

cat("# Table 2 - persistence/TP volatility\n")
tbl <- getAllSimResultsPersistence(files)
printAllSimResultsPersToLatex(tbl, to.file, filename2)

