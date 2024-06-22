## analyze estimation results and economic implications
## Tables 7-9
## Figures 1, 2

rm(list=ls())
source("R/jsz_fns.r")
source("R/rrp_functions.r")
source("R/analysis_fns.r")

to.file <- FALSE
init(N=3)

## load model estimates
models <- list(M0 = loadModel("M0", "estimates/mcmc_M0_N3_20160306.RData"),
               M1 = loadModel("M1", "estimates/mcmc_M1_N3_20160306.RData"),
               M2 = loadModel("M2", "estimates/mcmc_M2_N3_20160306.RData"),
               M3 = loadModel("M3", "estimates/mcmc_M3_N3_20160306.RData"),
               BMA = loadModel("BMA", "estimates/gvs_N3_20160306.RData"))

## cross-sectional fit
for (model in models)
    analyzeFit(model)

printAvgFactors(models)
printAvgYieldCurves(models, sample.mean=FALSE)

cat("# Table 7 - Persistence and volatility\n")
printPersVol(models, to.file)

## Figure 1: Term structure of volatility
plotVolatilities(models[c("M0", "BMA")], to.file)

## Figure 2: risk-neutral yield and term premium
plotExpTP(models[c("M0", "BMA")], to.file)

cat("# Table 8 - Historical changes in long-term rates and expectations \n")
printHistChanges(models, to.file)

cat("# Table 9 - Return predictability\n")
mat.sel <- match(c(2,5,7,10)*n.per, mats)
for (m in seq_along(names(models))) {
    cat("calculating returns for", models[[m]]$name, "\n")
    models[[m]] <- calculateReturns(models[[m]])
}
printCP(models, to.file)

