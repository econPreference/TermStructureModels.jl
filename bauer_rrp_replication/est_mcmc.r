## estimate Gaussian DTSM using block-wise Metropolis-Hastings algorithm
## parameterization in terms of RISK PRICES
## theta = (kinfQ, lamQ, Sigma, lam0, lam1, sigw)
## lambda = vec(lam0, lam1)

rm(list=ls())
source("R/rrp_functions.r")
source("R/jsz_fns.r")
source("R/estimation_fns.r")
set.seed(616)

## load data, init parameters
init(N=3)

## model specification
model <- "M3"
gamma <- numeric(N+N^2)
switch(model,
       M0 = {gamma[] <- 1},
       M1 = {gamma[7] <- 1},
       M2 = {gamma[c(1,4,7)] <- 1},
       M3 = {gamma[c(1,7)] <- 1})
cat("model to estimate: ", model, "\n", gamma, "\n")

## get starting values
pars.mle <- estML(Y, W, mats, gamma)
pars <- pars.mle

## priors
priors <- getPriors()

## diagonalized g-prior for parameters
rvar.res <- getLambdaRVAR(rep(1, N*(N+1)), cP=Y%*%t(W), pars$loads$K0Q.cP, pars$loads$K1Q.cP, pars$OmegaInv)
priors$lambda.sd <- sqrt(priors$g)*sqrt(diag(rvar.res$cov.mat))

## show significance
getCondPostLambda(rvar.res)

if (all(gamma==1)) {
    cat("Estimating unrestricted model...\n")
} else {
    cat("Estimating restricted model (", vec2str(gamma), ") ...\n")
}

results.file <- getResultsFileName(paste("mcmc", model, sep="_"), N, gamma)

########################################
## Metropolis-Hastings algorithm

## number of iterations
M <- 15000

## draws
Omega.i <- matrix(NA, M, N^2)
lambda.i <- matrix(NA, M, N+N^2)
lamQ.i <- matrix(NA, M, N)
kinfQ.i <- matrix(NA, M, 1)
sige2.i <- matrix(NA, M, 1)

## acceptance probabilities
alpha.Omega <- numeric(M)
alpha.thetaQ <- numeric(M)

count.nonstat <- 0

cat("Running MCMC algorithm...\n")
tic()
for (m in 1:M) {
    if( m %% 1000 == 0) {
        cat("*** iteration ",m,"\n")
        cat("acceptance probabilities:\n")
        cat("thetaQ => ", round(100*mean(alpha.thetaQ[1:(m-1)])), "\n")
        cat("Omega  => ", round(100*mean(alpha.Omega[1:(m-1)])), "\n")
    }

    pars.save <- pars
    pars <- drawLambda(pars)
    pars <- drawThetaQ(pars)
    pars <- drawOmega(pars)
    pars <- drawSige2(pars)

    ## if non-stationary, retain previous iteration's values
    if (max(abs(eigen(pars$Phi, only.values=TRUE)$values))>=1) {
        pars <- pars.save
        count.nonstat <- count.nonstat + 1
    }

    lambda.i[m,] <- pars$lambda
    Omega.i[m,] <- as.vector(pars$Omega)
    kinfQ.i[m] <- pars$kinfQ
    lamQ.i[m,] <- pars$lamQ
    sige2.i[m] <- pars$sige2
}
toc()

alpha.kinfQ <- alpha.thetaQ
alpha.lamQ <- alpha.thetaQ

save(file=results.file, gamma, priors, M, N, mats, W, dates, Omega.i, lamQ.i, kinfQ.i, lambda.i, sige2.i, alpha.Omega, alpha.thetaQ, alpha.kinfQ, alpha.lamQ)
