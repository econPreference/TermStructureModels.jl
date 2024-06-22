## estimate Gaussian DTSM using block-wise Metropolis-Hastings algorithm
## RJMCMC - Reversible-Jump Markov Chain Monte Carlo

rm(list=ls())
source("R/rrp_functions.r")
source("R/jsz_fns.r")
source("R/estimation_fns.r")
set.seed(616)
init(N=3)

## starting values
pars.mle <- estML(Y, W, mats)
pars <- pars.mle

## priors
priors <- getPriors()

## diagonalized g-prior for parameters
rvar.res <- getLambdaRVAR(rep(1, N*(N+1)), cP=Y %*% t(W), pars$loads$K0Q.cP, pars$loads$K1Q.cP, pars$OmegaInv)
priors$lambda.sd <- sqrt(priors$g)*sqrt(diag(rvar.res$cov.mat))

results.file <- getResultsFileName("rjmcmc", N)

M <- 55000 ## number of iterations

## draws
Omega.i <- matrix(NA, M, N^2)
lambda.i <- matrix(NA, M, (N+1)*N)
gamma.i <- matrix(NA, M, (N+1)*N)
lamQ.i <- matrix(NA, M, N)
kinfQ.i <- matrix(NA, M, 1)
sige2.i <- matrix(NA, M, 1)

## acceptance probabilities
alpha.Omega <- numeric(M)
alpha.thetaQ <- numeric(M)
alpha.jump <- numeric(M)*NA
count.nonstat <- 0

cat("Running MCMC algorithm...\n")
for (m in 1:M) {
    if( m %% 1000 == 0) {
        cat("*** iteration ",m,"\n")
        cat("acceptance probabilities:\n")
        cat("jump   => ", round(100*mean(alpha.jump[!is.na(alpha.jump)])), "\n")
        cat("thetaQ => ", round(100*mean(alpha.thetaQ[1:(m-1)])), "\n")
        cat("Omega  => ", round(100*mean(alpha.Omega[1:(m-1)])), "\n")
        cat("current model: ", pars$gamma, "\n")
    }

    pars.save <- pars
    pars <- drawJump(pars)
    pars <- drawOmega(pars)
    pars <- drawThetaQ(pars)
    pars <- drawSige2(pars)

    ## if non-stationary, retain previous iteration's values
    if (max(abs(eigen(pars$Phi, only.values=TRUE)$values))>=1) {
        pars <- pars.save
        count.nonstat <- count.nonstat + 1
    }

    gamma.i[m,] <- pars$gamma
    lambda.i[m,] <- pars$lambda
    Omega.i[m,] <- as.vector(pars$Omega)
    kinfQ.i[m] <- pars$kinfQ
    lamQ.i[m,] <- pars$lamQ
    sige2.i[m] <- pars$sige2
}

save(file=results.file, priors, M, N, mats, W, dates, Omega.i, lamQ.i, kinfQ.i, lambda.i, gamma.i, sige2.i, alpha.Omega, alpha.thetaQ, alpha.jump)
