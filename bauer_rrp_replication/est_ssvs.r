## estimate Gaussian DTSM using block-wise Metropolis-Hastings algorithm
## SSVS -- Stochastic Search Variables Selection (George, McCulloch, 1993)

rm(list=ls())
source("R/rrp_functions.r")
source("R/jsz_fns.r")
source("R/estimation_fns.r")
set.seed(616)
init()
pars.mle <- estML(Y, W, mats)
pars <- pars.mle

## priors
priors <- getPriors()

## get cond. posterior SD for lambda at MLE estimates
rvar.res <- getLambdaRVAR(rep(1, N*(N+1)), cP=Y%*%t(W), pars$loads$K0Q.cP, pars$loads$K1Q.cP, pars$OmegaInv)

## SSVS prior
priors$tau1 <-priors$c1 * sqrt(diag(rvar.res$cov.mat))  # for included elements of lambda
priors$tau0 <- priors$c0 * sqrt(diag(rvar.res$cov.mat)) # for excluded elements of lambda
cat("c0 =", priors$c0, " c1 =", priors$c1, "\n")

## look at prior density implied by c0 and c1
cat("c1^2/c0^2 =", priors$c1^2/priors$c0^2, "\n")

results.file <- getResultsFileName("ssvs", N)

########################################
## Metropolis-Hastings algorithm

## number of iterations
M <- 55000

## draws
Omega.i <- matrix(NA, M, N^2)
lambda.i <- matrix(NA, M, (N+1)*N)
gamma.i <- matrix(NA, M, N+N^2)
lamQ.i <- matrix(NA, M, N)
kinfQ.i <- matrix(NA, M, 1)
sige2.i <- matrix(NA, M, 1)

## acceptance probabilities
alpha.Omega <- numeric(M)
alpha.thetaQ <- numeric(M)

## probability of inclusion
prob.gamma <- matrix(NA, M, N+N^2)

count.nonstat <- 0

for (m in 1:M) {
    if( m %% 1000 == 0) {
        cat("*** iteration ",m,"\n")
        cat("acceptance probabilities:\n")
        cat("thetaQ  => ", round(100*mean(alpha.thetaQ[1:(m-1)])), "\n")
        cat("Omega   => ", round(100*mean(alpha.Omega[1:(m-1)])), "\n")
        cat("current model: ", pars$gamma, "\n")
        cat("prob. incl.: ", round(100*colMeans(prob.gamma[(m-19):(m-1),])), "\n")
    }

    pars.save <- pars
    pars <- drawLambdaSSVS(pars)
    pars <- drawGammaSSVS(pars)
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

save(file=results.file, priors, M, N, mats, W, dates, Omega.i, lamQ.i, kinfQ.i, lambda.i, gamma.i, sige2.i, alpha.Omega, alpha.thetaQ, prob.gamma)
