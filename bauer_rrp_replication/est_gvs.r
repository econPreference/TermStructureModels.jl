## estimate Gaussian DTSM using block-wise Metropolis-Hastings algorithm
## GVS -- Gibbs Variable Selection

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
priors$g <- 10000

## diagonalized g-prior for parameters
rvar.res <- getLambdaRVAR(rep(1, N*(N+1)), cP=Y%*%t(W), pars$loads$K0Q.cP, pars$loads$K1Q.cP, pars$OmegaInv)
priors$lambda.sd <- sqrt(priors$g)*sqrt(diag(rvar.res$cov.mat))

## use posterior (given opt. values for other params) as pseudo-prior
post.moments <- getCondPostLambda(rvar.res)
priors$lambda.pseudo.mean <- post.moments$mean
priors$lambda.pseudo.sd <- sqrt(diag(post.moments$var))

results.file <- getResultsFileName("gvs_10000", N)

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
prob.gamma <- matrix(NA, M, N+N^2) ## probability of inclusion
count.nonstat <- 0

## pars$lambda     - has excluded elements set to zero
## pars$lambda.all - carries over elements that are currently not in the model
pars$lambda.all <- pars$lambda  ## we start with unrestricted model

cat("Running MCMC algorithm...\n")
for (m in 1:M) {
    if( m %% 1000 == 0) {
        cat("*** iteration ",m,"\n")
        cat("acceptance probabilities:\n")
        cat("thetaQ  => ", round(100*mean(alpha.thetaQ[1:(m-1)])), "\n")
        cat("Omega   => ", round(100*mean(alpha.Omega[1:(m-1)])), "\n")
        cat("current model: ", pars$gamma, "\n")
        cat("current prob. incl.: ", round(100*prob.gamma[m-1,]), "\n")
    }

    pars.save <- pars
    pars <- drawGVS(pars)
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

    ## check that VAR dynamics are stationary
    if(max(abs(eigen(pars$Phi, only.values=TRUE)$values))>1)
        stop("*** EXPLOSIVE EIGENVALUES ***")

}

save(file=results.file, priors, M, N, mats, W, dates, Omega.i, lamQ.i, kinfQ.i, lambda.i, gamma.i, sige2.i, alpha.Omega, alpha.thetaQ, prob.gamma)


