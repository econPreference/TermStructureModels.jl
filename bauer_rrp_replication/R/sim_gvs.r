## GVS on simulated data

rm(list=ls())
options(error=recover)
source("R/rrp_functions.r")
source("R/jsz_fns.r")
source("R/estimation_fns.r")
set.seed(616)

true.pars <- initSimulation(restr = TRUE)

## restr = TRUE runs the simulation using a DGP with restricted risk prices. To instead use a DGP with unrestricted risk prices, use restr = FALSE (and change change character string in call to `getResultsFileName` to include "urp" instead of "rrp"

T <- 300
M <- 50000 	## number of iterations
B <- 5000  	## number of burn-in iterations
n <- 100

priors <- getPriors()

results.file <- getResultsFileName("sim_rrp_gvs", N)

## draws
Omega.i <- matrix(NA, n*M, N^2)
lambda.i <- matrix(NA, n*M, (N+1)*N)
gamma.i <- matrix(NA, n*M, (N+1)*N)
lamQ.i <- matrix(NA, n*M, N)
kinfQ.i <- numeric(n*M)
sige2.i <- numeric(n*M)
## pars.mle <- vector(mode='list', length(n))
Ysim <- vector(mode='list', length(n))

ind <- 0
for (i in 1:n) {
    cat("************************************\n")
    cat(" simulation ", i, "\n")
    cat("************************************\n")

    ## acceptance probabilities
    alpha.Omega <- numeric(B+M)
    alpha.thetaQ <- numeric(B+M)
    prob.gamma <- matrix(NA, B+M, N+N^2)
    count.nonstat <- 0

    Y <- simulateYields(true.pars, T)
    Ysim[[i]] <- Y

    ## starting values
    pars.mle <- estML(Y, W, mats)
    pars <- pars.mle

    ## save ML estimates
    ## pars.mle[[i]] <- pars[c('Omega', 'Sigma', 'lamQ', 'kinfQ', 'mu', 'Phi', 'sige2', 'loads', 'lambda', 'gamma')]

    ## prior for lambda
    rvar.res <- getLambdaRVAR(rep(1, N*(N+1)), cP=Y%*%t(W), pars$loads$K0Q.cP, pars$loads$K1Q.cP, pars$OmegaInv)
    priors$lambda.sd <- sqrt(priors$g)*sqrt(diag(rvar.res$cov.mat))   ## diagonalized g-prior

    ## posterior (given opt. values for other params) as pseudo-prior
    post.mom <- getCondPostLambda(rvar.res)
    priors$lambda.pseudo.mean <- post.mom$mean
    priors$lambda.pseudo.sd <- sqrt(diag(post.mom$var))

    pars$lambda.all <- pars$lambda

    for (m in 1:(B+M)) {
        if( m %% 1000 == 0) {
            cat("*** iteration ",m,"\n")
            cat("acceptance probabilities:\n")
            cat("thetaQ => ", round(100*mean(alpha.thetaQ[1:(m-1)])), "\n")
            cat("Omega  => ", round(100*mean(alpha.Omega[1:(m-1)])), "\n")
            cat("current model: ", pars$gamma, "\n")
            cat("current prob. incl.: ", round(100*prob.gamma[m-1,]), "\n")
        }

        pars.save <- pars
        pars <- drawGVS(pars)
        pars <- drawOmega(pars)
        pars <- drawThetaQ(pars)
        pars <- drawSige2(pars)

        if (max(abs(eigen(pars$Phi, only.values=TRUE)$values))>=1) {
            pars <- pars.save
            count.nonstat <- count.nonstat + 1
        }

        if (m>B) {
            ind <- ind+1
            gamma.i[ind,] <- pars$gamma
            lambda.i[ind,] <- pars$lambda
            Omega.i[ind,] <- as.vector(pars$Omega)
            kinfQ.i[ind] <- pars$kinfQ
            lamQ.i[ind,] <- pars$lamQ
            sige2.i[ind] <- pars$sige2
        }
    }
}

save(file=results.file, true.pars, priors, Ysim, W, B, M, N, T, n, n.per, mats, Omega.i, lamQ.i, kinfQ.i, lambda.i, gamma.i, sige2.i)

