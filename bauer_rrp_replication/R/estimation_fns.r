getPriors <- function()
    list(g = 100,  ## Zellner's g-prior
         c0 = 0.1,
         c1 = 10,
         p = rep(.5, N+N^2), ## probability of inclusion
         ## sig.kinfQ = 1/n.per/100,
         lamQ.min = -1,
         lamQ.max = 0)

checkKKT <- function(theta, obj, ...) {
    require(numDeriv)
    y <- obj(theta, ...)
    kkttol <- 10*.Machine$double.eps^(1/4)
    kkt2tol <- 100* (.Machine$double.eps^(1/4))
    ngatend <- grad(obj, theta, method="Richardson", side=NULL, method.args=list(), ...)
    cat("Gradient:")
    print(ngatend)
    kkt1 <- max(abs(ngatend)) <= kkttol*(1.0+abs(y))
    cat("kkt1 = ", kkt1, "\n")
    nhatend <- hessian(obj, theta,  method="Richardson", method.args=list(), ...)
    hev <- eigen(nhatend)$values
    cat("Eigenvalues:", hev, "\n")
    negeig <- (hev <= -kkttol*(1+abs(y)))
    cat("negeig = ", negeig, "\n")
    evratio <- tail(hev, 1)/hev[1]
    cat("evratio =", evratio, "\n")
    cat("evratio requirement >", kkt2tol,"\n")
    kkt2 <- (evratio > kkt2tol) && (!negeig)
    cat("kkt2 =", kkt2, "\n")
}

getOptim <- function(theta, obj, ...) {
    require(ucminf)
    obj <- match.fun(obj)
    cat('Starting optimization...\n')
    cat("Starting values:\n")
    print(theta)
    cat("LLK at starting values:", -obj(theta, ...), "\n")

    cat("1) optimization with Nelder-Mead\n")
    rval <- optim(theta, obj, gr=NULL, ..., control=list(maxit=10000)) ## , parscale=myparscale))
    theta <- rval$par
    if (rval$convergence>0)
        stop("optimization not converged")


    cat("2) optimization with gradient-based algorithm\n")
    rval <- optim(theta, obj, gr=NULL, ..., method = "L-BFGS-B", hessian = TRUE) ## , control=list(parscale=myparscale))
    if (rval$convergence>0) {
        print(rval)
        warnings("optimization not converged - using result of Nelder-Mead")
    } else {
      theta <- rval$par
    }
    cat("LLK at optimum:", -obj(theta, ...), "\n")
    checkKKT(theta, obj, ...)

    theta
}

getCondPostLambda <- function(rvar.res) {
    ## Calculate moments of conditional posterior distribution for lambda
    ## Arguments:
    ##  rvar.res - list with results from restricted VAR estimation
    ## Globals: priors, N
    post.var <- solve( diag(1/priors$lambda.sd^2) + rvar.res$inv.cov.mat )
    post.mean <- post.var %*% rvar.res$inv.cov.mat %*% rvar.res$lambda.hat

    out.matrix <- matrix(NA, N+N^2, 5)
    rownames(out.matrix) <- c(sapply(1:N, function(x) paste('lam0_', as.character(x), sep='')), sapply( rep((1:N),N)*10 + as.numeric(gl(N, N)), function(x) paste('Lam1_', as.character(x), sep='')))
    colnames(out.matrix) <- c('mean', 't-stat', 'LB', 'UB', 'sig 5%')
    out.matrix[,1] <- post.mean
    out.matrix[,2] <- post.mean/sqrt(diag(post.var))
    out.matrix[,3] <- post.mean-1.96*sqrt(diag(post.var))
    out.matrix[,4] <- post.mean+1.96*sqrt(diag(post.var))
    out.matrix[,5] <- sign(out.matrix[,3])*sign(out.matrix[,4])==1
    print(round(out.matrix,digi=4))

    return(list(mean=post.mean, var=post.var))
}

############################################################################
############################################################################
### MLE

obj.mle <- function(theta, Y, W, mats, gamma) {
    ## objective function is sum of negative log likelihoods
    pars <- theta2pars(theta)

    ## check restrictions on parameter space
    valid <- TRUE
    ## diagonal elements of Sigma positive and bounded away from zero
    if (any(diag(pars$Sigma)<1e-7)) valid <- FALSE
    if (any(diag(pars$Sigma)>1)) valid <- FALSE
    ## eigenvalues of Phi.Q not explosive
    if (any(pars$lamQ>0)) valid <- FALSE

    ## eigenvalues sorted
    if (any(pars$dlamQ>0)) valid <- FALSE

    ## if parameters satisfy restriction on param space
    if (valid) {
        ## evaluate likelihood function and return sum of negative logliks
        if (missing(gamma)) {
            warning("did not provide gamma")
            res.llk <- jsz.llk(Y, W, K1Q.X=diag(pars$lamQ), Sigma.cP=pars$Omega, mats=mats, dt=1)
        } else {
            res.llk <- jsz.llk(Y, W,K1Q.X=diag(pars$lamQ), Sigma.cP=pars$Omega, mats=mats, dt=1, restr=1, ind.restr=gamma)
        }

        return(sum(res.llk$llk))
    } else {
        ## else return penalty value
        return(1e6)
    }
}

scale.dlamQ <- 100
scale.Sigma <- 50000

theta2pars <- function(theta) {
    ## convert theta vector to list of individual parameters
    ## Globals: uses N
    ## Q parameters
    pars <- list(dlamQ=theta[1:N]/scale.dlamQ)
    pars$lamQ=cumsum(pars$dlamQ)
    ## P-innovation covariance matrix
    pars$Sigma <- matrix(0,N,N)
    pars$Sigma[lower.tri(pars$Sigma,diag=TRUE)] <- tail(theta,-N)/scale.Sigma
    pars$Omega <- pars$Sigma %*% t(pars$Sigma)
    return(pars)
}

pars2theta <- function(pars) {
    ## convert list of individual parameters to theta vector
    ## Globals: uses N
    dlamQ <- c(pars$lamQ[1],diff(pars$lamQ));
    if (length(pars$lamQ)!=N) stop("lamQ has wrong length")
    Sigma.vec <- pars$Sigma[lower.tri(pars$Sigma,diag=TRUE)]
    return(c(dlamQ*scale.dlamQ, Sigma.vec*scale.Sigma))
}

estML <- function(Y, W, mats, gamma) {
    ## Obtain MLE for affine Gaussian DTSM
    ## (plus additional model derived parameters used in MCMC sampler)
    ## Arguments:
    ##  gamma -- risk price specification: vector with length N+N^2 indicating which elements of lambda are unrestricted (1) or restricted to zero (0)
    ## Value:
    ##  pars -- list with parameter estimates

    getStartingValuesForMLE <- function(Sigma) {
        ## obtain starting values for lamQ for MLE
        ##  -> random seeds for Q-eigenvalues
        ##
        ## Arguments: Sigma
        ## Value: list with starting values

        if (missing(Sigma))
            error("Sigma needs to be provided")
        nSeeds <- 100;  # how many random seeds to try
        best.llk <- Inf
        Omega <- Sigma %*% t(Sigma)
        for (i in 1:nSeeds) {
            ##    lamQ <- -sort(abs(.01*rnorm(N)))
            (lamQ <- -abs(sort(runif(N, 0, .1))))
            res.llk <- jsz.llk(Y, W, K1Q.X=diag(lamQ), Sigma.cP=Omega, mats=mats, dt=1, restr=1, ind.restr=gamma)
            llk <- sum(res.llk$llk)
            if (llk<best.llk) {
                cat('Improved seed llk to ', llk, '\n')
                best.llk <- llk
                best.lamQ <- lamQ
            }
        }
        return(list(lamQ=best.lamQ, Sigma=Sigma))
    }

    cP <- Y %*% t(W)
    N <- nrow(W)
    J <- ncol(Y)
    if (missing(mats)||length(mats)!=J)
        stop("estML: mats needs to be provided and have length consistent with yield data")
    if (missing(W)||ncol(W)!=J)
        stop("estML: W needs to be provided and have dimensions consistent with yield data")
    if (missing(gamma))
        gamma <- rep(1, N*(N+1))

    cat("*** MLE ***\n")

    ## (1) estimate VAR -- just to get Sigma.hat
    lm <- ar.ols(cP, aic=FALSE, order.max=1, intercept=TRUE, demean=FALSE)
    Omega.hat <- lm$var.pred
    Sigma.hat <- t(chol(Omega.hat))

    ## (2) numerical optimization
    pars.start <- getStartingValuesForMLE(Sigma.hat)
    theta.start <- pars2theta(pars.start)
    theta <- getOptim(theta.start, obj.mle, Y, W, mats, gamma)
    pars <- theta2pars(theta)  ## lamQ, Omega
    ## value of likelihood
    res.llk <- jsz.llk(Y, W, K1Q.X=diag(pars$lamQ), Sigma.cP=pars$Omega, mats=mats, dt=1, restr=1, ind.restr=gamma)
    pars$kinfQ <- res.llk$K0Q.X[which(res.llk$K0Q.X!=0)]
    pars$mu <- res.llk$K0P.cP
    pars$Phi <- res.llk$K1P.cP + diag(N)
    pars$sige2 <- res.llk$sigma.e^2
    res.llkQ <- getLlkQ(Y, W, mats, pars$lamQ, pars$kinfQ, pars$Omega, pars$sige2)
    pars$loads <- res.llkQ$loads
    pars$lambda <- getRP(pars$loads, pars$mu, pars$Phi)$lambda
    ## remaining starting values needed
    pars$gamma <- gamma
    pars$lambda[pars$gamma==0] <- 0  ## make elements EXACTLY zero
    pars$OmegaInv <- solve(pars$Omega)
    pars$errors <- res.llkQ$errors
    pars$llkQ <- res.llkQ$llkQ
    pars$llkP <- getLlkP(t(cP), pars$mu, pars$Phi, pars$Omega, pars$OmegaInv)
    return(pars)
}

############################################################################
############################################################################

drawLambda <- function(pars) {
    ## prices of risk
    ## lambda = (lambda_0, lambda_1)
    ## draw jointly using Gibbs step
    ##
    ## Arguments:
    ##  pars - list with current model parameters
    ##
    ## Value:
    ##  pars - list with updated parameters
    ##
    ## Globals: Y, W, priors

    cP <- Y %*% t(W)

    if (sum(pars$gamma)>0) {
        ## at least one parameter unrestricted
        rvar.res <- getLambdaRVAR(pars$gamma, cP, pars$loads$K0Q.cP, pars$loads$K1Q.cP, pars$OmegaInv)
        if (sum(pars$gamma)>1) {
            ## more than one parameter unrestricted
            lambda.var <- solve(diag(1/priors$lambda.sd[pars$gamma==1]^2) + rvar.res$inv.cov.mat)
        } else {
            ## only one parameter unrestricted -- scalar
            lambda.var <- 1/(1/priors$lambda.sd[pars$gamma==1]^2 + rvar.res$inv.cov.mat)
        }
        lambda.mean <- lambda.var %*% rvar.res$inv.cov.mat %*% rvar.res$lambda.hat
        lambda.draw <- lambda.mean + t(chol(lambda.var)) %*% rnorm(sum(pars$gamma))
        pars$lambda <- rvar.res$R %*% lambda.draw
    } else {
        pars$lambda <- numeric(N+N^2)
    }

    Pdyn <- getPdyn(pars$loads, pars$lambda)
    pars$mu <- Pdyn$mu; pars$Phi <- Pdyn$Phi
    pars$llkP <- getLlkP(t(cP), Pdyn$mu, Pdyn$Phi, pars$Omega, pars$OmegaInv)

    return(pars)
}

############################################################################
############################################################################

drawLambdaSSVS <- function(pars) {
    ## prices of risk
    ## lambda = (lambda_0, lambda_1)
    ## draw jointly using Gibbs step
    ## given gamma = current parameter restrictions
    ## hierarchical prior according to SSVS
    ##
    ## Arguments:
    ##  pars - list with current model parameters
    ##
    ## Value:
    ##  pars - list with updated parameters
    ##
    ## Globals: Y, W, N, alpha.lambda, m

    cP <- Y %*% t(W)

    rvar.res <- getLambdaRVAR(rep(1, N*(N+1)), cP, pars$loads$K0Q.cP, pars$loads$K1Q.cP, pars$OmegaInv)
    lambda.hat <- rvar.res$lambda.hat ## least-squares estimate

    ## prior variance
    ## (I) prior conditional independence, use tau0 and tau1 as prior standard deviation
    if ("tau0" %in% names(priors)) {
        ## choose high or low prior SD, depending on whether included or excluded
        tau <- priors$tau0*(1-pars$gamma) + priors$tau1*(pars$gamma)
        ## invert and put into matrix D^-1
        D.inv <- diag(1/tau)
        R <- diag( N*(N+1) ); ## prior conditional independence
        R.inv <- R
    } else {
        stop("need to focus on case with prior conditional independence")
        ## (II) g-prior
        g <- priors$c0*(1-pars$gamma) + priors$c1*(pars$gamma)
        D.inv <- diag(1/g)
        pars$R.inv <- rvar.res$inv.cov.mat
    }
    DRD.inv <- D.inv %*% R.inv %*% D.inv

    if (any(is.na(DRD.inv)))
        stop("some elements of DRD^-1 are NA!?\n")

    ## posterior variance
    lambda.var <- solve( rvar.res$inv.cov.mat + DRD.inv )
    ## posterior mean
    lambda.mean <- lambda.var %*% rvar.res$inv.cov.mat %*% lambda.hat
    ## draw from conditional posterior
    lambda.new <- lambda.mean + t(chol(lambda.var)) %*% rnorm(N*(N+1))

    Pdyn <- getPdyn(pars$loads, lambda.new)
    pars$lambda <- lambda.new
    pars$mu <- Pdyn$mu; pars$Phi <- Pdyn$Phi
    pars$llkP <- getLlkP(t(cP), Pdyn$mu, Pdyn$Phi, pars$Omega, pars$OmegaInv)

    return(pars)
}

drawGammaSSVS <- function(pars) {
    ## draw vector of indicators using Gibbs step
    ## conditional posterior is Bernoulli
    ##  -> does not depend on the data
    ##
    ## Arguments:
    ##  pars - current model parameters
    ##
    ## Value:
    ##  new model parameters (only gamma changed)
    ##
    ## Globals: N, priors, m, priors, prob.gamma (probability of inclusion saved for tracking)
    ## Side effects: changes prob.gamma[m,]

    ## assumptions:
    ## - prior on gamma is independent Bernoulli
    ## - no other prior parameters depend on gamma


    for (i in sample.int(N+N^2)) {
        ## sample elements of gamma consecutively, in random order
        if ("tau0" %in% names(priors)) {
            ## (I) lambda | gamma is independent (R = I)
            a <- priors$tau1[i]^-1*exp(-.5*pars$lambda[i]^2/priors$tau1[i]^2)*priors$p[i]
            b <- priors$tau0[i]^-1*exp(-.5*pars$lambda[i]^2/priors$tau0[i]^2)*(1-priors$p[i])
        } else {
            ## (II) lambda | gamma, Omega is multivariate normal
            stop("focus on prior conditional independence")
            gamma.1 <- pars$gamma; gamma.1[i] <- 1
            g1 <- priors$c0*(1-gamma.1) + priors$c1*(gamma.1)
            D1.inv <- diag( 1/g1 )
            D1RD1.inv <- D1.inv %*% pars$R.inv %*% D1.inv
            gamma.0 <- pars$gamma; gamma.0[i] <- 0
            g0 <- priors$c0*(1-gamma.0) + priors$c1*(gamma.0)
            D0.inv <- diag( 1/g0 )
            D0RD0.inv <- D0.inv %*% pars$R.inv %*% D0.inv
            a <- sqrt(det(D1RD1.inv))*exp(-.5*t(pars$lambda)%*%D1RD1.inv%*%pars$lambda)*priors$p[i]
            b <- sqrt(det(D0RD0.inv))*exp(-.5*t(pars$lambda)%*%D0RD0.inv%*%pars$lambda)*(1-priors$p[i])
        }
        prob.gamma[m,i] <<- a/(a+b)
        pars$gamma[i] <- runif(1)< (a/(a+b))
    }

    if (any(is.na(pars$gamma))) {
        cat("some gamma's are NA!?\n")
        browser()
    }
    return(pars)
}

############################################################################
############################################################################

drawGVS <- function(pars) {
    ## draw pairs of (lambda_i, gamma_i)
    ## using Gibbs Variable Selection
    ##
    ## Arguments:
    ##  pars - current model parameters
    ##
    ## Value:
    ##  new model parameters
    ##
    ## Globals: Y, W, priors, prob.gamma, m (probability of inclusion saved for tracking)
    ## Side effects: changes prob.gamma[m,]

    ## assumptions:
    ## - prior on gamma is independent Bernoulli
    ## - no other prior parameters depend on gamma
    ## - conditional prior independence of lambda

    cP <- Y %*% t(W)

    ## 1. draw included elements from conditional posterior
    if (sum(pars$gamma)>0) {
        ## at least one parameter unrestricted
        rvar.res <- getLambdaRVAR(pars$gamma, cP, pars$loads$K0Q.cP, pars$loads$K1Q.cP, pars$OmegaInv)
        if (sum(pars$gamma)>1) {
            ## more than one parameter unrestricted
            lambda.var <- solve(diag(1/priors$lambda.sd[pars$gamma==1]^2) + rvar.res$inv.cov.mat)
        } else {
            ## only one parameter unrestricted -- scalar
            lambda.var <- 1/(1/priors$lambda.sd[pars$gamma==1]^2 + rvar.res$inv.cov.mat)
        }
        lambda.mean <- lambda.var %*% rvar.res$inv.cov.mat %*% rvar.res$lambda.hat
        lambda.draw <- lambda.mean + t(chol(lambda.var)) %*% rnorm(sum(pars$gamma))
        pars$lambda <- rvar.res$R %*% lambda.draw
    } else {
        pars$lambda <- numeric(N+N^2)
    }
    pars$lambda.all <- pars$lambda
    Pdyn <- getPdyn(pars$loads, pars$lambda)
    pars$mu <- Pdyn$mu; pars$Phi <- Pdyn$Phi
    pars$llkP <- getLlkP(t(cP), Pdyn$mu, Pdyn$Phi, pars$Omega, pars$OmegaInv)

    ## 2. draw excluded elements from pseudo-prior
    pars$lambda.all[pars$gamma==0] <- priors$lambda.pseudo.mean[pars$gamma==0] + priors$lambda.pseudo.sd[pars$gamma==0]*rnorm(sum(pars$gamma==0))

    ## 3. draw elements of gamma, in random order
    for (i in sample.int(N+N^2)) {
        ## (ii) draw gamma_i | lambda_i
        lambda.incl <- pars$lambda
        lambda.incl[i] <- pars$lambda.all[i]
        Pdyn.incl <- getPdyn(pars$loads, lambda.incl)

        lambda.excl <- pars$lambda
        lambda.excl[i] <- 0
        Pdyn.excl <- getPdyn(pars$loads, lambda.excl)

        ## log-likelihood given gamma_i = 1
        llkP.incl <- getLlkP(t(cP), Pdyn.incl$mu, Pdyn.incl$Phi, pars$Omega, pars$OmegaInv)
        ## log-likelihood given gamma_i = 0
        llkP.excl <- getLlkP(t(cP), Pdyn.excl$mu, Pdyn.excl$Phi, pars$Omega, pars$OmegaInv)

        ## prior of lambda_i (gamma_i=1) / pseudo-prior (gamma_i=0)
        prior.ratio <- priors$lambda.sd[i]^-1*exp(-.5*pars$lambda.all[i]^2/priors$lambda.sd[i]^2)/priors$lambda.pseudo.sd[i]^-1/exp(-.5*(pars$lambda.all[i]-priors$lambda.pseudo.mean[i])^2/priors$lambda.pseudo.sd[i]^2)

        ## P(gamma(i)=1|Y,gamma(-i),lambda,...) / P(gamma(i)=0|Y,gamma(-i),lambda,...)
        q <- exp(llkP.incl-llkP.excl)*prior.ratio*priors$p[i]/(1-priors$p[i])
        incl.prob <- ifelse(is.infinite(q), 1, q/(q+1))

        if (runif(1) < incl.prob) {
            ## include element i
            pars$gamma[i] <- 1
            pars$lambda <- lambda.incl
            pars$mu <- Pdyn.incl$mu; pars$Phi <- Pdyn.incl$Phi
            pars$llkP <- llkP.incl
        } else {
            ## exclude element i
            pars$gamma[i] <- 0
            pars$lambda <- lambda.excl
            pars$mu <- Pdyn.excl$mu; pars$Phi <- Pdyn.excl$Phi
            pars$llkP <- llkP.excl
        }

        prob.gamma[m,i] <<- incl.prob  ## P(gamma_i=1 | Y, ...)
    }
    return(pars)
}


############################################################################
############################################################################

getLambda.i <- function(i, pars) {
    ## Globals: Y, W, N
    cP <- Y %*% t(W)
    T <- nrow(cP) - 1

    ## draw lambda[i], given all other values of lambda/gamma
    ## -> approach suggested in Geweke and Kuo/Mallick to draw individual elements of lambda
    j <- ((i-1) %% N)+1 ## which equation
    ## get error variance
    sigm2 <- pars$Omega[j,j] - t(pars$Omega[j,-j]) %*% solve(pars$Omega[-j,-j]) %*% pars$Omega[j,-j]
    ## dependent variable in equation j
    ydat <- cP[2:(T+1),j] - rep(pars$loads$K0Q.cP[j],T) - cP[1:T,]%*%(pars$loads$K1Q.cP+diag(N))[j,]
    ## subtract out other terms that we condition on
    ind.row <- matrix(1:(N^2+N), N, N+1)[j,]
    ind.other <- setdiff(ind.row, i)
    lam.other <- pars$lambda[ind.other]
    xdat.row <- cbind(1, cP[1:T,]) ## all regressors
    xdat.other <- xdat.row[,match(ind.other, ind.row)] ## only other regressors
    zdat <- ydat - xdat.other %*% lam.other
    ## OLS estimate
    xdat <- xdat.row[,match(i, ind.row)]
    b.ols <- crossprod(zdat, xdat)/crossprod(xdat)
    ## posterior mean and variance
    omega2 <- sigm2/crossprod(xdat)
    sig2.post <- 1/(1/omega2 + 1/priors$lambda.sd[i]^2)
    lami.post <- sig2.post*b.ols/omega2 ## since prior mean is zero
    ## draw from posterior
    return(list(lami = lami.post + sqrt(sig2.post)*rnorm(1), lami.post=lami.post, sig2.post=sig2.post))
}

drawJump <- function(pars) {
    ## RJMCMC -- jump step
    ## given gamma = current parameter restrictions
    ##
    ## Arguments:
    ##  pars - list with current model parameters
    ##
    ## Value:
    ##  pars - list with updated parameters
    ##
    ## Globals: Y, W, N, alpha.jump, m
    ## Side effects: alpha.jump[m]

    cP <- Y %*% t(W)

    ## draw proposed model
    p.null <- 0.25  ## probability of null move
    if (runif(1)<p.null) {
        ## within-model move
        pars <- drawLambda(pars)
    } else {
        ## jump to other model
        gamma.prop <- pars$gamma
        ind <- sample.int(N+N^2,1)    ## pick one element randomly

        post.moments <- getLambda.i(ind, pars)
        u.mean <- post.moments$lami.post
        u.sd <- sqrt(post.moments$sig2.post)
        if (pars$gamma[ind]==0) {
            ## switch on -- dim(lambda) < dim(lambda')
            gamma.prop[ind] <- 1
            ## lambda' = g(lambda, u)  -- u is random scalar
            lambda.prop <- pars$lambda
            if (lambda.prop[ind]!=0)
                stop("should be zero")
            u <- u.mean + u.sd*rnorm(1)
            lambda.prop[ind] <- u

            ## ratio of priors is f(u) (other priors cancel out)
            ratio.prior <- priors$lambda.sd[ind]^-1*exp(-.5*u^2/priors$lambda.sd[ind]^2)

            ## ratio of proposals is 1/q(u) (jump proposal cancels out)
            ratio.prop <- 1/( u.sd^-1*exp(-.5*(u - u.mean)^2/u.sd^2) )
        } else {
            ## switch off -- dim(lambda) > dim(lambda')
            gamma.prop[ind] <- 0
            lambda.prop <- pars$lambda
            lambda.prop[ind] <- 0
            u.prime <- pars$lambda[ind]

            ## ratio of priors is 1/f(u) (other priors cancel out)
            ratio.prior <- 1/(priors$lambda.sd[ind]^-1*exp(-.5*u.prime^2/priors$lambda.sd[ind]^2))

            ## ratio of proposals is q(u) (jump proposal cancels out)
            ratio.prop <- u.sd^-1*exp(-.5*(u.prime - u.mean)^2/u.sd^2)
        }

        ## acceptance probability
        Pdyn.prop <- getPdyn(pars$loads, lambda.prop);
        llkP.prop <- getLlkP(t(cP), Pdyn.prop$mu, Pdyn.prop$Phi, pars$Omega, pars$OmegaInv)
        llr.P <- llkP.prop - pars$llkP
        alpha.jump[m] <<- min(exp(llr.P)*ratio.prior*ratio.prop, 1)

        if (runif(1)<alpha.jump[m]) {
            pars$gamma <- gamma.prop
            pars$lambda <- lambda.prop
            pars$mu <- Pdyn.prop$mu; pars$Phi <- Pdyn.prop$Phi
            pars$llkP <- llkP.prop
        }
    }
    return(pars)
}

############################################################################
############################################################################

drawOmega <- function(pars) {
    ## draw Omega -- variance-covariance matrix
    ## draw Sigma using Independence Proposal
    ##
    ## Arguments:
    ##  pars - list with current model parameters
    ##
    ## Value:
    ##   pars - list with updated model parameters
    ##
    ## Globals: N, Y, W, mats, alpha.Omega, m
    ## Side effects: changes alpha.Omega[m]
    require(mvtnorm)  # for rmvt
    require(numDeriv) # for hessian

    cP <- Y %*% t(W)
    ltri <- lower.tri(matrix(NA, N, N), diag=TRUE)
    obj.Sigma <- function(vSigma) {
        ## value of neg. log cond. posterior -- due to flat prior this
        ## is just log likelihood (unless prior restrictions on
        ## eigenvalues are violated)
        Sigma <- matrix(0,N,N)
        Sigma[ltri] <- vSigma  ##/scale.Sigma
        Omega <- Sigma %*% t(Sigma)
        res.llkQ <- getLlkQ(Y, W, mats, pars$lamQ, pars$kinfQ, Omega, pars$sige2)
        llkQ <- res.llkQ$llkQ
        Pdyn <- getPdyn(res.llkQ$loads, pars$lambda);
        llkP <- getLlkP(t(cP), Pdyn$mu, Pdyn$Phi, Omega, solve(Omega))
        -(llkQ + llkP)
    }
    cP <- Y %*% t(W)

    ## independence proposal
    vSigma.mean <- as.numeric(pars.mle$Sigma[ltri])
    ## variance: use Hessian of cond. posterior
    if (!("vSigma.var" %in% names(pars)))
        pars$vSigma.var <- makePD(solve(hessian(obj.Sigma, vSigma.mean)))
    nu <- 5
    scale.mat <- pars$vSigma.var*(nu-2)/nu
    vSigma.prop <- as.numeric(vSigma.mean + rmvt(1, sigma=scale.mat, df=nu))
    Sigma.prop <- matrix(0, N, N)
    Sigma.prop[ltri] <- vSigma.prop
    Omega.prop <- Sigma.prop %*% t(Sigma.prop)

    ## calculate acceptance probability
    vSigma.current <- as.numeric(pars$Sigma[ltri])
    ## (1) prior - make sure it's a covariance matrix
    if (all(eigen(Omega.prop)$values > 0)) {
        ## (2) Q-likelihood
        llkQ.prop <- getLlkQ(Y, W, mats, pars$lamQ, pars$kinfQ, Omega.prop, pars$sige2)
        llr.Q <- llkQ.prop$llkQ - pars$llkQ
        ## get P-dynamics implied by proposal
        Pdyn.prop <- getPdyn(llkQ.prop$loads, pars$lambda);
        ## (3) log-ratio P-likelihoods
        llkP.prop <- getLlkP(t(cP), Pdyn.prop$mu, Pdyn.prop$Phi, Omega.prop, solve(Omega.prop))
        llr.P <- llkP.prop - pars$llkP
        ## (4) log-ratio of proposals
        lr.prop <- dmvt(vSigma.current, sigma = scale.mat, df = nu, log = TRUE) -
            dmvt(vSigma.prop, sigma = scale.mat, df = nu, log = TRUE)
        alpha.Omega[m] <<- min(exp(llr.Q + llr.P + lr.prop), 1)
    } else {
        alpha.Omega[m] <<- 0
    }
    if (runif(1)<alpha.Omega[m]) {
        pars$Omega <- Omega.prop
        pars$OmegaInv <- solve(Omega.prop)
        pars$loads <- llkQ.prop$loads
        pars$errors <- llkQ.prop$errors
        pars$llkQ <- llkQ.prop$llkQ
        pars$mu <- Pdyn.prop$mu; pars$Phi <- Pdyn.prop$Phi
        pars$llkP <- llkP.prop
    }
    pars
}

############################################################################
############################################################################

drawThetaQ <- function(pars) {
    ## draws kinfQ and lamQ using Independence Proposal
    ##
    ## Arguments:
    ##  pars - list of current model parameters
    ##
    ## Value:
    ##  pars - list of updated model parameters
    ##
    ## Globals: Y, W, mats, alpha.thetaQ, m
    require(mvtnorm)  # for rmvt
    require(numDeriv) # for hessian

    scale.kinfQ <- 1000

    obj.thetaQ <- function(thetaQ) {
        ## value of neg. log cond. posterior -- due to flat prior this
        ## is just log likelihood (unless prior restrictions on
        ## eigenvalues are violated)
        kinfQ <- thetaQ[1]/scale.kinfQ
        dlamQ <- thetaQ[2:(N+1)]
        lamQ <- cumsum(dlamQ)
        if (all(dlamQ<0, lamQ<priors$lamQ.max, lamQ>priors$lamQ.min)) {
            res.llkQ <- getLlkQ(Y, W, mats, lamQ, kinfQ, pars$Omega, pars$sige2)
            llkQ <- res.llkQ$llkQ
            Pdyn <- getPdyn(res.llkQ$loads, pars$lambda);
            llkP <- getLlkP(t(cP), Pdyn$mu, Pdyn$Phi, pars$Omega, pars$OmegaInv)
            -(llkQ + llkP)
        } else {
            1e6
        }
    }
    cP <- Y %*% t(W)

    ## independence proposal
    dlamQ.mean <- c(pars.mle$lamQ[1], diff(pars.mle$lamQ))
    thetaQ.mean <- c(pars.mle$kinfQ * scale.kinfQ, dlamQ.mean)
    ## variance: use Hessian of cond. posterior
    if (!("thetaQ.var" %in% names(pars)))
        pars$thetaQ.var <- makePD(solve(hessian(obj.thetaQ, thetaQ.mean)))
    nu <- 5
    scale.mat <- pars$thetaQ.var*(nu-2)/nu
    thetaQ.prop <- as.numeric(thetaQ.mean + rmvt(1, sigma=scale.mat, df=nu))

    kinfQ.prop <- thetaQ.prop[1]/scale.kinfQ
    dlamQ.prop <- thetaQ.prop[2:(N+1)]
    lamQ.prop <- cumsum(dlamQ.prop)

    ## calculate acceptance probability
    dlamQ.current <- c(pars$lamQ[1], diff(pars$lamQ))
    thetaQ.current <- c(pars$kinfQ * scale.kinfQ, dlamQ.current)
    ## (1) prior
    if (all(dlamQ.prop<0, lamQ.prop<priors$lamQ.max, lamQ.prop>priors$lamQ.min)) {
        ## (2) Q-likelihood
        llkQ.prop <- getLlkQ(Y, W, mats, lamQ.prop, kinfQ.prop, pars$Omega, pars$sige2)
        llr.Q <- llkQ.prop$llkQ - pars$llkQ
        ## get P-dynamics implied by proposal
        Pdyn.prop <- getPdyn(llkQ.prop$loads, pars$lambda);
        ## (3) log-ratio P-likelihoods
        llkP.prop <- getLlkP(t(cP), Pdyn.prop$mu, Pdyn.prop$Phi, pars$Omega, pars$OmegaInv)
        llr.P <- llkP.prop - pars$llkP
        ## (4) log-ratio of proposals
        lr.prop <- dmvt(thetaQ.current, sigma = scale.mat, df = nu, log = TRUE) -
            dmvt(thetaQ.prop, sigma = scale.mat, df = nu, log = TRUE)
        alpha.thetaQ[m] <<- min(exp(llr.Q + llr.P + lr.prop), 1)
    } else {
        alpha.thetaQ[m] <<- 0
    }

    if (runif(1) < alpha.thetaQ[m]) {
        pars$kinfQ <- thetaQ.prop[1]/scale.kinfQ
        pars$lamQ <- cumsum(thetaQ.prop[2:(1+N)])
        pars$loads <- llkQ.prop$loads
        pars$errors <- llkQ.prop$errors
        pars$llkQ <- llkQ.prop$llkQ
        pars$mu <- Pdyn.prop$mu; pars$Phi <- Pdyn.prop$Phi
        pars$llkP <- llkP.prop
    }
    return(pars)
}

drawSige2 <- function(pars) {
    ## draw measurement error variance
    ## Gibbs step -- pooled linear regression, conjugate prior
    ##
    ## Arguments:
    ##  pars - current model parameters
    ##
    ## Value:
    ##  pars - updated model parameters
    ##
    ## Globals: Y, W
    J <- ncol(W)
    N <- nrow(W)

    ## prior: inverse gamma
    ## uninformative prior:
    alpha.0 <- 0
    delta.0 <- 0

    ssr <- sum(pars$errors^2)
    alpha.1 <- alpha.0 + (J-N)*nrow(Y)  ## only J-N independent linear combinations!
    delta.1 <- delta.0 + ssr

    pars$sige2 <- 1/rgamma(n=1,shape=alpha.1/2,rate=delta.1/2)
    pars$llkQ <- -.5*sum(pars$errors^2)/pars$sige2

    return(pars)
}

estAllModels <- function(Lam0.free=TRUE, kinfQ=TRUE) {
    ## estimate ALL models using MLE
    ## - take all parameters other than gamma as given (MLE estimates)
    source("R/rrp_functions.r", local=TRUE)
    source("R/estimation_fns.r", local=TRUE)
    if (!kinfQ)
        source("R/rinfQ_fns.r")
    set.seed(616)
    ## load data, init parameters -- creates global cariables
    init(N=3)
    cP <- Y %*% t(W)
    T <- nrow(cP)
    J <- length(mats)

    ## get MLE for unrestricted model
    ## (all parameters will be fixed at these values except for lambda/mu/Phi)
    if (kinfQ) {
        pars <- estML(Y, W, mats, rep(1, N+N^2))
    } else {
        pars <- estML.rinfQ(Y, W, mats, rep(1, N+N^2))
    }
    A <- pars$loads$AcP; B <- pars$loads$BcP

    Yhat <- rep(1,T)%*%A + cP%*%B
    cat("RMSE = ", 10000*n.per*sqrt( mean((Y-Yhat)^2) ),"\n")

    ## show persistence under Q
    ## (this will be the same for all models)
    PhiQ <- diag(N)+pars$loads$K1Q.cP
    cat("PhiQ[1,1] = ", PhiQ[1,1], ";",
    "maxev-Q = ", max(abs(eigen(PhiQ)$values)), ";",
    "IRF-Q = ",  irf.var1(PhiQ, 120)[120], "\n")
    if (!kinfQ)
        cat("rinfQ =", 1200*pars$rinfQ, "\n")

    cols <- 14
    if (Lam0.free) {
        K <- 2^12 ## all models
        models <- matrix(NA, K, cols)
        rownames(models) <- sapply(1:K, function(k) {
            gamma <- as.numeric(intToBits(k-1)[1:12])
            return(paste(gamma, collapse=""))
        })
    } else {
        K <- 2^9 ## only models with Lam0 = (1,1,1)
        models <- matrix(NA, K, cols)
        rownames(models) <- sapply(1:K, function(k) {
            gamma <- c(1,1,1, as.numeric(intToBits(k-1)[1:9]))
            return(paste(gamma, collapse=""))
        })
    }

    colnames(models) <- c("Phi11", "maxev-P", "IRF-P", "AIC", "BIC",
                          "E.r", "E.y", "sig.y", "sig.dy", "sig.dyrn", "sighat.dyrn",
                          "EP_1", "EP_2", "EP_3")
    for (k in 1:K) {
        if( k %% 500 == 0)
            cat("*** model", k, "out of", K, "\n")
        gamma <- as.numeric(strsplit(rownames(models)[k], "")[[1]])
        if (kinfQ) {
            res.llk <- jsz.llk(Y, W, K1Q.X=diag(pars$lamQ),
                               Sigma.cP=pars$Omega, mats=mats, dt=1, restr=1, ind.restr=gamma)
        } else {
            res.llk <- jsz.llk.rinfQ(Y, W, K1Q.X=diag(pars$lamQ), rinfQ=pars$rinfQ,
                                     Sigma.cP=pars$Omega, mats=mats, dt=1, restr=1, ind.restr=gamma)
        }
        numparam <- 1+N+N*(N+1)/2+sum(gamma)
        models[k,4] <- 2*sum(res.llk$llk) + 2*numparam
        models[k,5] <- 2*sum(res.llk$llk) + numparam*log(nrow(Y))
        Phi <- res.llk$K1P.cP + diag(N)
        mu <- res.llk$K0P.cP
        models[k,1] <- Phi[1,1]
        models[k,2] <- max(abs(eigen(Phi)$values))
        ## continue only if stationary
        if (models[k,2]<1) {
            models[k,3] <- irf.var1(Phi, 120)[120]
            ## population moments
            EcP <- as.numeric(solve(diag(N) - Phi) %*% mu)
            models[k,6] <- 1200*(res.llk$rho0.cP + crossprod(res.llk$rho1.cP, EcP))
            models[k,7] <- 1200*(A[J] + EcP%*%B[,J])
            VarcP <- matrix( solve(diag(N^2) - kronecker(Phi, Phi))%*%as.numeric(pars$Omega), N, N)
            models[k,8] <- 1200*sqrt(t(B[,J]) %*% VarcP %*% B[,J])
            VardcP <- (diag(N) - Phi) %*% VarcP %*% t(diag(N) - Phi) + pars$Omega
            models[k,9] <- 1200*sqrt(t(B[,J]) %*% VardcP %*% B[,J])
            ## vol of risk-neutral rates
            loads.rn <- gaussian.loadings(mats, mu, Phi-diag(N), pars$Omega, pars$loads$rho0, pars$loads$rho1)
            Brn <- loads.rn$B
            models[k,10] <- 1200*sqrt(t(Brn[,J]) %*% VardcP %*% Brn[,J])
            models[k,11] <- 1200*sqrt(t(Brn[,J]) %*% cov(cP[2:T,]-cP[1:(T-1),]) %*% Brn[,J])
            models[k, 12:14] <- EcP
        }
    }
    order.aic <- order(models[,4])
    order.bic <- order(models[,5])
    cat("Unrestricted and best restricted models:\n")
    print(round(models[c(K, order.bic[1:20]), 1:10], digi=4))

    ## mean risk factors
    tmp <- rbind(colMeans(cP),
                 models[c(K, order.bic[1:10]), 12:14])
    print(round(1200*tmp, digi=2))
    return(models)
}
