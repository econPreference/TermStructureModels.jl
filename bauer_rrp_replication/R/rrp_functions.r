init <- function(N=3) {
    ## initialize model settings and load data
    ## Globals, side effects:
    ##  sets n.per, N, W, Y, dates, mats

    n.per <<- 12 # monthly data
    N <<- N      # three risk factors

    ## load yield data
    cat("loading Anh Le's yield data...\n")
    if ('data' %in% dir()) {
        load("data/le_data_monthly.RData")  # Y, dates
    } else {
        load("../data/le_data_monthly.RData")  # Y, dates
    }
    mats <- c(12,24,36,48,60,84,120)
    Y <- Y[,mats]/n.per

    if (dim(Y)[2]!=length(mats)) stop("wrong number of columns in yield matrix");

    start.sample <- 19900101
    start.date <<- c(1990,1)
    end.sample <- 20071231

    sel.sample <- dates>=start.sample & dates<=end.sample
    Y <- Y[sel.sample,]  ## change local variable
    dates <- dates[sel.sample]
    cat("sample from", as.character(min(dates)), "to", as.character(max(dates)), "\n")

    ## first N principal components => pricing factors
    W <<- getW(Y, N)

    ## make local variables global   -- alt:  assign("Y", Y, env=.GlobalEnv)
    Y <<- Y; dates <<- dates; mats <<- mats
}

tic <- function(gcFirst = TRUE, type=c("elapsed", "user.self", "sys.self"))
{
    type <- match.arg(type)
    assign(".type", type, envir=baseenv())
    if(gcFirst) gc(FALSE)
    tic <- proc.time()[type]
    assign(".tic", tic, envir=baseenv())
    invisible(tic)
}

toc <- function()
{
    type <- get(".type", envir=baseenv())
    toc <- proc.time()[type]
    tic <- get(".tic", envir=baseenv())
    print(toc - tic)
    invisible(toc)
}

getAllYields <- function(dates) {
    ## get yields for all monthly maturities
    ## (Anh Le's yield data)
    start.sample <- min(dates)
    end.sample <- max(dates)
    load("data/le_data_monthly.RData")  # Y, dates
    return(Y[dates>=start.sample & dates<=end.sample,]/n.per)
}

getW <- function(Y, N=3) {
    eig <- eigen(cov(Y))
    J <- ncol(Y)
    W <- t(eig$vectors[,1:N])
    ## normalize level so that loadings sum to one
    a <- sum(W[1,])
    W[1,] <- W[1,]/a
    ## normalize slope so that difference between loadings on long and short yield is one
    b <- W[2,J] - W[2,1]
    W[2,] <- W[2,]/b
    if (N>2) {
        ## normalize curvature
        c <- W[3,J] + W[3,1] - 2*W[3,floor(J/2)]
        W[3,] <- W[3,]/c
    }
    return(W)
}

initSimulation <- function(restr = TRUE) {
    ## initialize model settings for simulation
    ## and set data-generating parameters and model restrictions
    ## Value:
    ##   returns list with DGP parameters and restrictions
    ## Globals, side effects:
    ##   sets W, N, n.per, mats

    n.per <<- 12
    N <<- 2

    ## load data to obtain MLE
    cat("loading Anh Le's yield data...\n")
    if ('data' %in% dir()) {
        load("data/le_data_monthly.RData")  # Y, dates
    } else {
        load("../data/le_data_monthly.RData")  # Y, dates
    }
    mats <- c(12,24,36,48,60,84,120)
    Y <- Y[,mats]/n.per

    start.sample <- 19900101
    end.sample <- 20080000
    sel.sample <- dates>=start.sample & dates<=end.sample
    Y <- Y[sel.sample,]

    W <<- getW(Y, N)		## global parameter
    mats <<- mats      		## global parameter

    ## choose DGP parameters
    ## (1) MLE on actual data, no restrictions
    pars.urp <- estML(Y, W, mats)

    if (restr) {
        ## which parameters are significant?
        cat("results of MLE for two-factor model\n")
        rvar.res <- getLambdaRVAR(rep(1, N*(N+1)), cP=Y%*%t(W), pars.urp$loads$K0Q.cP, pars.urp$loads$K1Q.cP, pars.urp$OmegaInv)
        lambda.se <- sqrt(diag(rvar.res$cov.mat))
        out.matrix <- matrix(NA, N+N^2, 3)
        rownames(out.matrix) <- c(sapply(1:N, function(x) paste('lam0_', as.character(x), sep='')), sapply( rep((1:N),N)*10 + as.numeric(gl(N, N)), function(x) paste('Lam1_', as.character(x), sep='')))
        colnames(out.matrix) <- c('mean', 't-stat', 'sig 5%')
        out.matrix[,1] <- pars.urp$lambda
        out.matrix[,2] <- pars.urp$lambda/lambda.se

        out.matrix[,3] <- abs(out.matrix[,2])>1.96
        print(round(out.matrix,digi=4))

        ## choose true restrictions based on significance in the data
        true.gamma <- out.matrix[,3]

        ## 2. MLE of restricted model to get plausible parameter values
        pars.rrp <- estML(Y, W, mats, true.gamma)

        ## 3. set values to something similar (basically, rounded values)
        true.pars <- pars.rrp[c('Sigma', 'kinfQ', 'lamQ', 'lambda')]
    } else {
        ## UNRESTRICTED MODEL AS DGP
        true.gamma <- rep(1, N+N^2)
        true.pars <- pars.urp[c('Sigma', 'kinfQ', 'lamQ', 'lambda')]
        true.pars$lambda[1:2] <- c(.1, -.1)/1200
        true.pars$lambda[3:6] <- c(-.1, .1, -.1, -.1)
    }

    cat("# DGP parameters:\n")
    print(true.pars)
    true.pars$Omega <- true.pars$Sigma %*% t(true.pars$Sigma)
    true.pars$loads <- jsz.loadings(W, diag(true.pars$lamQ), true.pars$kinfQ, true.pars$Omega, mats, dt=1)
    muQ <- true.pars$loads$K0Q.cP*1200
    PhiQ <- diag(N) + true.pars$loads$K1Q.cP
    true.Pdyn <- getPdyn(true.pars$loads, true.pars$lambda)
    true.pars$Phi <- true.Pdyn$Phi
    print(PhiQ)
    cat("eigenvalues of Phi^Q:", eigen(PhiQ)$values, "\n")
    print(true.pars$Phi)
    cat("abs eigenvalues of DGP Phi: ", abs(eigen(true.pars$Phi)$values), "\n")

    if (max(abs(eigen(true.Pdyn$Phi)$values))>1)
        stop("explosive DGP")
    true.pars$mu <- true.Pdyn$mu
    true.pars$sige2 <- (2/120000)^2

    print(muQ)
    print(true.pars$mu)

    cat("average yields in population: \n")
    EcP <- solve(diag(N)-true.pars$Phi)%*%true.pars$mu
    EY <- true.pars$loads$AcP + t(EcP)%*%true.pars$loads$BcP
    cat(round(EY*1200, digi=2), "\n")

    true.pars$gamma <- true.gamma
    true.pars
}

simulateYields <- function(pars, T) {
    ## simulate yield data
    ##
    ## Parameters:
    ##  pars -- list with model parameters and cross-sectional loadings
    ##          mu, Phi, Sigma, sige2, loads$AcP, loads$BcP
    ##  T -- sample size
    ##
    ## Value:
    ##  yields (TxJ)
    cat("Simulating yield data...\n")
    N <- length(pars$mu)
    J <- length(pars$loads$AcP)
    ## simulate time series of factors
    cP.sim <- matrix(NA, T, N)
    cP.sim[1,] <- solve(diag(N)-pars$Phi)%*%pars$mu
    for (t in 2:T)
        cP.sim[t,] <- pars$mu + pars$Phi %*% cP.sim[t-1,] + pars$Sigma %*% rnorm(N)
    ## calculate yields
    Y.sim <- matrix(NA, T, J)
    for (t in 1:T)
        Y.sim[t,] <- pars$loads$AcP + cP.sim[t,]%*%pars$loads$BcP + sqrt(pars$sige2)*rnorm(J)
    return(Y.sim)
}

getPdyn <- function(loads, lambda) {
    ## calculate mu and Phi from Q dynamics and risk prices
    N <- length(loads$K0Q.cP)
    lam0 <- lambda[1:N]
    lam1 <- matrix(lambda[(N+1):(N*(N+1))], N, N)
    return(list(mu = loads$K0Q.cP + lam0, Phi = diag(N) + loads$K1Q.cP + lam1))
}

getRP <- function(loads, mu, Phi) {
    ## calculate risk prices from P and Q dynamics
    lam0 <- mu - loads$K0Q.cP
    lam1 <- Phi - loads$K1Q.cP - diag(N)
    lambda <- c(lam0, lam1)
    return(list(lambda=lambda, lam0=lam0, lam1=lam1))
}

getLlkP <- function(Xp, mu, Phi, Omega, OmegaInv) {
    ## calculate value of P-likelihood
    T <- ncol(Xp); N <- nrow(Xp)
    if (missing(OmegaInv)) OmegaInv <- solve(Omega)
    e <- Xp[,2:T] - (mu %*% matrix(1,1,T-1) + Phi %*% Xp[,1:(T-1)]) # N*(T-1)
    llkP <- -.5*N*log(2*pi)-.5*log(det(Omega))-.5*colSums(e*(OmegaInv%*%e))#1*(T-1)
    return(sum(llkP))
}

getLlkQ <- function(Y, W, mats, lamQ, kinfQ, Omega, sige2) {
    ## calculate value of Q-likelihood
    ## Value:
    ##  list with loadings, errors, and value of Q-likelihood
    cP <- Y %*% t(W)
    loads <- jsz.loadings(W, diag(lamQ), kinfQ, Omega, mats, dt=1)
    Y.hat <- rep(1,nrow(cP))%*%loads$AcP + cP%*%loads$BcP
    errors <- Y - Y.hat
    ## dropping constants -T*(J-N)*.5*log(2*pi) - .5*T*(J-N)*log(sigma.e^2)
    ## -> if need to include constants here, also adjust drawSige2
    llkQ <- -.5*sum(errors^2)/sige2
    return(list(loads=loads, errors=errors, llkQ=llkQ))
}

## ratio of inverse Wishart densities
iwish.ratio <- function(Wnum,Wdenom,v,Snum,Sdenom) {
    ## - parameterization as in MCMCpack / Wikipedia
    ## - same precision -- cancels out
    fact1 <- (det(Snum)/det(Sdenom))^(v/2)*(det(Wnum)/det(Wdenom))^(-(v+N+1)/2)
    fact2 <- exp(-.5*(sum(diag(Snum%*%solve(Wnum)))-sum(diag(Sdenom%*%solve(Wdenom)))))
    if (fact1 == 0 | fact2 == 0) {
        iwish.ratio <- 0
    } else {
        iwish.ratio <- fact1*fact2
    }
}

getForwLoads <- function(mats, mu, Phi, Omega, rho0, rho1) {
    loads2 <- gaussian.loadings(mats+1, mu, Phi-diag(N), Omega, rho0, rho1)
    loads1 <- gaussian.loadings(mats, mu, Phi-diag(N), Omega, rho0, rho1)
    Af <- loads2$A*(mats+1)-loads1$A*mats
    Bf <- t(t(loads2$B)*(mats+1))-t(t(loads1$B)*mats)
    return(list(A=Af, B=Bf))
}

vec2str <- function(x)
    paste(x, collapse="")

str2vec <- function(x)
    substring(x, seq(1,nchar(x)),seq(1,nchar(x)))

getResultsFileName <- function(method, N, gamma=1) {
    ## if specific model is estimates (MLE/MCMC) then indicate whether restricted (rrp) or not (urp)
    rp <- ifelse(method %in% c("mle", "mcmc"), ifelse(all(gamma==1), "urp_",
                                                      paste("rrp_", vec2str(gamma),"_", sep="")), "")
    results.file <- paste("estimates/", method, "_",
                          "N", N, "_",
                          rp,
                          as.character(format(Sys.time(), "%Y%m%d")),
                          ".RData",sep="")
                                        #results.file <- getVerFilename(results.file)
    cat("file name for estimates: ", results.file, "\n")
    return(results.file)
}

matrix.power <- function(x, y) {
    ## calculate matrix power, allowing for non-integer exponents
    ## x^y where x is a matrix, y a scalar
    if (length(x)>1) {
        eig <- eigen(x)
        x.y <- eig$vectors%*%diag(eig$values^y)%*%solve(eig$vectors)
        dimnames(x.y) <- dimnames(x)
    } else {
        x.y <- x^y
    }
    return(Re(x.y))
}

plot.recessions <- function(yrange) {
    rec.90.from <- 1990+7/12
    rec.90.to <- 1991+3/12
    rec.01.from <- 2001+3/12
    rec.01.to <- 2001+11/12
    rec.07.from <- 2007+12/12
    rec.07.to <- 2009+6/12
    polygon(x=c( rec.90.from, rec.90.from, rec.90.to, rec.90.to),
            y=c(yrange, rev(yrange)),
            density=NA, col="gray", border=NA)
    polygon(x=c( rec.01.from, rec.01.from, rec.01.to, rec.01.to),
            y=c(yrange, rev(yrange)),
            density=NA, col="gray", border=NA)
    polygon(x=c( rec.07.from, rec.07.from, rec.07.to, rec.07.to),
            y=c(yrange, rev(yrange)),
            density=NA, col="gray", border=NA)
}


makePD <- function(A) {
    if (min(eigen(A)$values)<0) {
        cat('matrix not PD, making adjustment...\n')
        D <- diag(eigen(A)$values)
        V <- eigen(A)$vectors
        D[D<0] <- 0+1e-6
        return(V %*% D %*% t(V))
    } else {
        return(A)
    }
}

irf.var1 <- function(Phi, max.lag = 500, g=1, h=1) {
    ## calculate impulse response function for a VAR(1)
    ## for the g'th variable in response to shocks to the h'th variable

    if (length(Phi)>1) {
        k <- dim(Phi)[1]
        Psi <- array(0, c(max.lag,k,k))
        Psi[1,,] <- Phi
        for (i in 2:max.lag)
            Psi[i,,] <- Phi %*% Psi[i-1,,]
        irf.var1 <- Psi[,g,h]
    } else {
        Psi <- numeric(max.lag)
        for (i in 1:max.lag)
            Psi[i] <- Phi^i
        irf.var1 <- Psi
    }
}
