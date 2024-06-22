##########################################################
## affine model functions
## Joslin-Singleton-Zhou normalization
## mostly based on Scott Joslin's code

getLambdaRVAR <- function(gamma, cP, K0Q.cP, K1Q.cP, OmegaInv) {
    ## restricted VAR estimation to obtain risk prices
    ## mu.Q, Phi.Q given
    ## restriction: vec( [mu, Phi] ) = R * lambda + vec( [mu^Q, phi^Q])
    ## estimate lambda -- Luetkepohl (5.2.6)
    N <- length(K0Q.cP)
    T <- nrow(cP)

    ## restriction:  beta = R*lambda + r
    R <- matrix(0, N*(N+1), sum(gamma==1))
    R[ cbind(which(gamma==1), seq(1, sum(gamma))) ] <- 1
    r <- c(K0Q.cP, as.numeric(K1Q.cP + diag(N)))

    Z <- t(cbind(1, cP[1:(T-1),]));    ## regressors
    Y <- t(cP[2:T,]); y <- as.numeric(Y) ## dependent variable

    ## just for fun -- unrestricted VAR identical to c(mu.hat, as.numeric(Phi.hat))
    ## beta.hat <- kronecker(solve(tcrossprod(Z))%*%Z, diag(N)) %*% y

    z <- y - kronecker(t(Z), diag(N)) %*% r
    inv.cov.mat <- t(R) %*% kronecker(tcrossprod(Z), OmegaInv) %*% R
    cov.mat <- solve(inv.cov.mat)
    lambda.hat <- as.numeric( cov.mat %*% t(R) %*% kronecker(Z, OmegaInv) %*% z )
    lambda.full <- R %*% lambda.hat
    return(list(lambda.hat=lambda.hat, Lam0=lambda.full[1:N], Lam1=matrix(tail(lambda.full, -N), N, N), cov.mat=cov.mat, inv.cov.mat=inv.cov.mat, R=R))
}

jsz.llk <- function (yields.o, W, K1Q.X, kinfQ=NA, K0P.cP=NA, K1P.cP=NA, Sigma.cP, mats, dt, sigma.e=NA, restr=0, ind.restr=NA, Lam1.rank=NA, ev.max=NA, Lam0, Lam1) {
    ## Compute the likelihood for a Gaussian term structure.
    ## Source "A New Perspective on Gaussian Dynamic Term Structure Models" by Joslin, Singleton and Zhu
    ##
    ## INPUTS:
    ## yields.o   : (T+1)*J,  matrix of observed yields (first row are t=0 observations, which likelihood conditions on)
    ## mats       : 1*J,      maturities in years
    ## dt         : scalar,   length of period in years
    ##
    ## W          : N*J,      vector of portfolio weights to fit without error.
    ## K1Q,X      : N*N,      normalized latent-model matrix (does not have to be diagonal, see form below)
    ## kinfQ      : scalar,  when the model is stationary, the long run mean of the annualized short rate under Q is -kinfQ/K1(m1,m1)
    ## Sigma.cP   : N*N,      positive definite matrix that is the covariance of innovations to cP
    ##
    ## OPTIONAL INPUTS -- concentrated out if not supplied:
    ## K0P.cP     : N*1       intercept in VAR for cP
    ## K1P.cP     : N*N       mean reversion matrix in VAR for cP
    ## sigma.e    : scalar    standard error of yield observation errors
    ## restr      : scalar    what type of restrictions on market prices of risk or VAR params
    ##                        0 -- no restrictions
    ##                        1 -- zero restrictions indicated by vector ind.restr
    ##                        2 -- reduced rank restriction (JSZ)
    ##                        3 -- eigenvalue restr. lam_1^Q = lam_1^P (JPS)
    ##                        4 -- restrictions on VAR as in Duffee-Forecasting
    ##                        planned -- zero restrictions and eigenvalue restriction
    ## ind.restr  : N^2*1     for case 1 -- vector with indicators for zero restrictions on Lam1
    ## Lam1.rank  : scalar    for case 2 -- rank of Lamda.1
    ## ev.max     : scalar    for cases 3 and 4 -- largest eigenvalue of Phi -- NA means Q-ev
    ##
    ## OUTPUT:
    ## llk        : T*1       time series of -log likelihoods (includes 2-pi constants)
    ## AcP        : 1*J       yt = AcP' + BcP'*Xt  (yt is J*1 vector)
    ## BcP        : N*J       AcP, BcP satisfy internal consistency condition that AcP*W' = 0, BcP*W' = I_N
    ## AX         : 1*J       yt = AX' + BX'*Xt
    ## BX         : N*J       Xt is the 'jordan-normalized' latent state
    ## ...
    ##
    ## The model takes the form:
    ##   r(t) = rho0.cP + rho1.cP'*cPt
    ##        = 1'*Xt  (Xt is the 'jordan-normalized' state
    ##        = 1 period discount rate (annualized)
    ##
    ## Under Q:
    ##   X(t+1) - X(t)   = K0Q.X  + K1Q.X*X(t)  + eps_X(t+1),   cov(eps_X(t+1)) = Sigma.X
    ##   cP(t+1) - cP(t) = K0Q.cP + K1Q.cP*X(t) + eps_cP(t+1),  cov(eps_cP(t+1)) = Sigma.cP
    ##   where Sigma.X is chosen to match Sigma.cP
    ## and K0Q_X(m1) = kinfQ where m1 is the multiplicity of the highest eigenvalue (typically 1)

    ## Under P:
    ##   cP(t+1) - cP(t) = K0P.cP + K1P.cP*X(t) + eps_cP(t+1),  cov(eps_cP(t+1)) = Sigma.cP
    ##
    ## Model yields are given by:
    ##   yt^m = AcP' + BcP'*cPt  (J*1)
    ## And observed yields are given by:
    ##  yt^o = yt^m + epsilon.e(t)
    ## where V*epsilon.e~N(0,sigma.e^2 I_(J-N))
    ## and V is an (J-N)*J matrix which projects onto the span orthogonal to the
    ## row span of W.  This means errors are orthogonal to cPt and cPt^o = cPt^m.
    ##

########################################################################
    ## Setup
    T <- nrow(yields.o)-1
    J <- ncol(yields.o)
    N <- nrow(W)
    cP <- yields.o %*% t(W) # (T+1)*N, cP stands for math caligraphic P.
########################################################################

########################################################################
    ## COMPUTE THE Q-LIKELIHOOD:
    ## First find the loadings for the model:
    ## yt = AcP' + BcP'*cPt, AcP is 1*J, BcP is N*J

    if (is.na(kinfQ)) {
        ## concentrate out kinfQ
        ## AcP = alpha0_cP*kinf + alpha1_cP
        rho0.cP <- 0
        ## AcP0, AX0 will be the loadings with rho0_cP = 0, which won't be correct
        loads <- jsz.loadings.rho0cP(W, K1Q.X, rho0.cP, Sigma.cP, mats, dt)
        ## [BcP, AcP0, K0Q_cPx, K1Q_cP, rho0_cP, rho1_cP, K0Q_X, K1Q_X, AX0, BX, Sigma_X, alpha0_cP, alpha1_cP, alpha0_X, alpha1_X, m1]
        BcP <- loads$BcP; BX <- loads$BX
        alpha0.X <- loads$alpha0.X; alpha1.X <- loads$alpha1.X
        alpha0.cP <- loads$alpha0.cP; alpha1.cP <- loads$alpha1.cP
        K1Q.X <- loads$K1Q.X; m1 <- loads$m1;
        ## back out kinfQ that fits average yields
        require(MASS)
        V <- t(Null(t(W)))
        kinfQ <- t(colMeans(yields.o[2:(T+1),]) - t(alpha1.cP) - t(BcP)%*%colMeans(cP[2:(T+1),]))%*%(t(V)%*%V%*%t(alpha0.cP)) / (alpha0.cP%*%t(V)%*%V%*%t(alpha0.cP))
        kinfQ <- as.numeric(kinfQ)

        ## get correct loadings
        AX <- alpha0.X*kinfQ + alpha1.X;
        AcP <- alpha0.cP*kinfQ + alpha1.cP;

        ## get these to return to caller (not used in jsz.llk)
        K0Q.X <- matrix(0,N,1);
        K0Q.X[m1] <- kinfQ;
        params <- jsz.rotation(W, K1Q.X, K0Q.X, dt, BX, AX);
        K0Q.cP <- params$K0Q.cP; K1Q.cP <- params$K1Q.cP;
        rho0.cP <- params$rho0.cP; rho1.cP <- params$rho1.cP
    } else {
        loads <- jsz.loadings(W, K1Q.X, kinfQ, Sigma.cP, mats, dt)
        BcP <- loads$BcP; AcP <- loads$AcP; K0Q.cP <- loads$K0Q.cP; K1Q.cP <- loads$K1Q.cP;
        rho0.cP <- loads$rho0.cP; rho1.cP <- loads$rho1.cP; K0Q.X <- loads$K0Q.X; K1Q.X <- loads$K1Q.X;
        AX <- loads$AX; BX <- loads$BX;
    }
    yields.m <- rep(1,T+1)%*%AcP + cP %*% BcP # (T+1)*J, model-implied yields
    yield.errors <- yields.o[2:(T+1),] - yields.m[2:(T+1),]; # T*J
    square_orthogonal_yield.errors <- yield.errors^2; # T*J, but N-dimensional projection onto W is always 0, so effectively (J-N) dimensional

    ## Compute optimal sigma.e if it is not supplied
    if (is.na(sigma.e))
        sigma.e <- sqrt( sum(square_orthogonal_yield.errors)/(T*(J-N)) )

    term1 <- .5*rowSums(square_orthogonal_yield.errors)/sigma.e^2
    term2 <- (J-N)*.5*log(2*pi)
    term3 <- .5*(J-N)*log(sigma.e^2)
    llkQ <- term1 + term2 + term3 # 1*T

########################################################################

########################################################################
    ## COMPUTE THE P-LIKELIHOOD:

    if (any(is.na(K0P.cP))||any(is.na(K1P.cP))) {
        if (restr==0) {
            ## unrestricted MPR -- run OLS for unconstrained VAR
            ## Run OLS to obtain maximum likelihood estimates of K0P, K1P
            var1 <- ar.ols(cP, order=1, aic=FALSE, demean=FALSE, intercept=TRUE)
            K1P.cP <- var1$ar[,,]-diag(N)
            K0P.cP <- var1$x.intercept
###########################
        } else if (restr==1) {
            ## zero restrictions on MPR
            if (missing(Lam0)|missing(Lam1)) {
                ## Lam0/Lam1 not provided -- concentrated out of LLK
                if (length(ind.restr)!=N+N^2)
                    stop("jsz.llk: number of indicators not equal to elements of [Lam0, Lam1]")
                if (sum(ind.restr)>0) {
                    mats <- getLambdaRVAR(ind.restr, cP, K0Q.cP, K1Q.cP, solve(Sigma.cP))
                    Lam0 <- mats$Lam0
                    Lam1 <- mats$Lam1
                } else {
                    Lam0 <- matrix(0, N, 1)
                    Lam1 <- matrix(0, N, N)
                }
            } else {
                if (!any(is.na(ind.restr)))
                    warning("jsz.llk: restr=1, Lam0/Lam1 and ind.restr provided -> ignoring ind.restr")
            }

            K0P.cP <- K0Q.cP + Lam0
            K1P.cP <- K1Q.cP + Lam1
###########################
        } else {
            stop("invalid value of restr")
        }
    } else {
        if (!restr==0) stop("can't provide VAR params AND have restrictions on MPR!")
    }

    innovations = t(cP[2:(T+1),]) - (K0P.cP%*%matrix(1,1,T) + (K1P.cP+diag(N))%*%t(cP[1:T,])) # N*T

    llkP = .5*N*log(2*pi) + .5*log(det(Sigma.cP)) + .5*colSums(innovations*solve(Sigma.cP, innovations)) # 1*T

########################################################################

    jsz.llk <- list(llk=t(llkQ + llkP), AcP=AcP, BcP=BcP, AX=AX, BX=BX, K0P.cP=K0P.cP, K1P.cP=K1P.cP, sigma.e=sigma.e, K0Q.cP=K0Q.cP, K1Q.cP=K1Q.cP, rho0.cP=rho0.cP, rho1.cP=rho1.cP, K0Q.X=K0Q.X, K1Q.X=K1Q.X, cP=cP, llkP=llkP, llkQ=llkQ)

}

########################################################################
########################################################################

jsz.loadings <- function(W, K1Q.X, kinfQ, Sigma.cP, mats, dt) {
    ## Inputs:
    ##   mats       : 1*J,      maturities in years
    ##   dt         : scalar,   length of period in years
    ##   W          : N*J,      vector of portfolio weights to fit without error.
    ##   K1Q.X      : N*N
    ##   kinfQ      : scalar,   determines long run mean
    ##   Sigma.cP   : N*N  covariance of innovations
    ##
    ## Returns:
    ##   AcP    : 1*J
    ##   BcP    : N*J
    ##   K0Q.cP : N*1
    ##   K1Q.cP : N*N
    ##   rho0.cP: scalar
    ##   rho1.cP: N*1
    ##   K0Q.X  : N*1
    ##   K1Q.X  : N*N
    ##   AX  : 1*J
    ##   BX  : N*J
    ##
    ## This function:
    ## 1. Compute the loadings for the normalized model:
    ##     X(t+1) - X(t) = K0Q.X + K1Q.X*X(t) + eps_X(t+1), cov(eps_X)=Sigma_X
    ##     and r(t) = 1.X(t)
    ##     where r(t) is the annualized short rate, (i.e. price of 1-period zero coupon bond at time t is exp(-r(t)*dt))
    ##    and K0Q_X(m1) = kinf, and K0Q_X is 0 in all other entries.
    ##      m1 is the multiplicity of the first eigenvalue.
    ##    Sigma.X is not provided -> solved for so that Sigma.cP (below) is matched.
    ##    yt = AX' + BX'*Xt
    ##
    ## 2. For cPt = W*yt and the model above for Xt, find AcP, BcP so that
    ##    yt = AcP' + BcP'*cPt
    ##
    ## 3. Computes the rotated model parameters K0Q.cP, K1Q.cP, rho0.cP, rho1.cP

    J <- length(mats)
    N <- nrow(K1Q.X)
    rho0d <- 0
    rho1d <- rep(1,N)
    mats.periods <- round(mats/dt)
    M <- max(mats.periods)

    adjK1QX <- jszAdjustK1QX(K1Q.X)
    K1Q.X <- adjK1QX$K1Q.X
    m1 <- adjK1QX$m1

    K0Q.X <- matrix(0,N,1)
    K0Q.X[m1] <- kinfQ

############################################################
    ## we need to compute Sigma.X by first computing BX
    ##
    ## First compute the loadings ignoring the convexity term -- BX will be correct
    ## yt = AX' + BX'*Xt
    ## yt is J*1, AX is 1*J, BX is N*J, Xt is N*1, W is N*J
    ##
    ## cPt = W*yt  (cPt N*1, W is N*J)
    ##     = W*AX' + W*BX'*Xt
    ##     = WAXp + WBXp*Xt
    ##
    ## Substituting:
    ## yt = AX' + BX'*(WBXp\(cPt - WAXp))
    ##    = (I - BX'*(WBXp\WAXp))*AX' + BX'*WBXp\cPt
    ##    = AcP' + BcP'*cPt
    ## where AcP = AX*(I - BX'*(WBXp\WAXp))'
    ##       BcP = (WBXp)'\BX
    ##
    ## Sigma.cP = W*BX'*Sigma_X*(W*BX')'
    ## Sigma.X = (W*BX')\Sigma.cP/(W*BX')'

    loads.X.prelim <- gaussian.loadings(mats.periods, K0Q.X, K1Q.X, matrix(0, N, N), rho0d*dt, rho1d*dt, dt)

    BX <- loads.X.prelim$B
    WBXp <- W %*% t(BX)  # N*N

    Sigma.X <- solve(WBXp, Sigma.cP) %*% solve(t(WBXp)) # (W*BX')\Sigma.cP/(BX*W');

############################################################
    ## Now with Sigma_X in hand, compute loadings for AX
    loads.X <- gaussian.loadings(mats.periods, K0Q.X, K1Q.X, Sigma.X, rho0d*dt, rho1d*dt, dt)

    AX <- loads.X$A  # 1*J

############################################################
    ## Rotate the model to obtain the AcP, BcP loadings.
    ## (See above for calculation)
    WAXp <- W %*% t(AX)  # N*1
    WBXpinv <- solve(WBXp) # N*N

    BcP <- t(WBXpinv) %*% BX # N*J
    AcP <- AX %*% t(diag(J) - t(BX)%*% solve(WBXp,W))  # 1*J

    ## compute rotated model parameters
    K1Q.cP <- WBXp %*% K1Q.X %*% WBXpinv
    K0Q.cP <- WBXp %*% K0Q.X - K1Q.cP %*% WAXp

    rho1.cP <- t(WBXpinv) %*% rep(1,N)
    rho0.cP <- -t(WAXp) %*% rho1.cP

############################################################
    jsz.loadings <- list(AX=AX, BX=BX, AcP=AcP, BcP=BcP, K0Q.cP=K0Q.cP, K1Q.cP=K1Q.cP, rho0.cP=rho0.cP, K0Q.X=K0Q.X, K1Q.X=K1Q.X, rho1.cP=rho1.cP, Sigma.X=Sigma.X)

}

########################################################################
########################################################################

jsz.rotation <- function(W, K1Q.X, K0Q.X, dt, BX, AX) {
    ## Inputs:
    ##   W          : N*J,      vector of portfolio weights to fit without error.
    ##   K1Q.X      : N*N
    ##   K0Q.X      : N*1
    ##   dt         : scalar,   length of period in years
    ##   BX         : N*J  (BX, AX) are optional (saves time)
    ##   AX         : 1*J
    ##
    ## Returns:  [K0Q.cP, K1Q.cP, rho0.cP, rho1.cP]
    ##   K0Q.cP : N*1
    ##   K1Q.cP : N*N
    ##   rho0.cP : scalar
    ##   rho1.cP : N*1
    ##
    ## r(t) = rho0.cP + rho1.cP'*cPt
    ##      = 1'*Xt
    ##      = 1 period discount rate (annualized)
    ##
    ## Under Q:
    ##   X(t+1) - X(t)   = K0Q.X  + K1Q.X*X(t)  + eps_X(t+1),   cov(eps_X(t+1)) = Sigma_X
    ##   cP(t+1) - cP(t) = K0Q.cP + K1Q.cP*X(t) + eps_cP(t+1),  cov(eps_cP(t+1)) = Sigma.cP
    ## Where Sigma_X is chosen to match Sigma.cP
    ##
    ## cPt = W*yt  (cPt N*1, W is N*J)
    ##     = W*AX' + W*BX'*Xt
    ##     = WAXp + WBXp*Xt
    ##
    ## Delta(cP) = WBXp*Delta(Xt)
    ##           = WBXp*(K1Q.X*Xt + sqrt(Sigma_X)*eps(t+1))
    ##           = WBXp*(K1Q.X)*(WBXp\(cPt - WAXp)) + sqrt(Sigma.cP)*eps(t+1)
    ##           = WBXp*(K1Q.X)/WBXp*cPt - WBXp*(K1Q.X)/WBXp*WAXp] + sqrt(Sigma.cP)*eps(t+1)
    ##
    ## rt = 1'*Xt  [annualized 1-period rate]
    ##    = 1'*(WBXp\(cPt - WAXp))
    ##    = [- 1'*(WBXp\WAXp)] + ((WBXp)'1)'*cPt

    N <- nrow(K1Q.X)
    WBXp <- W %*% t(BX)
    WAXp <- W %*% t(AX)
    WBXpinv <- solve(WBXp)

    K1Q.cP <- WBXp %*% K1Q.X %*% WBXpinv
    K0Q.cP <- WBXp %*% K0Q.X - K1Q.cP %*% WAXp

    rho1.cP = t(WBXpinv) %*% rep(1,N)
    rho0.cP = - t(WAXp) %*% rho1.cP

    jsz.rotation <- list(K0Q.cP=K0Q.cP, K1Q.cP=K1Q.cP, rho0.cP=rho0.cP, rho1.cP=rho1.cP)
}

########################################################################
########################################################################

gaussian.loadings <- function(maturities, K0d, K1d, H0d, rho0d, rho1d, timestep=1) {
    ## maturities: M*1
    ## K0d      : N*1
    ## K1d      : N*1
    ## H0d      : N*N
    ## rho0d    : scalar
    ## rho1d    : N*1
    ## timestep : optional argument.
    ##
    ## By : N*M
    ## Ay : 1*M  (faster to not compute with only one output argument)
    ##
    ## r(t)   = rho0d + rho1d'Xt
    ##        = 1 period discount rate
    ## P(t)   =  price of  t-period zero coupon bond
    ##        = EQ0[exp(-r0 - r1 - ... - r(t-1)]
    ##        = exp(A+B'X0)
    ## yields = Ay + By'*X0
    ##   yield is express on a per period basis unless timestep is provided.
    ##   --For example, if the price of a two-year zero is exp(-2*.06)=exp(-24*.005),
    ##   --and we have a monthly model, the function will return Ay+By*X0=.005
    ##   --unless timestep=1/12 is provided in which case it returns Ay+By*X0=.06
    ##
    ## Where under Q:
    ##   X(t+1) - X(t) = K0d + K1d*X(t) + eps(t+1),  cov(eps(t+1)) = H0d
    ##
    ## A1 = -rho0d
    ## B1 = -rho1d
    ## At = A(t-1) + K0d'*B(t-1) .5*B(t-1)'*H0d*B(t-1) - rho0d
    ## Bt = B(t-1) + K1d'*B(t-1) - rho1d
    ##
    ## maturities: 1*M # of periods

    M = length(maturities)
    N = length(K0d)
    Atemp = 0
    Btemp = matrix(0,N,1)
    Ay = matrix(NA,1,M)
    By = matrix(NA,N,M)

    curr_mat = 1
    K0dp <- t(K0d)
    K1dp <- t(K1d)
    for (i in 1:maturities[M]) {
        Atemp <- Atemp + K0dp%*%Btemp +.5%*%t(Btemp)%*%H0d%*%Btemp - rho0d
        Btemp <- Btemp + K1dp%*%Btemp - rho1d

        if (i==maturities[curr_mat]) {
            Ay[1,curr_mat] <- -Atemp/maturities[curr_mat]
            By[,curr_mat] <- -Btemp/maturities[curr_mat]
            curr_mat <- curr_mat + 1
        }
    }

    gaussian.loadings <- list(A = Ay/timestep, B = By/timestep)
}

gaussian.loadings.diag <- function(maturities, K0d, K1d.diag, H0d, rho0d, rho1d, timestep=1) {
    ## DOESN'T HANDLE UNIT ROOTS!!
    ## THIS FUNCTION ASSUMES K1d is diagonal

    stop("gaussian.loadings.diag: numerical issue (floating point arithmetic) -- for roots very close to unity (daily model), this gives the wrong result")

    ## K0d      : N*1
    ## K1d.diag : N*1
    ## H0d      : N*N
    ## rho0d    : scalar
    ## rho1d    : N*1
    ## timestep : optional argument.

    ## We can compute the loadings by recurrence relations (see gaussian.loadings)
    ## or in closed form by noting that
    ##    r0+r1+..+r(t-1) = c.X(0) + alpha0 + alpha1*eps1 + ... + alpha(t-1)*eps(t-1)
    ##                    ~ N(c.X(0) + alpha0, alpha1'H0d*alph1 + ... + alpha(t-1)'*H0d*alpha(t-1))
    ##
    ## And then use the MGF of Y~N(mu,Sigma) is E[exp(a.Y)] = a'*mu + .5*a'*Sigma*a
    ## (or similarly use the partial geometric sum formulas repeatedly)
    ##
    ## Let G = K1+I
    ## X(0)
    ## X(1) = K0 + G*X(0) + eps1
    ## X(2) = K0 + G*K0 + G^2*X(0) + G*eps1 + eps2
    ## X(3) = K0 + G*K0 + G^2*K0 + G^3*X(0) + G^2*eps1 + G*eps2 + eps3
    ## X(n) = sum(I+G+..+G^(n-1))*K0 + G^n*X(0) + sum(i=1..n,G^(n-i)*epsi)
    ##      = (I-G\(I-G^n)*K0 + G^n*X(0) + sum(i=1..n,G^(n-i)*epsi)
    ##
    ## cov(G^n*eps) = G^n*cov(eps)*(G^n)'
    ## vec(cov(G^n*eps) = kron(G^n,G^n)*vec(eps)
    ##                   = (kron(G,G)^n)*vec(eps)
    ##
    ## sum(X(i),i=1..n) = mu0 + mu1*X0 + u
    ##    mu0 = (I-G)\(I - (I-G)\(G-G^(n+1))*K0
    ##    mu1 = (I-G)\(G-G^(n+1))
    ##    vec(cov(u)) = see below.
    ##  u = (I-G)\(I-G^n)*eps1 +
    ##      (I-G)\(I-G^(n-1))*eps2 + ..
    ##      (I-G)\(I-G)*epsn
    ##  cov(u) = (I-G)\Sig/(I-G)'
    ## Sig = sum(cov(eps)) + sum(i=1..n,G^i*cov(eps)) +
    ##       sum(i=1..n,cov(eps)G^i') + sum(i=1..n,G^i*cov(eps)*G^i')
    ## compute the last one using vec's.  see below.

    ## K1d_diag is N*1 -- the diagonal of K1d
    M = length(maturities);
    N = length(K0d);
    Ay = matrix(NA, 1, M);
    By = matrix(NA, N, M);

    if (length(K1d.diag)!=N) stop("K1d.diag needs to be an N-vector with diagonal elements of K1d")

    G = K1d.diag+1; ## N*1
    GG <- outer(G, G)
    GGvec <- as.numeric(GG)
    H0dvec <- as.numeric(H0d)
    outer.K1d.diag <- outer(K1d.diag,K1d.diag)

    for (m in 1:M) {
        mat = maturities[m];
        if (mat==1) {
            By[,m] = rho1d;
            Ay[m] = rho0d;
        } else {

            ## my way
            ## X(0) + ... + X(n-1) = mu0 + mu1 X0 + ...
            ##    mu0 = [I*n - (1-Phi^n)(1-Phi)^(-1)](1-Phi)^(-1) = (I*n - mu1)(1-Phi)^(-1)
            ##    mu1 = (1-Phi^n)(1-Phi)^(-1)
            ## I <- diag(N)
            ## (mu1 <- (I-Phi^mat)%*%solve(I-Phi))  # NxN diagonal matrix because Phi is diagonal
            ## (mu0 <- (I*mat - mu1)%*%solve(I-Phi)%*%mu)
            ## (By.n <- rho1d%*%mu1/mat)

            i <- mat-1
            ## X(0) + ... + X(i)~N(mu0 + mu1*X(0), Sigma0)
            (mu1 <- 1 - (G - G^(i+1))/K1d.diag); ## N*1
            (mu0 <- -((i+1)*1 - mu1)/K1d.diag*K0d)
            By[,m] <- rho1d*mu1/mat

            Sigma_term1 <- i*H0d; ## N*N
            Sigma_term2 = matrix(rep(mu1-1, N), N, N)*H0d  ## N*N
            Sigma_term3 = t(Sigma_term2);
            Sigma_term4 = as.numeric(GG - GG^(i+1))/(1 - GGvec)*H0dvec;
            Sigma_term4 = matrix(Sigma_term4, N, N);
            Sigma0 = (Sigma_term1 - Sigma_term2 - Sigma_term3 + Sigma_term4)/outer.K1d.diag;
            Ay[m] <- rho0d + (rho1d%*%mu0 - .5*rho1d %*% Sigma0 %*% rho1d)/mat;
        }
    }
    return(list(A = Ay/timestep, B = By/timestep))

}
jszAdjustK1QX <- function(K1Q.X, eps1=1e-3) {
    ## function [K1Q_X, isTypicalDiagonal, m1] = jszAdjustK1QX(K1Q_X, eps1);
    ##
    ## This function adjusts diagonal K1Q_X to give a non-diagonal but more
    ## computationally tractable K1Q_X.
    ##
    ## K1Q_X can fall into a few cases:
    ##   0. diagonal
    ##   1. not diagonal
    ##   2. zero eigenvalue
    ##   3. near repeated roots
    ## In cases 1-3, the diagonal closed form solver doesn't work, so compute differently.
    ## In case 1-2, we will use the recursive solver, though there are more efficient methods.
    ## In case 3, we will add a positive number above the diagonal.  this gives a different member of the set of observationally equivalent models.
    ##   So for example:
    ##      [lambda1, 0; 0, lambda2] is replaced by [lambda1, f(lambda2-lambda1); 0, lambda2] when abs(lambda1 - lambda2)<eps0
    ##   By making f not just 0/1, it will help by making the likelihood
    ##   continuous if we parameterize by kinf. (not an issue in some cases.)
    ##
    ## We also order the diagonal of diagonal K1Q.


    ## Cutoff function sets the super diagonal entry to something between 0 and
    ## 1, depending on how close the eigenvalues are.
    cutoff.fun <- function(x, eps1) {
        eps1 = 1e-3;
        eps0 = 1e-5;
        ##    xc <- 1*(x<eps0) + (1 - (x - eps0)/(eps1 - eps0))*(x>=eps0 && x<eps1) + 0*(x>eps1);
        xc <- 1*(log(x)<log(eps0)) +
            (1 - (log(x) - log(eps0))/(log(eps1) - log(eps0)))*(log(x)>=log(eps0) & log(x)<log(eps1)) +
                0*(log(x)>log(eps1));
        xc[x==0] <- 1;
        return(xc)
    }

    N <- nrow(K1Q.X)

    diag.K1Q.X <- diag(K1Q.X);
    isDiagonal <- all(K1Q.X==diag(diag.K1Q.X));

    ## For diagonal matrix, sort the diagonal and check to see if we have near repeated roots.
    if (isDiagonal) {
        diag.K1Q.X <- -sort(-diag.K1Q.X);
        K1Q.X <- diag(diag.K1Q.X);

        hasNearUnitRoot <- !all(abs(diag.K1Q.X)>eps1); ## Only applicable for diagonal
        hasNearRepeatedRoot <- !all(abs(diff(diag.K1Q.X))>eps1); ## Only applicable for diagonal
        isTypicalDiagonal <- isDiagonal && !hasNearRepeatedRoot && !hasNearUnitRoot;
    } else {
        isTypicalDiagonal <- FALSE
    }

    ## If we have a near repeated root, add a constnat above the diagonal. This
    ## representative of the equivalence class gives easier inversion for latent
    ## states vs. yields.  By varying the constant

    if (isDiagonal && !isTypicalDiagonal) {
        diff.diag <- abs(diff(diag.K1Q.X))
        super.diag <- cutoff.fun(diff.diag)
        K1Q.X[1:(N-1),2:N] <- K1Q.X[1:(N-1),2:N] +
            if (length(super.diag) == 1) {super.diag} else {diag(super.diag)}
    }
#######################################

    super.diag = diag(K1Q.X[-N,-1]);
    m1 <- max(which(cumprod(c(1,super.diag))>0))

    return(list(K1Q.X=K1Q.X, isTypicalDiagonal=isTypicalDiagonal, m1=m1))

}

########################################################################
########################################################################

jsz.loadings.rho0cP <- function(W, K1Q.X, rho0.cP, Sigma.cP, mats, dt) {
    ## like jsz.loadings but parameterized in terms of rho0.cP instead of kinfQ

    J <- length(mats)
    N <- nrow(K1Q.X)
    rho0d <- 0
    rho1d <- rep(1,N)
    mats.periods <- round(mats/dt)
    M <- max(mats.periods)

    adjK1QX <- jszAdjustK1QX(K1Q.X)
    K1Q.X <- adjK1QX$K1Q.X
    m1 <- adjK1QX$m1

    K0Q.X <- matrix(0,N,1)
    K0Q.X[m1] <- 1

    loads.X.prelim <- gaussian.loadings(mats.periods, K0Q.X, K1Q.X, matrix(0, N, N), rho0d*dt, rho1d*dt, dt)

    BX <- loads.X.prelim$B
    alpha0.X <- loads.X.prelim$A
    WBXp <- W %*% t(BX)  # N*N
    Sigma.X <- solve(WBXp, Sigma.cP) %*% solve(t(WBXp)) # (W*BX')\Sigma.cP/(BX*W');

    ## Now with Sigma_X in hand, compute loadings for AX
    loads.X <- gaussian.loadings(mats.periods, K0Q.X, K1Q.X, Sigma.X, rho0d*dt, rho1d*dt, dt)
    AX1 <- loads.X$A  # 1*J
    ## AX1 gives the intercept with K0Q_X all zeros except 1 in the m1-th entry.
    ## So AX = alpha0_X*kinf + alpha1_X which alpha1_X = AX1 - alpha0_X
    alpha1.X <- AX1 - alpha0.X;

    ## Need to find what kinf should be to get the desired rho0_cP:
    ## rt = 1'*Xt
    ## cPt = (W*alpha0_X')*kinf + (W*alpha1_X') + (W*BX')*Xt
    ## rt = 1'*(W*BX')^(-1)*[cPt -(W*alpha0_X')*kinf - (W*alpha1_X')]
    ## --> rho0_cP = -1'*(W*BX')^(-1)*(W*alpha0_X')*kinf - 1'*(W*BX')^(-1)*(W*alpha1_X')
    a0 <- matrix(1,1,N)%*%solve(WBXp, W%*%t(alpha0.X));
    a1 <- matrix(1,1,N)%*%solve(WBXp, W%*%t(alpha1.X));
    kinfQ <- as.numeric(-(rho0.cP + a1)/a0);  ## -a0*kinf - a1 = rho0_cP
    K0Q.X[m1] <- kinfQ;

    AX <- alpha0.X*kinfQ + alpha1.X
    ## AcP = alpha0.cP*kinfQ + alpha1.cP
    alpha0.cP = t((diag(J) - (t(BX)%*%solve(WBXp))%*%W)%*%t(alpha0.X));
    alpha1.cP = t((diag(J) - (t(BX)%*%solve(WBXp))%*%W)%*%t(alpha1.X));
    ## could now get AcP using these -> check consistency with below

############################################################
    ## Rotate the model to obtain the AcP, BcP loadings.
    ## (See above for calculation)
    WAXp <- W %*% t(AX)  # N*1
    WBXpinv <- solve(WBXp) # N*N
    BcP <- t(WBXpinv) %*% BX # N*J
    AcP <- AX %*% t(diag(J) - t(BX)%*% solve(WBXp,W))  # 1*J

    ## compute rotated model parameters
    K1Q.cP <- WBXp %*% K1Q.X %*% WBXpinv
    K0Q.cP <- WBXp %*% K0Q.X - K1Q.cP %*% WAXp

    rho1.cP <- t(WBXpinv) %*% rep(1,N)
                                        #  rho0.cP <- - t(WAXp) %*% rho1.cP

############################################################
    return(list(AX=AX, BX=BX, AcP=AcP, BcP=BcP, K0Q.cP=K0Q.cP, K1Q.cP=K1Q.cP, rho0.cP=rho0.cP, K0Q.X=K0Q.X, K1Q.X=K1Q.X, rho1.cP=rho1.cP, alpha0.X=alpha0.X, alpha1.X=alpha1.X, alpha0.cP=alpha0.cP, alpha1.cP=alpha1.cP, m1=m1))

}

########################################################################
