getParameterNames <- function(N=3, include.P=TRUE, include.ev=TRUE) {
    param.names <- c('kinfQ',
                     sapply(1:N, function(x) paste('lamQ_', as.character(x), sep='')),
                     sapply(1:N, function(x) paste('lam0_', as.character(x), sep='')),
                     sapply( rep((1:N),N)*10 + as.numeric(gl(N, N)), function(x) paste('Lam1_', as.character(x), sep='')),
                     sapply(1:(N*(N+1)/2), function(x) paste('Sigma_', as.character(x), sep='')))
    if (include.P)
        param.names <- c(param.names,
                         sapply(1:N, function(x) paste('mu_', as.character(x), sep='')),
                         sapply( rep((1:N),N)*10 + as.numeric(gl(N, N)), function(x) paste('Phi_', as.character(x), sep='')))
    if (include.ev)
        param.names <- c(param.names,
                         sapply(1:N, function(x) paste('eig(Phi)--', as.character(x), sep='')))
    return(param.names)
}

ineff <- function(x, L=500) {
    ## get inefficiency factor
    ## note: x must not be thinned out!
    if (is.matrix(x)) {
        stop("expecting vector")
    } else {
        rho <- acf(x, plot=FALSE, lag.max=L)$acf
        rho[1] + 2*sum((1-(1:L)/L)*rho[2:(L+1)])  ## Bartlett Kernel / Newey-West
    }
}

lrv <- function(x, L=500) {
    ## get long-run variance -- Newey-West
    if (is.matrix(x)) {
        lrv.x <- numeric(ncol(x))
        for (i in 1:ncol(x)) {
            ## autocovariances
            phi <- acf(x[,i], plot=FALSE, lag.max=L, type="covariance")$acf
            ## -> phi is vector of length L+1: variance and autocovariances
            lrv.x[i] <- phi[1] + 2*sum((1-(1:L)/L)*phi[2:(L+1)])
        }
        return(lrv.x)
    } else {
        phi <- acf(x, plot=FALSE, lag.max=L, type="covariance")$acf
        phi[1] + 2*sum((1-(1:L)/L)*phi[2:(L+1)])
    }
}

getAllSimResults <- function(filename.mcmc, filename.ssvs, filename.gvs, filename.rjmcmc) {
    ## inference about gamma -- print table with all simulation results
    ## - compare MCMC, SSVS, GVS, RJMCMC

    load(filename.mcmc)  ## load now so that we have N
    out.matrix <- matrix(NA, 2+2+3*3, N+N^2)
    rownames(out.matrix) <- c("true gamma", "true lambda", "MCMC mean", "MCMC sig", sapply(c("SSVS", "GVS", "RJMCMC"), function(z) c(paste(z, "gamma"), paste(z, "lambda"), paste(z, "lambda|correct model"))))
    corr.model.chosen <- numeric(4)
    freq.corr.model <- numeric(3)

    v <- 2^seq(N*(N+1)-1, 0, -1) # to translate vector gamma.i into model indicator

    ## DGP
    out.matrix[1,] <- true.pars$gamma
    out.matrix[2,] <- true.pars$lambda
    out.matrix[2,1:N] <- out.matrix[2,1:N]*1200
    true.model <- as.numeric(true.pars$gamma %*% v)

    ## MCMC
    cat("*** MCMC ***\n")
    ## parameter estimates
    out.matrix[3,] <- colMeans(lambda.i)
    out.matrix[3,1:N] <- out.matrix[3,1:N]*1200
    ## frequency of significant estimates among n samples
    is.sig <- matrix(NA, n, N+N^2)
    for (j in 1:n) {
        ind.smpl <- ((j-1)*M+1):(j*M)
        q <- apply(lambda.i[ind.smpl,], 2, quantile, c(.025, .975))
        is.sig[j,] <- (q[1,]*q[2,])>0
    }
    sig.model <- is.sig %*% v
    row <- 4
    out.matrix[row,] <- colMeans(is.sig)
    corr.model.chosen[1] <- mean(sig.model==true.model) # how often is correct model chosen model (based on significance)
    cat("Frequency of correct model:", corr.model.chosen[1], "\n")

    ## SSVS, GVS, RJMCMC
    filenames <- c(filename.ssvs, filename.gvs, filename.rjmcmc)
    for (i in 1:length(filenames)) {
        cat("*** model selection", i, "***\n")
        cat(filenames[i], "\n")
        load(filenames[i])
        if (as.numeric(true.pars$gamma %*% v) != true.model)
            stop("Different DGP model!")
        out.matrix[row+1,] <- colMeans(gamma.i)
        out.matrix[row+2,] <- colMeans(lambda.i)
        out.matrix[row+2,1:N] <- out.matrix[row+2,1:N]*1200
        model.i <- gamma.i %*% v
        if (nrow(gamma.i)!=M*n)
            stop("gamma.i should have M*n rows")
        freq.corr.model[i] <- mean(model.i==true.model)
        cat("Frequency of correct model across all iterations:", freq.corr.model[i],"\n")
        modal.model <- numeric(n)
        for (j in 1:n) {
            ind.smpl <- ((j-1)*M+1):(j*M)
            modal.model[j] <- names(which.max(table(model.i[ind.smpl])))
        }
        corr.model.chosen[1+i] <- mean(modal.model==true.model)
        cat("How often is modal model the correct model:", corr.model.chosen[1+i], "\n")
        out.matrix[row+3,] <- colMeans(lambda.i[model.i==true.model,])
        out.matrix[row+3,1:N] <- out.matrix[row+3,1:N]*1200
        row <- row+3
    }
    out.matrix <- cbind(out.matrix, c(NA, NA, corr.model.chosen[1], NA,
                                      as.numeric(t(cbind(corr.model.chosen[2:4], freq.corr.model, NA)))))
    print(round(out.matrix, digi=4))

    return(out.matrix)

}

printAllSimResultsToLatex <- function(out.matrix, to.file=F, filename) {
    ## latex table
    if (to.file) {
        if (missing(filename)) {
            filename <- "tables/sim.tex"
        }
        cat("*** writing table ", filename, "\n")
        sink(filename)
    }
    ind <- 1:(ncol(out.matrix)-1)
    cat("\\begin{tabular}{lccccccc}\n \\hline \\hline \n")
    cat(" & \\multicolumn{6}{c}{Element of $\\gamma$} & Freq.~of \\\\ \n")
    cat(" & (1) & (2) & (3) & (4) & (5) & (6) & corr.~model\\\\ \n \\hline \n")
    cat("DGP ", sprintf("& %1.0f ", out.matrix[1,ind]), "\\\\ \\hline \n")
    cat("MCMC", sprintf("& %4.2f ", out.matrix[4,ind]), "& ",
        tail(out.matrix[3,], 1)*100, "\\% \\\\ \n", sep="")
    row <- 5
    for (s in c("SSVS", "GVS", "RJMCMC")) {
        cat(s, sprintf("& %4.2f ", out.matrix[row, ind]), "& ",
            tail(out.matrix[row, ],1)*100, "\\% \\\\ \n", sep="")
        row <- row + 3
    }

    cat("\\hline \\hline \n \\end{tabular}\n")
    if (to.file)
        sink()
}

getAllSimResultsPersistence <- function(filenames) {
    ## analyze simulation results: persistence, volatilities
    ## - for each sample, get posterior mean and CI of persistence statistic
    ## - compare to persistence of DGP
    if (length(filenames)!=4)
        stop("need to provide four filenames for MCMC, SSVS, GVS, and RJMCMC")
    cat("analyze persistence statistics in simulations results...\n")
    h.irf <- 60
    p.q <- c(0.025, 0.975)
    pers.fns <- list(
        ## persistence statistics
        function(pars) max(abs(eigen(pars$Phi)$values)),
        function(pars) irf.var1(pars$Phi, h.irf)[h.irf],
        function(pars) {
            Bfrn <- 2*pars$Brn[,mats==120] - pars$Brn[,mats==60]
            1200*sqrt(diag(t(Bfrn)%*%pars$VardcP%*%Bfrn))
        },
        function(pars) {
            Bf <- 2*pars$B[,mats==120] - pars$B[,mats==60]
            Bfrn <- 2*pars$Brn[,mats==120] - pars$Brn[,mats==60]
            1200*sqrt(diag(t(Bf-Bfrn)%*%pars$VardcP%*%(Bf-Bfrn)))
        })
    load(filenames[1])  ## load now so that we have N and DGP
    tbl <- matrix(NA, 1+4*2, length(pers.fns))
    rownames(tbl) <- c("DGP", "MCMC-mean", "MCMC-CI", "SSVS-mean", "SSVS-CI", "GVS-mean", "GVS-CI", "RJMCMC-mean", "RJMCMC-CI")
    colnames(tbl) <- c('max eq-P', 'IRF-P(5y)', 'vol(frn)', 'vol(ftp)')
    ## DGP
    true.pars$PhiQ <- true.pars$loads$K1Q.cP+diag(N)
    true.pars$B <- true.pars$loads$BcP
    true.pars$VarcP <- matrix( solve(diag(N^2) - kronecker(true.pars$Phi, true.pars$Phi))%*%as.numeric(true.pars$Omega), N, N)
    true.pars$VardcP <- (diag(N) - true.pars$Phi) %*% true.pars$VarcP %*% t(diag(N) - true.pars$Phi) + true.pars$Omega
    true.pars$Brn <- gaussian.loadings(mats, true.pars$mu, true.pars$Phi-diag(N), true.pars$Omega, true.pars$loads$rho0.cP, true.pars$loads$rho1.cP)$B
    true.model <- vec2str(true.pars$gamma)
    pers.true <- sapply(pers.fns, function(f) f(true.pars))
    tbl[1,] <- pers.true

    Bf <- 2*true.pars$B[,mats==120] - true.pars$B[,mats==60]
    cat("True volatility:", 1200*sqrt(diag(t(Bf)%*%true.pars$VardcP%*%Bf)), "\n")

    row <- 2
    ## simulations
    for (filename in filenames) {
        cat(filename, "\n")
        load(filename)
        if (!all.equal(vec2str(true.pars$gamma), true.model))
            stop("Different DGP model!")
        pers.means <- matrix(NA, n, length(pers.fns))
        pers.contained <- matrix(NA, n, length(pers.fns))
        for (j in 1:n) {
            if (j %% 10 == 0)
                cat("sample", j,"\n")
            ind <- seq((j-1)*M+1, j*M, 10)  ## thin out for speed
            pars.smpl <- getMCMCpars(kinfQ.i[ind], lamQ.i[ind,], lambda.i[ind,], Omega.i[ind,], mats, W, flag.mom=T, flag.rn=T, flag.forw=F)
            col <- 0
            for (fn in pers.fns) {
                col <- col+1
                pers.dist <- na.omit(sapply(pars.smpl, function(pars) fn(pars)))
                pers.means[j,col] <- mean(pers.dist)
                pers.contained[j,col] <- prod(pers.true[col]-quantile(pers.dist, p.q))<=0
            }
        }
        tbl[row,] <- colMeans(pers.means)
        tbl[row+1,] <- colMeans(pers.contained)
        row <- row+2
        print(round(tbl, digi=4))
    }
    cat("True volatility:", 1200*sqrt(diag(t(Bf)%*%pars$VardcP%*%Bf)), "\n")
    print(round(tbl, digi=4))
    tbl
}


printAllSimResultsPersToLatex <- function(tbl, to.file=F, filename) {
    ## latex table
    if (to.file) {
        if (missing(filename)) {
            filename <- "tables/sim_pers.tex"
        }
        cat("*** writing table ", filename, "\n")
        sink(filename)
    }
        cat("\\begin{tabular}{llccccc}\n \\hline \\hline \n")
        cat(" & & \\multicolumn{2}{c}{Persistence} & \\multicolumn{2}{c}{Volatilities}  \\\\ \n")
        cat(" & & max.~eigenv. & IRF(5y) & $\\Delta \\tilde{f}_t$ & $\\Delta ftp_t$ \\\\ \n \\hline \n")

        cat("DGP & ", sprintf("& %5.4f ", tbl[1,]), "\\\\ \n \\hline \n")
        row <- 2
        for (s in c("MCMC", "SSVS", "GVS", "RJMCMC")) {
            cat(s, "& posterior mean ", sprintf("& %5.4f ", tbl[row,]), " \\\\ \n")
            cat("     & CI contains DGP ", sprintf("& %2.0f\\%% ", 100*tbl[row+1,]), "\\\\ \n  \\hline \n")
            row <- row+2
        }
        cat("\\hline \n \\end{tabular}\n")
    if (to.file)
        sink()
}

printParameterEstimates <- function(filename, to.latex=FALSE) {
    ## display parameter estimates for MCMC estimates (typically URP)
    cat("Showing results for", filename, "\n")
    load(filename)
    cat("MCMC iterations: ", M, "\n")
    mcmc.ind <- getMCMCind(M)

    ## get posterior means, SDs, quantiles of kinfQ, lamQ, lambda, Sigma
    Sigma.i <- t(apply( Omega.i, 1, function(z) as.numeric(t(chol(matrix(z, N, N)))) ))
    pars <- list(kinfQ = mean(kinfQ.i[mcmc.ind]),
                 lamQ = colMeans(lamQ.i[mcmc.ind,])+1,
                 lambda = colMeans(lambda.i[mcmc.ind,]),
                 Sigma = matrix( colMeans(Sigma.i[mcmc.ind,]), N, N ),
                 sigma.e = mean(sqrt(sige2.i[mcmc.ind])))

    ## posterior standard deviations
    pars.sd <- list(kinfQ = sd(kinfQ.i[mcmc.ind]),
                    lamQ = apply(lamQ.i[mcmc.ind,], 2, sd),
                    lambda = apply(lambda.i[mcmc.ind,], 2, sd),
                    Sigma = matrix( apply(Sigma.i[mcmc.ind,], 2, sd), N, N),
                    sigma.e = sd(sqrt(sige2.i[mcmc.ind])))

    ## credibility intervals
    q.p <- c(0.025, 0.975)
    lower.tri.ind <- as.numeric(which(lower.tri(matrix(0,N,N), diag=T)))
    pars.q <- list(kinfQ = quantile(kinfQ.i[mcmc.ind], q.p),
                   lamQ = apply(lamQ.i[mcmc.ind,], 2, quantile, probs=q.p)+1,
                   lambda = apply(lambda.i[mcmc.ind,], 2, quantile, probs=q.p),
                   Sigma = apply(Sigma.i[mcmc.ind,lower.tri.ind], 2, quantile, probs=q.p),
                   sigma.e = quantile(sqrt(sige2.i[mcmc.ind]), q.p))

    ## simple table for display --- only posterior
    no.pars <- 1+N+N+N^2+N*(N+1)/2+1
    out.matrix <- matrix(NA, no.pars, 6)
    rownames(out.matrix) <- c(getParameterNames(N, include.P=FALSE, include.ev=FALSE), "sigma.e")
    colnames(out.matrix) <- c('mean', 'sd', 't-stat', '2.5%', '97.5%', 'sig.')
    ind.lamQ <- 2:(N+1)
    ind.lambda <- (max(ind.lamQ)+1):(max(ind.lamQ)+N+N^2)
    ind.Sigma <- (max(ind.lambda)+1):(max(ind.lambda)+(N*(N+1)/2))
    ## posterior mean
    out.matrix[1, 1] <- pars$kinfQ*1200
    out.matrix[ind.lamQ, 1] <- pars$lamQ
    out.matrix[ind.lambda, 1] <- pars$lambda
    out.matrix[ind.Sigma, 1] <- pars$Sigma[lower.tri(pars$Sigma, diag=TRUE)]*1200
    out.matrix[nrow(out.matrix), 1] <- pars$sigma.e*1200
    ## posterior SD
    out.matrix[1, 2] <- pars.sd$kinfQ*1200
    out.matrix[ind.lamQ, 2] <- pars.sd$lamQ
    out.matrix[ind.lambda, 2] <- pars.sd$lambda
    out.matrix[ind.Sigma, 2] <- pars.sd$Sigma[lower.tri(pars$Sigma, diag=TRUE)]*1200
    out.matrix[nrow(out.matrix), 2] <- pars.sd$sigma.e*1200
    ## t-stats
    out.matrix[,3] <- abs(out.matrix[,1])/out.matrix[,2]
    ## quantiles
    out.matrix[1, 4:5] <- pars.q$kinfQ*1200
    out.matrix[ind.lamQ, 4:5] <- t(pars.q$lamQ)
    out.matrix[ind.lambda, 4:5] <- t(pars.q$lambda)
    out.matrix[ind.Sigma, 4:5] <- t(pars.q$Sigma*1200)
    out.matrix[nrow(out.matrix), 4:5] <- pars.q$sigma.e*1200
    ## "significance"
    out.matrix[,6] <- out.matrix[,4]*out.matrix[,5]>0
    ## print comparison table
    print(round(out.matrix, digi=4))

    ## table with prior-posterior comparison
    tbl <- matrix(NA, 20, 8)
    rownames(tbl) <- c('kinfQ*1200','','lamQ','', 'lam0*1200','', 'lam1','','','','','',
                       'Sigma*1200','','','','','', 'sigma.e*1200', '')
    colnames(tbl) <- c('','Prior','','','Posterior','','acc.','ineff.')
    tbl.sig <- matrix(0, 20, 8)
    col.alpha <- N+N+1
    col.ineff <- col.alpha+1
    col.prior <- 1:N
    col.post <- (N+1):(2*N)
    row.lam1 <- seq(6+1, 6+N*2, 2)
    row.Sigma <- seq(max(row.lam1+2), max(row.lam1)+1+2*N, 2)
    row.sige <- max(row.Sigma)+2
    ## kinfQ
    tbl[1,min(col.post)] <- pars$kinfQ*1200
    tbl.sig[1, min(col.post)] <- prod(pars.q$kinfQ)>0
    tbl[2,min(col.post)] <- pars.sd$kinfQ*1200
    tbl[1,col.alpha] <- mean(alpha.kinfQ[mcmc.ind])*100
    tbl[1,col.ineff] <- ineff(kinfQ.i[mcmc.ind])
    ## lamQ
    tbl[3,col.prior] <- (priors$lamQ.max+priors$lamQ.min)/2+1
    tbl[4,col.prior] <- sqrt(1/12*(priors$lamQ.max-priors$lamQ.min)^2)
    tbl[3,col.post] <- pars$lamQ
    tbl.sig[3, col.post] <- apply(pars.q$lamQ, 2, prod)>0
    tbl[4,col.post] <- pars.sd$lamQ
    tbl[3,col.alpha] <- mean(alpha.lamQ[mcmc.ind])*100
    tbl[3,col.ineff] <- mean(apply(lamQ.i[mcmc.ind,], 2, ineff))
    ## lam0
    tbl[5,col.prior] <- 0
    tbl[6,col.prior] <- priors$lambda.sd[1:N]*1200
    tbl[5,col.post] <- pars$lambda[1:N]*1200
    tbl.sig[5,col.post] <- apply(pars.q$lambda[,1:N], 2, prod)>0
    tbl[6,col.post] <- pars.sd$lambda[1:N]*1200
    tbl[5,col.ineff] <- mean(apply(lambda.i[mcmc.ind,1:N], 2, ineff))
    ## lam1
    tbl[row.lam1,col.prior] <- 0
    tbl[row.lam1+1,col.prior] <- tail(priors$lambda.sd, N^2)
    tbl[row.lam1,col.post] <- tail(pars$lambda, N^2)
    tbl.sig[row.lam1,col.post] <- apply(pars.q$lambda[,(N+1):(N+N^2)], 2, prod)>0
    tbl[row.lam1+1,col.post] <- tail(pars.sd$lambda, N^2)
    tbl[min(row.lam1),col.ineff] <- mean(apply(lambda.i[mcmc.ind,(N+1):(N+N^2)], 2, ineff))
    ## Sigma
    tbl[row.Sigma,col.post] <- pars$Sigma*1200
    Sigma.sig <- matrix(0, N, N)
    Sigma.sig[lower.tri(Sigma.sig, TRUE)] <- apply(pars.q$Sigma, 2, prod)>0
    tbl.sig[row.Sigma,col.post] <- Sigma.sig
    tbl[row.Sigma+1,col.post] <- pars.sd$Sigma*1200
    tbl[min(row.Sigma),col.alpha] <- mean(alpha.Omega[mcmc.ind])*100
    tbl[min(row.Sigma),col.ineff] <- mean(apply(Sigma.i[mcmc.ind,lower.tri.ind], 2, ineff))
    ## sigma.e
    tbl[row.sige, min(col.post)] <- mean(sqrt(sige2.i[mcmc.ind]))*1200
    tbl.sig[row.sige, min(col.post)] <- prod(pars.q$sige2.i)>0
    tbl[row.sige+1, min(col.post)] <- sd(sqrt(sige2.i[mcmc.ind]))*1200
    tbl[row.sige, col.ineff] <- ineff(sqrt(sige2.i[mcmc.ind]))
    print(round(tbl, digi=4))
    ## print(tbl.sig)

    ## latex table
    if (to.latex) {
        filename <- "tables/parameters_urp.tex"
        cat("*** writing table ", filename, "\n")
        sink(filename)
        rownames(tbl) <- c('$k_\\infty^\\mathds{Q}$','','$\\phi^\\mathds{Q}$','', '$\\lambda_0$','', '$\\lambda_1$','','','','','', '$\\Sigma$','','','','','', '$\\sigma_e$', '')
        cat("\\begin{tabular}{lcccccccc} \\hline \\hline \n")
        cat("Parameter & \\multicolumn{3}{c}{Prior} & \\multicolumn{3}{c}{Posterior} & Acc. & Ineff. \\\\ \\hline \n")
        for (row in 1:nrow(tbl)) {
            cat(rownames(tbl)[row], "&")
            for (col in 1:ncol(tbl)) {
                if (row %% 2 == 1) {
                    ## print parameter
                    if (!is.na(tbl[row,col])) {
                        ## is significant?
                        if (tbl.sig[row,col]) {
                            cat(sprintf("\\textbf{%6.4f}", tbl[row,col])) ## significant non-zero parameter
                        } else {
                            if (isTRUE(all.equal(tbl[row,col],0))) {
                                cat("0")  				## zero parameter
                            } else if (col==col.alpha) {
                                cat(sprintf("%4.1f", tbl[row,col]))  	## acceptance rate
                            } else if (col==col.ineff) {
                                cat(sprintf("%4.1f", tbl[row,col]))  	## inefficiency factor
                            } else {
                                cat(sprintf("%6.4f", tbl[row,col]))  	## non-zero parameter
                            }
                        }
                    }
                } else {
                    ## print standard error
                    if (!(is.na(tbl[row,col])||isTRUE(all.equal(tbl[row,col],0)))) {
                        cat(sprintf("(%6.4f)", tbl[row,col]))
                    }
                }
                if(col<ncol(tbl)) cat("&")
            }
            cat("\\\\ \n")
        }
        cat("\\hline \\hline \n \\end{tabular}\n")
        sink()
    }

}

getMCMCpars <- function(kinfQ.i, lamQ.i, lambda.i, Omega.i, mats, W, flag.mom=T, flag.rn=T, flag.forw=T) {
    ## returns posterior distribution of model parameters and loadings
    ## Arguments:  posterior distribution of primitive model parameters
    ##   kinfQ.i  - M-vector
    ##   lamQ.i   - MxN matrix
    ##   lambda.i - Mx(N^2+N) matrix
    ##   Omega.i  - MxN^2 matrix
    ## Value: list of lists
    ##   pars - posterior distribution of parameters and loadings
    ## Globals: mats, W
##    cat("calculating posterior distribution of model parameters and loadings...\n")
    M <- length(kinfQ.i)
    pars <- list()
    N <- ncol(lamQ.i)
    for (i in 1:M) {
        if (i %% 1000 == 0)
            cat("... iteration", i, "\n")
        pars[[i]] <- list(kinfQ=kinfQ.i[i],
                          lamQ=lamQ.i[i,],
                          lambda=lambda.i[i,],
                          Omega=matrix(Omega.i[i,], N, N))
        pars[[i]]$Sigma <-  t(chol(pars[[i]]$Omega))
        loads <- jsz.loadings(W, diag(lamQ.i[i,]), kinfQ.i[i], pars[[i]]$Omega, mats, dt=1)
        pars[[i]]$A <- loads$AcP; pars[[i]]$B <- loads$BcP
        Pdyn <- getPdyn(loads, lambda.i[i,])
        pars[[i]]$mu <- Pdyn$mu
        pars[[i]]$Phi <- Pdyn$Phi
        pars[[i]]$muQ <- loads$K0Q.cP
        pars[[i]]$PhiQ <- loads$K1Q.cP + diag(N)
        pars[[i]]$rho0 <- loads$rho0.cP
        pars[[i]]$rho1 <- loads$rho1.cP
        pars[[i]]$Lam0 <- as.numeric(pars[[i]]$lambda[1:N])
        pars[[i]]$Lam1 <- matrix( tail(pars[[i]]$lambda, -N), N, N)
        if (flag.mom) {
            ## unconditional moments
            pars[[i]]$EcP <- as.numeric(solve(diag(N) - pars[[i]]$Phi) %*% pars[[i]]$mu)
            pars[[i]]$VarcP <- matrix( solve(diag(N^2) - kronecker(pars[[i]]$Phi, pars[[i]]$Phi))%*%as.numeric(pars[[i]]$Omega), N, N)
            pars[[i]]$VardcP <- (diag(N) - pars[[i]]$Phi) %*% pars[[i]]$VarcP %*% t(diag(N) - pars[[i]]$Phi) + pars[[i]]$Omega
        }
        if (flag.rn) {
            ## loadings risk-neutral yields
            loads.rn <- gaussian.loadings(mats, pars[[i]]$mu, pars[[i]]$Phi-diag(N), pars[[i]]$Omega, pars[[i]]$rho0, pars[[i]]$rho1)
            pars[[i]]$Arn <- loads.rn$A; pars[[i]]$Brn <- loads.rn$B
        }
        if (flag.forw) {
            ## loadings forward rates
            H <- max(mats)
            loadsf <- getForwLoads(1:H, pars[[i]]$muQ, pars[[i]]$PhiQ, pars[[i]]$Omega, pars[[i]]$rho0, pars[[i]]$rho1)
            loadsfrn <- getForwLoads(1:H, pars[[i]]$mu, pars[[i]]$Phi, pars[[i]]$Omega, pars[[i]]$rho0, pars[[i]]$rho1)
            pars[[i]]$Af <- loadsf$A; pars[[i]]$Bf <- loadsf$B
            pars[[i]]$Afrn <- loadsfrn$A; pars[[i]]$Bfrn <- loadsfrn$B
        }
    }
    return(pars)
}

getPostMeans <- function(pars.smpl) {
    J <- length(pars.smpl[[1]]$A)
    H <- length(pars.smpl[[1]]$Af)
    return(list(kinfQ = mean(sapply(pars.smpl, function(z) z$kinfQ)),
                lamQ = rowMeans(sapply(pars.smpl, function(z) z$lamQ)),
                lambda = rowMeans(sapply(pars.smpl, function(z) z$lambda)),
                Omega = matrix(rowMeans(sapply(pars.smpl, function(z) z$Omega)), N, N),
                Sigma = matrix(rowMeans(sapply(pars.smpl, function(z) z$Sigma)), N, N),
                mu = rowMeans(sapply(pars.smpl, function(z) z$mu)),
                Phi = matrix(rowMeans(sapply(pars.smpl, function(z) z$Phi)), N, N),
                muQ = rowMeans(sapply(pars.smpl, function(z) z$muQ)),
                PhiQ = matrix(rowMeans(sapply(pars.smpl, function(z) z$PhiQ)), N, N),
                rho0 = mean(sapply(pars.smpl, function(z) z$rho0)),
                rho1 = rowMeans(sapply(pars.smpl, function(z) z$rho1)),
                Lam0 = rowMeans(sapply(pars.smpl, function(z) z$Lam0)),
                Lam1 = matrix(rowMeans(sapply(pars.smpl, function(z) z$Lam1)), N, N),
                A = matrix(rowMeans(sapply(pars.smpl, function(z) z$A)), 1, J),
                B = matrix(rowMeans(sapply(pars.smpl, function(z) z$B)), N, J),
                Arn = matrix(rowMeans(sapply(pars.smpl, function(z) z$Arn)), 1, J),
                Brn = matrix(rowMeans(sapply(pars.smpl, function(z) z$Brn)), N, J),
                Af = matrix(rowMeans(sapply(pars.smpl, function(z) z$Af)), 1, H),
                Bf = matrix(rowMeans(sapply(pars.smpl, function(z) z$Bf)), N, H),
                Afrn = matrix(rowMeans(sapply(pars.smpl, function(z) z$Afrn)), 1, H),
                Bfrn = matrix(rowMeans(sapply(pars.smpl, function(z) z$Bfrn)), N, H)))
}

getMCMCind <- function(M, thin=1) {
    M0 <- 5001 ## M/2 + 1
    if (thin>1) {
        return(seq(M0, M, thin))
    } else {
        return(M0:M)
    }
}

printGammaStats <- function(filenames, to.latex=FALSE) {
    ## summary statistics for gamma -- across models
    ## 1) posterior mean
    ## 2) Monte Carlo standard error
    ## 3) Inefficiency factor
    load(filenames[1]) ## to get N
    out.matrix <- matrix(NA, N+N^2, 3*length(filenames))
    colnames(out.matrix) <- rep(c("Mean", "MCSE", "Ineff."), length(filenames))
    for (i in 1:length(filenames)) {
        if (i>1)
            load(filenames[i])		## load estimation results
        mcmc.ind <- getMCMCind(M)
        gamma.i <- gamma.i[mcmc.ind,]	## drop burn-in iterations
        ## describe distribution of gamma
        out.matrix[,(i-1)*3+1] <- colMeans(gamma.i)
        gamma.lrv <- lrv(gamma.i)
        out.matrix[,(i-1)*3+2] <- sqrt(gamma.lrv/length(mcmc.ind)) ## numerical SE
        out.matrix[,(i-1)*3+3] <- gamma.lrv/apply(gamma.i, 2, var) ## inefficiency factor (alt: ineff(gamma.i))
    }
    ## ## compate to ''t-stats'' from unrestricted model
    ## load.list <- load(filename.mcmc)
    ## t.stats <- abs(colMeans(lambda.i[mcmc.ind,])/apply(lambda.i[mcmc.ind,], 2, sd)),
    print(round(out.matrix,digi=3))
    if (to.latex) {
        require(xtable)
        latex.tbl <- xtable(out.matrix, digits=c(NA, rep(c(3,3,1), length(filenames))))
        filename <- "tables/gamma.tex"
        cat("*** writing table ", filename, "\n")
        sink(filename)
        print(latex.tbl, include.rownames=TRUE, include.colnames=TRUE, only.contents=TRUE)
        sink()
    }
}

printModelFreq <- function(filename, flag.plot=FALSE) {
    ## show which models are visited most frequently
    ## for one set of results
    cat("showing results for", filename, "\n")
    load(filename)			## load estimation results
    mcmc.ind <- getMCMCind(M)
    gamma.i <- gamma.i[mcmc.ind,]	## drop burn-in iterations
    ## which models occur most often?
    model.i <- apply(gamma.i, 1, vec2str) ## string describing the model in each iteration, e.g. "101010"
    freq <- sort(table( model.i ), decreasing = T) 	## sorted table with absolute frequencies
    freq.data <- as.data.frame(freq)
    freq.data$rel.freq <- freq/sum(freq)			## add relative frequency
    freq.data <- cbind(freq.data, t(sapply(names(freq), str2vec)))  ## add columns with indicators
    ## show 10 most frequent models
    print(freq.data[1:10,])
    cat("most frequently visited model: ", names(freq)[1], "\n")
    cat("number of models with freq. at least 1%:", sum(freq.data$rel.freq>=.01), "\n")
    ## how many models are visited by the sampler at least once?
    P <- dim(gamma.i)[2]
    K <- 2^P # number of possible models (2^9=512)
    cat("Number of models visited: ", length(freq), "/", K, "=", length(freq)/K*100, "%\n")
    ## plot distribution of model frequencies
    rel.prob <- freq/freq[1]
    if (flag.plot) {
        dev.new()
        plot(rev(rel.prob), main="Relative model frequency", ylab="Model frequency", xlab="Model")
    }
    cat("prior vs. posterior probability of at most n unrestricted parameters:\n")
    for (i in 1:12)
        cat(i, " - ", pbinom(0:i, 12, .5)[i], (sum(rowSums(gamma.i)<=i))/nrow(gamma.i), "\n")
    cat("Posterior mean number of unrestricted risk prices:",
        mean(rowSums(gamma.i)), "\n")
}

printModels <- function(filenames, all.models, to.latex=FALSE) {
    ## display most frequently visited models
    ## - frequencies and Bayes factors for several estimation results
    ## first set of estimate sdetermines model order in the table
    n <- 10  ## how many models in the table?
    load(filenames[1])
    mcmc.ind <- getMCMCind(M)
    gamma.i <- gamma.i[mcmc.ind,]	## drop burn-in iterations
    model.i <- apply(gamma.i, 1, vec2str) ## string describing model in each iteration
    freq <- sort(table( model.i ), decreasing = T)/length(mcmc.ind) ## sorted table with absolute frequencies
    tbl <- as.data.frame(freq[1:n])
    colnames(tbl) <- "freq.1"
    tbl$odds.1 <- tbl$freq.1[1]/tbl$freq.1
    visited <- numeric(length(filenames))
    visited[1] <- length(freq)
    ## remaining estimates get added to the table
    for (i in 2:length(filenames)) {
        load(filenames[i])
        gamma.i <- gamma.i[mcmc.ind,]
        model.i <- apply(gamma.i, 1, vec2str)
        rel.freq <- sapply(rownames(tbl), function(model) sum(model.i==model)/length(mcmc.ind))
        visited[i] <- length(unique(model.i))
        freq.colname <- paste('freq.', as.character(i), sep="")
        tbl[[freq.colname]] <- rel.freq
        tbl[[paste('odds.', as.character(i), sep="")]] <- tbl[[freq.colname]][1]/tbl[[freq.colname]]
    }
    rownames(tbl) <- sapply(rownames(tbl), function(model) paste(which(str2vec(model)==1), collapse=","))
    r <- match(names(freq[1:n]), rownames(all.models))
    tbl <- cbind(tbl, all.models[r, c("AIC", "BIC")])
    print(tbl)
    ## formatted latex table for paper
    if (to.latex) {
        require(xtable)
        N <- length(filenames)
        latex.tbl <- xtable(tbl, digits=c(NA, rep(c(4,1), N), 1, 1))
        filename <- "tables/models.tex"
        cat("*** writing table ", filename, "\n")
        sink(filename)
        print(latex.tbl, include.rownames=TRUE, include.colnames=FALSE, only.contents=TRUE,
              hline.after=nrow(tbl))
        cat("models ")
        for (i in 1:N)
            cat("& \\multicolumn{2}{c|}{", visited[i], "/", 2^(N+N^2),"}")
        cat("\\\\ \n")
        cat("visited ")
        for (i in 1:N)
            cat("& \\multicolumn{2}{c|}{", round(100*visited[i]/2^(N+N^2), digi=1), "\\%}")
        cat("\\\\ \n")
        cat("\\hline \n")
        sink()
    }
    return(tbl)
}

loadModel <- function(name, filename) {
    cat("*** loading model", name, "\n")
    cat("*** results file:", filename, "\n")
    load(filename)
    if (!all.equal(get("dates", ".GlobalEnv"), dates) |
        !all.equal(get("W", ".GlobalEnv"), W) |
        !all.equal(get("mats", ".GlobalEnv"), mats)) {
        cat("saved data different from data in memory\n")
        print(range(dates))
        print(range(get("dates", ".GlobalEnv")))
        print(all.equal(get("dates", ".GlobalEnv"), dates))
        print(W)
        print(get("W", ".GlobalEnv"))
        print(all.equal(get("W", ".GlobalEnv"), W))
        print(mats)
        print(get("mats", ".GlobalEnv"))
        print(all.equal(get("mats", ".GlobalEnv"), mats))
        stop("loadModel: saved estimates are for data set different from the one in memory")
    }
    if (is.vector(gamma)) {
        cat("*** model specification:", vec2str(gamma), "\n")
    } else {
        cat("*** Bayesian Model Averaging\n")
    }
    mcmc.ind = getMCMCind(M, thin=10)
    model <- list(
        name = name,
        ## list with distribution of all model parameters and loadings
        pars.smpl = getMCMCpars(kinfQ.i[mcmc.ind], lamQ.i[mcmc.ind,], lambda.i[mcmc.ind,],
        Omega.i[mcmc.ind,], mats, W),
        W = W, N = N, M = M, mats = mats, J = length(mats),
        cP = Y %*% t(W), T = length(dates))
    model <- within(model, {
        ## posterior means of parameters
        pars <- getPostMeans(pars.smpl)
        ## model-implied short rate
        r <- rep(1,T)*pars$rho0 + cP %*% pars$rho1
        ## UNCONDITIONAL MOMENTS
        ## (1) moments at point estimates
        EcP <- as.numeric(solve(diag(N) - pars$Phi) %*% pars$mu)
        VarcP <- matrix( solve(diag(N^2) - kronecker(pars$Phi, pars$Phi))%*%as.numeric(pars$Omega), N, N)
        ## (2) posterior mean of moments
        pars$EcP <- rowMeans(sapply(pars.smpl, function(z) z$EcP))
        pars$VarcP <- matrix(rowMeans(sapply(pars.smpl, function(z) z$VarcP)), N, N)
        pars$VardcP <- matrix(rowMeans(sapply(pars.smpl, function(z) z$VardcP)), N, N)
        ## -> difference between (1) and (2) quite substantial for VarcP because of skewness
        ## fitted yields and term premia
        Yhat <- rep(1,T)%*%pars$A + cP%*%pars$B
        Yrn <-  rep(1,T)%*%pars$Arn + cP%*%pars$Brn
        Ytp <- Yhat - Yrn
        ## forward rates and term premia
        H <- max(mats)
        fhat <- rep(1,T) %*% pars$Af + cP %*% pars$Bf
        frn <- rep(1,T) %*% pars$Afrn + cP %*% pars$Bfrn
        ftp <- fhat-frn
    })
    return(model)
}

analyzeFit <- function(model) {
    cat("Cross-sectional fit for model", model$name, "\n")
    with(model, {
        rmse.smpl <- sapply(pars.smpl, function(pars) {
            Yhat <- rep(1,T)%*%pars$A + cP%*%pars$B
            return(sqrt(mean((Y-Yhat)^2))*120000)
        })
        cat("Root-mean-square error in bps: ", mean(rmse.smpl), "\n")
        print(summary(rmse.smpl))
    })
}

printPersVol <- function(models, to.latex = FALSE) {
    ## table summarizing persistence and volatility across models
    ##  - both P and Q measure
    K <- length(models)
    J <- length(models[[1]]$mats)
    h.irf <- 120
    h.vol <- 120
    N <- models[[1]]$N
    stat.fns <- list(
        function(pars) max(abs(eigen(pars$PhiQ)$values)),
        function(pars) max(abs(eigen(pars$Phi)$values)),
        function(pars) {
            Bf <- 2*pars$B[,mats==120] - pars$B[,mats==60]
            return(1200*sqrt(diag(t(Bf)%*%pars$VardcP%*%Bf)))
        },
        function(pars) {
            Bfrn <- 2*pars$Brn[,mats==120] - pars$Brn[,mats==60]
            return(1200*sqrt(diag(t(Bfrn)%*%pars$VardcP%*%Bfrn)))
        },
        function(pars) {
            Bf <- 2*pars$B[,mats==120] - pars$B[,mats==60]
            Bfrn <- 2*pars$Brn[,mats==120] - pars$Brn[,mats==60]
            return(1200*sqrt(diag(t(Bf-Bfrn)%*%pars$VardcP%*%(Bf-Bfrn))))
        })
    L <- length(stat.fns)
    tbl <- matrix(NA, K*2, L*2)
    colnames(tbl) <- c('Q-EV', '', 'P-EV', '',
                       'vol(f)', '', 'vol(frn)', '', 'vol(ftp)', '')
    A <- cbind(rep(names(models), each=2), rep(c("mean/median","lb/ub"), K))
    rownames(tbl) <- apply(A, 1, function(z) paste(z[1], z[2], sep="-"))
    row <- 1
    for (m in models) {
        col <- 1
        for (i in 1:L) {
            stat.fn <- stat.fns[[i]]
            dist <- sapply(m$pars.smpl, function(pars) stat.fn(pars))
            tbl[row,col] <- mean(dist, na.rm=T)
            tbl[row,col+1] <- median(dist, na.rm=T)
            tbl[row+1, col:(col+1)] <- quantile(dist, c(0.025, 0.975), na.rm=T)
            col <- col+2
        }
        row <- row+2
    }
    if (to.latex) {
        filename <- "tables/persistence.tex"
        cat("*** writing table", filename, "\n")
        sink(filename)
        cat("\\begin{tabular}{l|cc|ccc} \\hline \\hline \n")
        cat("Model & \\multicolumn{2}{c|}{Max.~eigenvalue} & \\multicolumn{3}{c}{Volatilities} \\\\ \n")
        cat("      & $\\mathds{Q}$ & $\\mathds{P}$ & $\\Delta \\hat{f}_t$ & $\\Delta \\tilde{f}_t$ & $\\Delta ftp_t$ \\\\ \\hline \n")
        for (row in 1:nrow(tbl)) {
            if (row %% 2 == 1) {
                cat(names(models)[ceiling(row/2)])
                ## point estimates
                ## cat(sprintf("& %6.4f (%6.4f) ", tbl[row,c(1,3)], tbl[row,c(2,4)]),
                ##     sprintf("& %4.2f (%4.2f) ", tbl[row,c(5,7,9)], tbl[row,c(6,8,10)]),
                cat(sprintf("& %6.4f ", tbl[row,c(1,3)]),
                    sprintf("& %4.2f ", tbl[row,c(5,7,9)]),
                    "\\\\ \n")
            } else {
                ## credibility intervals
                cat(sprintf("& [%6.4f, %6.4f]", tbl[row,1], tbl[row,2]))
                cat(sprintf("& [%6.4f, %6.4f]", tbl[row,3], tbl[row,4]))
                for (col in seq(5,ncol(tbl)-1,2))
                    cat(sprintf("& [%4.2f, %4.2f]", tbl[row,col], tbl[row,col+1]))
                cat("\\\\ \n \\hline \n")
            }
        }
        cat("\\hline \n \\end{tabular}\n")
        sink()
    } else {
        print(round(tbl, digi=3))
    }
}

plotVolatilities <- function(models, to.file=FALSE) {
    K <- length(models)
    H <- length(models[[1]]$pars$Af)
    if (to.file) {
        filename <- "figures/volatilities.eps"
        postscript(filename, width=6.5, height=7, horizontal=FALSE, pointsize=12)
        cat("*** writing figure", filename, "\n")
    } else {
        dev.new()
    }
    lwds <- c(1,2,2)
    ltys <- c(1,1,2)
    colors <- c("black", "blue", "blue")
##    par(mfrow = c(ceiling(K/2), 2))
    par(mfrow = c(K, 1))
    yrange <- c(0, 0.52)
    model.names <- names(models)  ## c("M_0", "M_1", "M_2", "BMA")
    model.names <- gsub("M0", "M_0", model.names)
    cP <- Y %*% t(W)
    VardcP <- cov(diff(cP))
    for (k in 1:K) {
        par(mar = c(4,4,2,1)+.1)
        vol.smpl <- sapply(models[[k]]$pars.smpl,  function(pars)
                           sqrt(diag(t(pars$Bf)%*%VardcP%*%pars$Bf)))
        volrn.smpl <- sapply(models[[k]]$pars.smpl,  function(pars)
                             sqrt(diag(t(pars$Bfrn)%*%VardcP%*%pars$Bfrn)))
        plot(1:H, 1200*apply(vol.smpl, 1, mean), ylim=yrange, type="l",
             xlab="Horizon", ylab="Percent", lwd=lwds[1], lty=ltys[1], col=colors[1])
        if (k==1) {
            title(expression(M[0]))
        } else {
            title(expression(BMA))
        }
        lines(1:H, 1200*apply(volrn.smpl, 1, median), lwd=lwds[2], lty=ltys[2], col=colors[2])
        lines(1:H, 1200*apply(volrn.smpl, 1, quantile, 0.025), lwd=lwds[3], lty=ltys[3], col=colors[3])
        lines(1:H, 1200*apply(volrn.smpl, 1, quantile, 0.975), lwd=lwds[3], lty=ltys[3], col=colors[3])
    }
    if (to.file)
        dev.off()
}

printAvgFactors <- function(models) {
    cat("Mean risk factors\n")
    K <- length(models)
    tbl <- matrix(NA, 1+K*2, 3)
    rownames(tbl) <- c("Actual", apply(cbind(rep(names(models),each=2), rep(c("uncond. mean at point est.", "post. mean of uncond. mean"))), 1, paste, collapse="-"))
    colnames(tbl) <- c("level", "slope", "curve")
    cP <- Y%*%t(W)
    tbl[1,] <- colMeans(cP)
    row <- 2
    for (m in models) {
        tbl[row, ] <- m$EcP
        tbl[row+1, ] <- m$pars$EcP
        row <- row+2
    }
    print(round(1200*tbl, digi=3))
}

printAvgYieldCurves <- function(models, sample.mean=TRUE) {
    cat("Yield curve across models\n")
    K <- length(models)
    J <- length(models[[1]]$mats)
    tbl <- matrix(NA, 1+K*3, J)
    rownames(tbl) <- c("Actual", apply(cbind(rep(names(models),each=3), rep(c("fitted", "risk-neutral", "term premium"))), 1, paste, collapse="-"))
    colnames(tbl) <- mats/12
    tbl[1,] <- 1200*colMeans(Y)
    row <- 2
    for (m in models) {
        if (sample.mean) {
            cP.bar <- colMeans(m$cP)
        } else {
            cP.bar <- m$EcP
        }
        for (j in 1:J) {
            tbl[row,j] <- 1200*(m$pars$A[j] + m$pars$B[,j] %*% cP.bar)
            tbl[row+1,j] <- 1200*(m$pars$Arn[j] + m$pars$Brn[,j] %*% cP.bar)
        }
        tbl[row+2,] <- tbl[row,]-tbl[row+1,]
        row <- row+3
    }
    print(round(tbl, digi=3))
    cat("Forward rates across models\n")
    tbl.forw <- matrix(rep(2:J, each=1+K*3), 1+K*3, J-1)*tbl[,2:J] -
        matrix(rep(1:(J-1), each=1+K*3), 1+K*3, J-1)*tbl[,1:(J-1)]
    print(round(tbl.forw, digi=3))

}

plotExpTP <- function(models, to.file=FALSE) {
    ## plot expectations and term premium component of long term forward rate
    m2 <- 120; m1 <- 60;
    K <- length(models)
    j2 <- which(mats==m2); j1 <- which(mats==m1)
    actual <- ts( 1200*(Y[,j2]), start=start.date, frequency=12)
    fitted <- ts( 1200*(models[[1]]$Yhat[,j2]), start=start.date, frequency=12)
    if (to.file) {
        filename <- "figures/exp_tp.eps"
        postscript(filename, width=6.5, height=8.5, horizontal=FALSE, pointsize=12)
        cat("*** writing figure", filename, "\n")
    } else {
        dev.new()
    }
    lwds <- c(1,2,2,2,2,2,2,2,2)
    ltys <- c(1,1,2,3,4,5,1,2,3)
    colors <- c("black", "black", "blue", "red", "green", "gray25", "brown", "green4", "blue")
    par(mfrow = c(2, 1))
    yrange <- range(c(0, actual))
    model.names <- names(models)  ## c("M_0", "M_1", "M_2", "BMA")
    ## top panel
    par(mar = c(4,3,2,1)+.1)
    plot(actual, type="l", ylab="Percent", ylim=yrange, col=colors[1], lwd=lwds[1], lty=ltys[1])
    plot.recessions(yrange)
    lines(actual, col=colors[1], lwd=lwds[1], lty=ltys[1])
    lines(fitted, col=colors[2], lwd=lwds[2], lty=ltys[2])
    for (k in 1:K) {
        exp <- ts(1200*(models[[k]]$Yrn[,j2]), start=start.date, frequency=12)
        lines(exp, col=colors[2+k], lwd=lwds[2+k], lty=ltys[2+k])
    }
    legend("topright", c("Actual yield", "Fitted yield", paste("RN", model.names)), lwd=lwds, col=colors, lty=ltys, bg="white", cex=.9)
    title("Risk-neutral yields")
    ## bottom panel
    par(mar = c(4,3,2,1)+.1)
    plot(actual, type="l", ylab="Percent", ylim=yrange, col=colors[1], lwd=lwds[1], lty=ltys[1])
    plot.recessions(yrange)
    lines(actual, col=colors[1], lwd=lwds[1], lty=ltys[1])
    lines(fitted, col=colors[2], lwd=lwds[2], lty=ltys[2])
    for (k in 1:K) {
        ##tp <- ts(1200*(m2*models[[k]]$Ytp[,j2]-m1*models[[k]]$Ytp[,j1])/(m2-m1), start=start.date, frequency=12)
        tp <- ts(1200*(models[[k]]$Ytp[,j2]), start=start.date, frequency=12)
        lines(tp, col=colors[2+k], lwd=lwds[2+k], lty=ltys[2+k])
    }
    legend("topright", c("Actual yield", "Fitted yield", paste("TP", model.names)), lwd=lwds, col=colors, lty=ltys, bg="white", cex=.9)
    title("Term premium")
    if (to.file)
        dev.off()
}

printHistChanges <- function(models, to.latex=FALSE) {
    ## decompose historical changes over sample period and over conundrum period
    cat("Decomposition of yield changes over historical episodes\n")
    fillTable <- function(tbl) {
        ## actual rates
        ## yields
        tbl[1, col] <- 1200*mean(Y[ind.from,j2])
        tbl[1, col+2] <- 1200*mean(Y[ind.to,j2])
        tbl[1, col+4] <- tbl[1, col+2] - tbl[1, col]
        ## ## forward rates
        ## tbl[1, col] <- 1200*mean(m2*Y[ind.from,j2]-m1*Y[ind.from,j1])/(m2-m1)
        ## tbl[1, col+2] <- 1200*mean(m2*Y[ind.to,j2]-m1*Y[ind.to,j1])/(m2-m1)
        ## tbl[1, col+4] <- tbl[1, col+2] - tbl[1, col]
        ## expectations across models
        row <- 2
        for (m in models) {
            if (length(ind.from)>1) cP.from <- colMeans(m$cP[ind.from,]) else cP.from <- m$cP[ind.from,]
            if (length(ind.to)>1) cP.to <- colMeans(m$cP[ind.to,]) else cP.to <- m$cP[ind.to,]
            ## yields
            rn.from.smpl <- 1200*sapply(m$pars.smpl, function(pars)
                                        pars$Arn[,j2]+cP.from%*%pars$Brn[,j2])
            rn.to.smpl <- 1200*sapply(m$pars.smpl, function(pars)
                                      pars$Arn[,j2]+cP.to%*%pars$Brn[,j2])
            ## ## forward rates
            ## rn.from.smpl <- 1200*sapply(m$pars.smpl, function(pars)
            ##                              ( m2*(pars$Arn[,j2]+cP.from%*%pars$Brn[,j2])
            ##                               - m1*(pars$Arn[,j1]+cP.from%*%pars$Brn[,j1]) ) / (m2-m1) )
            ## rn.to.smpl <- 1200*sapply(m$pars.smpl, function(pars)
            ##                            ( m2*(pars$Arn[,j2]+cP.to%*%pars$Brn[,j2])
            ##                             - m1*(pars$Arn[,j1]+cP.to%*%pars$Brn[,j1]) ) / (m2-m1) )
            drn.smpl <- rn.to.smpl - rn.from.smpl
            tbl[row, col] <- mean(rn.from.smpl)
            tbl[row+1, col:(col+1)] <- quantile(rn.from.smpl, c(0.025, 0.975))
            tbl[row, col+2] <- mean(rn.to.smpl)
            tbl[row+1, (col+2):(col+3)] <- quantile(rn.to.smpl, c(0.025, 0.975))
            tbl[row, col+4] <- mean(drn.smpl)
            tbl[row+1, (col+4):(col+5)] <- quantile(drn.smpl, c(0.025, 0.975))
            row <- row + 2
        }
        return(tbl)
    }
    M <- length(models)
    tbl <- matrix(NA, 1+2*M, 12)
    rownames(tbl) <- c("actual",
                       apply(cbind(rep(names(models),each=2),
                                   c("expect.", "CI")), 1, paste, collapse="-"))
    colnames(tbl) <- c("1990", "", "2007", "", "Change", "",
                       "Jun-94", "", "Jun-95", "", "Change", "")
    T <- nrow(Y)
    j1 <- which(mats==60)
    j2 <- which(mats==120)
    m2 <- mats[j2]; m1 <- mats[j1]
    ## secular decline
    ind.from <- 1:12
    ind.to <- (T-11):T
    col <- 1
    tbl <- fillTable(tbl)
    ## conundrum period
    ind.from <- which(dates==20040630)
    ind.to <- which(dates==20050630)
    col <- 7
    tbl <- fillTable(tbl)
    if (to.latex) {
        filename <- "tables/histchanges.tex"
        cat("*** writing table", filename, "\n")
        sink(filename)
        cat("\\begin{tabular}{ll|ccc|ccc} \\hline \\hline \n")
        cat(" && \\multicolumn{3}{c|}{Sample period} & \\multicolumn{3}{c}{Conundrum period} \\\\ \n")
        cat(" && 1990 & 2007 & Change & Jun-94 & Jun-95 & Change  \\\\ \\hline \n")
        ## Actual forward rates
        cat("\\multicolumn{2}{l|}{Ten-year yield}", sprintf(" & %3.1f ", tbl[1, c(1,3,5,7,9,11)]), "\\\\ \n \\hline \n", sep="")
        ## Expectations
        cat("Exp. \n")
        row <- 2
        for (m in models) {
            ## point estimates
            cat("&", m$name, sprintf("& %3.1f ", tbl[row,c(1,3,5,7,9,11)]), "\\\\ \n", sep="")
            row <- row+1
            ## CIs
            cat("&", sprintf("& [%3.1f, %3.1f]", tbl[row,c(1,3,5,7,9,11)], tbl[row, c(2,4,6,8,10,12)]), "\\\\ \n", sep="")
            row <- row+1
        }
        cat("\\hline \\hline \n \\end{tabular}\n")
        sink()
    } else {
        print(round(tbl, digi=1))
    }
}

calculateReturns <- function(model) {
    return(within(model, {
        ## time-varying prices of risk
        Sig.lambda.t <- rep(1,T)%o%pars$Lam0 + cP %*% t(pars$Lam1) ## lambda_t
        ## expected return on level-mimicking portfolio
        ## xPC1 <- lambda.t[1:(T-1),1]/abs(sum(W[1,]*mats)) ## see JPS Appendix C
        ## one-period expected excess returns
        Erx <- matrix(NA, T-1, J)
        ## one-period realized excess returns (based on fitted yields)
        rx <- matrix(NA, T-1, J)
        ## annual holding period
        h <- 12
        mats.nmh <- mats[mats>h]-h  ## consider only positive maturities
        ## realized excess returns -- calculated based on ACTUAL yields
        rx.h <- matrix(NA, T-h, J)
        ## model-implied Erx.h
        Erx.h <- matrix(NA, T-h, J)
        ## posterior distribution of loadings for n-1, n-h, and h -- add to pars.smpl
        pars.smpl <- lapply(pars.smpl, function(pars) {
            loads.nm1 <- gaussian.loadings(mats-1, pars$muQ, pars$PhiQ-diag(N), pars$Omega, pars$rho0, pars$rho1)
            pars$Anm1 <- loads.nm1$A; pars$Bnm1 <- loads.nm1$B
            loads.nmh <- gaussian.loadings(mats.nmh, pars$muQ, pars$PhiQ-diag(N), pars$Omega, pars$rho0, pars$rho1)
            pars$Anmh <- loads.nmh$A; pars$Bnmh <- loads.nmh$B
            loads.h <- gaussian.loadings(h, pars$muQ, pars$PhiQ-diag(N), pars$Omega, pars$rho0, pars$rho1)
            pars$Ah <- loads.h$A; pars$Bh <- loads.h$B
            return(pars)
        })
        ## calculate posterior means of these loadings
        pars$Anm1 <- matrix(rowMeans(sapply(pars.smpl, function(z) z$Anm1)), 1, J)
        pars$Anmh <- matrix(rowMeans(sapply(pars.smpl, function(z) z$Anmh)), 1, length(mats.nmh))
        pars$Ah <- mean(sapply(pars.smpl, function(z) z$Ah))
        pars$Bnm1 <- matrix(rowMeans(sapply(pars.smpl, function(z) z$Bnm1)), N, J)
        pars$Bnmh <- matrix(rowMeans(sapply(pars.smpl, function(z) z$Bnmh)), N, length(mats.nmh))
        pars$Bh <- rowMeans(sapply(pars.smpl, function(z) z$Bh))

        ## get yields of all maturities to calculate realized returns
        Yall <- getAllYields(dates)

        ## calculate realized and expected excess returns for one-month holding period
        for (n in mats) {
            rx[,mats==n] <- - (n-1)*Yall[(1+1):T, n-1] + n*Yall[1:(T-1), n] - Yall[1:(T-1), 1]
            cBnm1 <- as.numeric(-(n-1)*pars$Bnm1[,mats==n])
            Erx[,mats==n] <- rep(-.5*cBnm1 %*% pars$Omega %*% cBnm1, T-1) + Sig.lambda.t[1:(T-1),] %*% cBnm1
        }
        ## posterior distribution of mean expected excess returns
        Erx.mean.smpl <- sapply(pars.smpl, function(pars) {
            Erx.mean <- numeric(J)
            for (n in mats) {
                cBnm1 <- as.numeric(-(n-1)*pars$Bnm1[,mats==n])
                Erx.mean[mats==n] <- -.5*cBnm1 %*% pars$Omega %*% cBnm1 + t(cBnm1) %*% (pars$Lam0 + pars$Lam1 %*% pars$EcP)
            }
            return(Erx.mean)
        })
        ## calculate realizes and expected excess returns for h-month holding period
        yhhat <- rep(1,T)*pars$Ah + cP%*%pars$Bh # need fitted yield with maturity h
        cAh <- -h*pars$Ah; cBh <- as.numeric(-h*pars$Bh) # bond price loadings: cA.n <- -n*A.n
        Phi.h <- matrix.power(pars$Phi, h)
        for (n in mats[mats>h]) {
            rx.h[,mats==n] <- - (n-h)*Yall[(1+h):T, n-h] + n*Yall[1:(T-h), n] - h*Yall[1:(T-h), h]
            cAnmh <- -(n-h)*pars$Anmh[mats.nmh==n-h]; cBnmh <- as.numeric(-(n-h)*pars$Bnmh[,mats.nmh==n-h])
            cAn <- -n*pars$A[mats==n]; cBn <- as.numeric(-n*pars$B[,mats==n])
            Erx.h[,mats==n] <- rep(cAnmh - cAn + cAh + EcP %*% (cBnmh - cBn + cBh), T-h) +
                t(t(cP[1:(T-h),])-EcP) %*% t(cBnmh%*%Phi.h - cBn + cBh)
        }
        ## credibility intervals around time series of expected returns
        Erx10h.smpl <- sapply(pars.smpl, function(pars) {
            n <- mats[J] ## ten-year yield
            cAn <- -n*pars$A[mats==n]; cBn <- as.numeric(-n*pars$B[,mats==n])
            cAnmh <- -(n-h)*pars$Anmh[mats.nmh==n-h]; cBnmh <- as.numeric(-(n-h)*pars$Bnmh[,mats.nmh==n-h])
            cAh <- -h*pars$Ah; cBh <- as.numeric(-h*pars$Bh)
            Phi.h <- Re(matrix.power(pars$Phi, h))
            return(rep(cAnmh - cAn + cAh + EcP %*% (cBnmh - cBn + cBh), T-h) +
                   t(t(cP[1:(T-h),])-pars$EcP) %*% t(cBnmh%*%Phi.h - cBn + cBh))
        })
        ## CP-expected returns
        xdat <- cP[1:(T-h),]
        ydat <- rx.h[, mats==120]
        cp.reg <- lm(ydat ~ xdat)
        Erx10h.cp <- ts(cp.reg$fitted.values*100, start=start.date, freq=12)
    }))
}



getPredPop <- function(Bh, Bn, Bnmh, Phi, Omega, h) {
    ## calculate model-implied predictability of excess returns
    J <- ncol(Bn)
    N <- nrow(Bn)
    R2 <- numeric(J)
    cBh <- as.numeric(-h*Bh)
    VarcP <- matrix( solve(diag(N^2) - kronecker(Phi, Phi))%*%as.numeric(Omega), N, N)
    ## -> only way this is consistent with long-sample simulation
    mats.nmh <- mats[mats>h]-h  ## consider only positive maturities
    for (n in mats[mats>h]) {
        cBnmh <- as.numeric(-(n-h)*Bnmh[,mats.nmh==n-h])
        cBn <- as.numeric(-n*Bn[,mats==n])
        tmpsum <- matrix(0, N, N)
        Phi.i <- diag(N)
        for (i in 0:(h-1)) {
            tmpsum <- tmpsum + Phi.i %*% Omega %*% t(Phi.i)
            Phi.i <- Phi %*% Phi.i
        }  ## Phi.i ends up as Phi^h
        alpha <- cBnmh%*%Phi.i - cBn + cBh
        c <- alpha %*% VarcP %*% t(alpha) + cBnmh %*% tmpsum %*% cBnmh
        R2[mats==n] <- alpha %*% VarcP %*% t(alpha) / c
    }
    return(R2)
}

getPredSmallSample <- function(An, Anmh, Bn, Bnmh, mu, Phi, Sigma, h, T, M=1000) {
    ## predictability of excess returns in small sample
    ## calculate distribion of R^2
    set.seed(616)
    J <- ncol(Bn)
    N <- nrow(Bn)
    mats.nmh <- mats[mats>h]-h  ## consider only positive maturities
    cp.boot <- matrix(NA, M, J)
    EcP <- as.numeric(solve(diag(N) - Phi) %*% mu) # unconditional mean
    VarcP <- matrix( solve(diag(N^2) - kronecker(Phi, Phi))%*%as.numeric(Sigma %*% t(Sigma)), N, N)
    ## simulate short sample, regress rx on risk factors, distribution of R^2
    for (b in 1:M) {
        cPsim <- matrix(NA, T, N)
        cPsim[1,] <-  mvrnorm(1, EcP, VarcP)
        for (t in 2:T)
            cPsim[t,] <- mu + Phi %*% cPsim[t-1,] + Sigma %*% rnorm(N)
        Ysim <- rep(1, T)%*%An + cPsim %*% Bn
        Ysim.nmh <- rep(1, T)%*%Anmh + cPsim %*% Bnmh
        xdat <- cPsim[1:(T-h),]
        for (n in mats[mats>h]) {
            ydat <- -(n-h)*Ysim.nmh[(1+h):T,mats.nmh==n-h] + n*Ysim[1:(T-h),mats==n]  - h*Ysim[1:(T-h), mats==h]  ## h-period excess returns in simulated yield data
            cp.reg <- lm(ydat ~ xdat)
            cp.boot[b, mats==n] <- summary(cp.reg)$r.squared
        }
    }
    return(cp.boot)
}

printCP <- function(models, to.latex=FALSE) {
    ## table with Cochrane-Piazzesi results
    tbl <- matrix(NA, length(mat.sel)*2, 1+length(models)*2)
    rownames(tbl) <- paste(rep(mats[mat.sel],each=2), rep(seq(1,2),length(mat.sel)), sep=".")
    colnames(tbl) <- c("Data", paste(rep(names(models),each=2), rep(c("Pop", "Smpl"),length(models)), sep="."))
    h <- models[[1]]$h; rx.h <- models[[1]]$rx.h
    cP <- models[[1]]$cP; T <- models[[1]]$T
    xdat <- cP[1:(T-h),]
    row <- 1
    for (m in mats[mat.sel]) {
        ## CP regression
        ydat <- rx.h[, mats==m]
        cp.reg <- lm(ydat ~ xdat)
        tbl[row, 1] <- summary(cp.reg)$r.squared
        row <- row + 2
    }
    ## model-implied R^2
    col <- 2
    for (model in models) {
        row <- 1
        ## population R^2
        tbl[seq(1, by=2, length=length(mat.sel)), col] <- getPredPop(
                         model$pars$Bh, model$pars$B, model$pars$Bnmh, model$pars$Phi,
                         model$pars$Omega, model$h)[mat.sel]
        ## small-sample R^2
        cp.boot <- getPredSmallSample(model$pars$A, model$pars$Anmh, model$pars$B, model$pars$Bnmh, model$pars$mu, model$pars$Phi, model$pars$Sigma, model$h, model$T)
        for (j in mat.sel) {
            tbl[row, col+1] <- mean(cp.boot[,j])
            tbl[row+1, col+1] <- sd(cp.boot[,j])
            row <- row+2
        }
        col <- col+2
    }
    print(round(tbl, digi=3))
    if (to.latex) {
        filename <- "tables/cp.tex"
        cat("*** writing table", filename, "\n")
        sink(filename)
        cat("\\begin{tabular}{l|c", rep("|cc", length(models)), "} \\hline \\hline \n")
        cat("   & ",
            sapply(names(models), function(s) paste("& \\multicolumn{2}{|c}{", s, "}", sep="")),
            "\\\\ \n")
        cat("Maturity & Data ",
            rep("& Pop. & Smpl.", length(models)),
            "\\\\ \\hline \n")
        for (row in 1:nrow(tbl)) {
            if (row %% 2 == 1) {
                i <- ceiling(row/2) ## which of the selected maturities
                cat(mats[mat.sel[i]]/12, "years")
                cat(sprintf("& %4.2f ", tbl[row,]), "\\\\ \n")
            } else {
                ## standard errors
                for (col in 1:ncol(tbl))
                    if (!is.na(tbl[row, col])) {
                        cat(sprintf("& (%4.2f)", tbl[row,col]))
                    } else {
                        cat("&")
                    }
                cat("\\\\ \n \\hline \n")
            }
        }
        cat("\\hline \n \\end{tabular}\n")
        sink()
    }

}

pricedRisks <- function(filename) {
    ## estimate probabilities that level/slope/curve risk is priced
    cat("showing results for", filename, "\n")
    load(filename)			## load estimation results
    mcmc.ind <- getMCMCind(M)
    gamma.i <- gamma.i[mcmc.ind,]	## drop burn-in iterations
    tbl <- matrix(NA, 3, 2)
    colnames(tbl) <- c("Prob(priced)", "Prob(time-var)")
    rownames(tbl) <- c("Level", "Slope", "Curvature")
    for (i in 1:3) {
        tbl[i,1] <- sum(rowSums(gamma.i[, i-1+c(1,4,7,10)])>0)/length(mcmc.ind)
        tbl[i,2] <- sum(rowSums(gamma.i[, i-1+c(4,7,10)])>0)/length(mcmc.ind)
    }
    print(tbl)
    cat("Prob(Level risk  time-varying) =", mean(rowSums(gamma.i[, c(4,7,10)])>0), "\n")
    cat("Prob(Any risk    time-varying) =", mean(rowSums(gamma.i[, 4:12])>0), "\n")
}


