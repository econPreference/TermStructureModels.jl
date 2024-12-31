"""
    log_marginal(PCs, macros, rho, tuned::Hyperparameter, tau_n, Wₚ; psi=[], psi_const=[], medium_tau, kappaQ_prior_pr, fix_const_PC1)
This file calculates a value of our marginal likelihood. Only the transition equation is used to calculate it. 
# Input
- tuned is a point where the marginal likelihood is evaluated. 	
- `psi_const` and `psi` are multiplied with prior variances of coefficients of the intercept and lagged regressors in the orthogonalized transition equation. They are used for imposing zero prior variances. A empty default value means that you do not use this function. `[psi_const psi][i,j]` is corresponds to `phi[i,j]`. 
# Output
- the log marginal likelihood of the VAR system.
"""
function log_marginal(PCs, macros, rho, tuned::Hyperparameter, tau_n, Wₚ; psi=[], psi_const=[], medium_tau, kappaQ_prior_pr, fix_const_PC1)

    p, nu0, Omega0, q, mean_phi_const = tuned.p, tuned.nu0, tuned.Omega0, tuned.q, tuned.mean_phi_const

    prior_kappaQ_ = prior_kappaQ(medium_tau, kappaQ_prior_pr)
    dP = length(Omega0)

    if isempty(psi)
        psi = ones(dP, dP * p)
    end
    if isempty(psi_const)
        psi_const = ones(dP)
    end

    yphi, Xphi = yphi_Xphi(PCs, macros, p)
    T = size(yphi, 1)
    prior_phi0_ = prior_phi0(mean_phi_const, rho, prior_kappaQ_, tau_n, Wₚ; psi_const, psi, q, nu0, Omega0, fix_const_PC1)
    prior_C_ = prior_C(; Omega0)
    prior_phi = hcat(prior_phi0_, prior_C_)
    m = mean.(prior_phi)
    V = var.(prior_phi)

    log_marginal_ = -log(2π)
    log_marginal_ *= (T * dP) / 2
    for i in 1:dP
        νᵢ = ν(i, dP; nu0)
        Sᵢ = S(i; Omega0)
        Vᵢ = V[i, 1:(end-dP+i-1)]
        Kphiᵢ = Kphi(i, V, Xphi, dP)
        Sᵢ_hat = S_hat(i, m, V, yphi, Xphi, dP; Omega0)
        logdet_Kphiᵢ = cholesky(Kphiᵢ).L |> Matrix |> diag |> x -> log.(x) |> x -> 2 * sum(x)
        if Sᵢ_hat < 0 || isinf(logdet_Kphiᵢ)
            return -Inf
        end

        log_marginalᵢ = sum(log.(Vᵢ))
        log_marginalᵢ += logdet_Kphiᵢ
        log_marginalᵢ /= -2
        log_marginalᵢ += loggamma(νᵢ + 0.5T)
        log_marginalᵢ += νᵢ * log(Sᵢ)
        log_marginalᵢ -= loggamma(νᵢ)
        log_marginalᵢ -= (νᵢ + 0.5T) * log(Sᵢ_hat)

        log_marginal_ += log_marginalᵢ
    end

    return log_marginal_
end

"""
    ν(i, dP; nu0)
"""
function ν(i, dP; nu0)
    return (nu0 + i - dP) / 2
end

"""
    S(i; Omega0)
"""
function S(i; Omega0)
    return Omega0[i] / 2
end

"""
    Kphi(i, V, Xphi, dP)
"""
function Kphi(i, V, Xphi, dP)
    Xphiᵢ = Xphi[:, 1:(end-dP+i-1)]
    Vᵢ = V[i, 1:(end-dP+i-1)]
    return diagm(1 ./ Vᵢ) + Xphiᵢ'Xphiᵢ
end

"""
    phi_hat(i, m, V, yphi, Xphi, dP)
"""
function phi_hat(i, m, V, yphi, Xphi, dP)
    Kphiᵢ = Kphi(i, V, Xphi, dP)
    Xphiᵢ = Xphi[:, 1:(end-dP+i-1)]
    yphiᵢ = yphi[:, i]
    mᵢ = m[i, 1:(end-dP+i-1)]
    Vᵢ = V[i, 1:(end-dP+i-1)]

    return Kphiᵢ \ (diagm(1 ./ Vᵢ) * mᵢ + Xphiᵢ'yphiᵢ)
end

"""
    S_hat(i, m, V, yphi, Xphi, dP; Omega0)
"""
function S_hat(i, m, V, yphi, Xphi, dP; Omega0)

    yphiᵢ = yphi[:, i]
    mᵢ = m[i, 1:(end-dP+i-1)]
    Vᵢ = V[i, 1:(end-dP+i-1)]
    Kphiᵢ = Kphi(i, V, Xphi, dP)
    phiᵢ_hat = phi_hat(i, m, V, yphi, Xphi, dP)

    Sᵢ_hat = S(i; Omega0)
    Sᵢ_hat += (yphiᵢ'yphiᵢ + mᵢ' * diagm(1 ./ Vᵢ) * mᵢ - phiᵢ_hat' * Kphiᵢ * phiᵢ_hat) / 2

    return Sᵢ_hat
end
