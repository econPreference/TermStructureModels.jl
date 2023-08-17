"""
    log_marginal(PCs, macros, ρ, tuned::Hyperparameter, τₙ, Wₚ; ψ=[], ψ0=[], medium_τ, medium_τ_pr)
This file calculates a value of our marginal likelihood. Only the transition equation is used to calculate it. 
# Input
- tuned is a point where the marginal likelihood is evaluated. 	
- `ψ0` and `ψ` are multiplied with prior variances of coefficients of the intercept and lagged regressors in the orthogonalized transition equation. They are used for imposing zero prior variances. A empty default value means that you do not use this function. `[ψ0 ψ][i,j]` is corresponds to `ϕ[i,j]`. 
# Output
- the log marginal likelihood of the VAR system.
"""
function log_marginal(PCs, macros, ρ, tuned::Hyperparameter, τₙ, Wₚ; ψ=[], ψ0=[], medium_τ, medium_τ_pr, fix_const_PC1)

    (; p, ν0, Ω0, q, μϕ_const) = tuned

    prior_κQ_ = prior_κQ(medium_τ, medium_τ_pr)
    dP = length(Ω0)

    if isempty(ψ)
        ψ = ones(dP, dP * p)
    end
    if isempty(ψ0)
        ψ0 = ones(dP)
    end

    yϕ, Xϕ = yϕ_Xϕ(PCs, macros, p)
    T = size(yϕ, 1)
    prior_ϕ0_ = prior_ϕ0(μϕ_const, ρ, prior_κQ_, τₙ, Wₚ; ψ0, ψ, q, ν0, Ω0, fix_const_PC1)
    prior_C_ = prior_C(; Ω0)
    prior_ϕ = hcat(prior_ϕ0_, prior_C_)
    m = mean.(prior_ϕ)
    V = var.(prior_ϕ)

    log_marginal_ = -log(2π)
    log_marginal_ *= (T * dP) / 2
    for i in 1:dP
        νᵢ = ν(i, dP; ν0)
        Sᵢ = S(i; Ω0)
        Vᵢ = V[i, 1:(end-dP+i-1)]
        Kϕᵢ = Kϕ(i, V, Xϕ, dP)
        Sᵢ_hat = S_hat(i, m, V, yϕ, Xϕ, dP; Ω0)
        logdet_Kϕᵢ = logdet(Kϕᵢ)
        if Sᵢ_hat < 0 || isinf(logdet_Kϕᵢ)
            return -Inf
        end

        log_marginalᵢ = sum(log.(Vᵢ))
        log_marginalᵢ += logdet_Kϕᵢ
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
    ν(i, dP; ν0)
"""
function ν(i, dP; ν0)
    return (ν0 + i - dP) / 2
end

"""
    S(i; Ω0)
"""
function S(i; Ω0)
    return Ω0[i] / 2
end

"""
    Kϕ(i, V, Xϕ, dP)
"""
function Kϕ(i, V, Xϕ, dP)
    Xϕᵢ = Xϕ[:, 1:(end-dP+i-1)]
    Vᵢ = V[i, 1:(end-dP+i-1)]
    return diagm(1 ./ Vᵢ) + Xϕᵢ'Xϕᵢ
end

"""
    ϕ_hat(i, m, V, yϕ, Xϕ, dP)
"""
function ϕ_hat(i, m, V, yϕ, Xϕ, dP)
    Kϕᵢ = Kϕ(i, V, Xϕ, dP)
    Xϕᵢ = Xϕ[:, 1:(end-dP+i-1)]
    yϕᵢ = yϕ[:, i]
    mᵢ = m[i, 1:(end-dP+i-1)]
    Vᵢ = V[i, 1:(end-dP+i-1)]

    return Kϕᵢ \ (diagm(1 ./ Vᵢ) * mᵢ + Xϕᵢ'yϕᵢ)
end

"""
    S_hat(i, m, V, yϕ, Xϕ, dP; Ω0)
"""
function S_hat(i, m, V, yϕ, Xϕ, dP; Ω0)

    yϕᵢ = yϕ[:, i]
    mᵢ = m[i, 1:(end-dP+i-1)]
    Vᵢ = V[i, 1:(end-dP+i-1)]
    Kϕᵢ = Kϕ(i, V, Xϕ, dP)
    ϕᵢ_hat = ϕ_hat(i, m, V, yϕ, Xϕ, dP)

    Sᵢ_hat = S(i; Ω0)
    Sᵢ_hat += (yϕᵢ'yϕᵢ + mᵢ' * diagm(1 ./ Vᵢ) * mᵢ - ϕᵢ_hat' * Kϕᵢ * ϕᵢ_hat) / 2

    return Sᵢ_hat
end
