"""
log_marginal(PCs, macros, ρ, Hyperparameter_::Hyperparameter, τₙ, Wₚ; ψ=[], ψ0=[], medium_τ)
* This file derives hyper-parameters for priors. The marginal likelihood for the transition equation is maximized at the selected hyperparameters. 
* Input: Data should contain initial observations. 
    * ρ only indicates macro variables' persistencies.
    * medium_τ is a vector of representative medium maturities that are used for constructing prior for κQ.
*Output: the log marginal likelihood of the VAR system.
"""
function log_marginal(PCs, macros, ρ, Hyperparameter_::Hyperparameter, τₙ, Wₚ; ψ=[], ψ0=[], medium_τ)

    (; p, ν0, Ω0, q, μϕ_const, fix_const_PC1) = Hyperparameter_
    # if max(q[1, 1] / (p^q[3, 1]), q[1, 2] / (p^q[3, 2])) < (0.001)^2
    #     return -Inf
    # end

    prior_κQ_ = prior_κQ(medium_τ)
    dP = length(Ω0)
    if isempty(ψ) || isempty(ψ0)
        ψ = ones(dP, dP * p)
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
