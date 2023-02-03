#########################
##Tuing hyperparameters##
#########################
#This file derives hyper-parameters for priors. The marginal likelihood for the transition equation is maximized at the selected hyperparameters. The hyperparameters are 

#    * p: the lag of the transition equation
#    * ν0(d.f.), Ω0(scale): hyper-parameters of the Inverse-Wishart prior distribution for the error covariance matrix
#    * q: the degree of shrikages of the intercept and the slope coefficient of the transition equation
#        * q[1]: shrikages for the lagged dependent variable
#        * q[2]: shrikages for other variables
#        * q[3]: shrikages for the lag itself
#        * q[4]: shrikages for the intercept

function marginal(PCs, macros; p, ν0, Ω0, q, κQ, ψ, ψ0, ρ)
    T = size(PCs)[1]
    dP = length(Ω0)

    yϕ, Xϕ = yϕ_Xϕ(PCs, macros; p)
    prior_ϕ0_ = prior_ϕ0(ρ; κQ, ψ0, ψ, q, ν0, Ω0)
    prior_C_ = prior_C(; Ω0)
    prior_ϕ = [prior_ϕ0_ prior_C_]
    m = mean.(prior_ϕ)
    V = var.(prior_ϕ)

    log_marginal = -log(2π)
    log_marginal *= (T * dP) / 2
    for i in 1:dP
        νᵢ = ν(i; ν0, dP)
        Sᵢ = S(i; Ω0)
        Vᵢ = V[i, 1:(end-dP+i-1)]
        Kϕᵢ = Kϕ(i, V; Xϕ, dP)
        Sᵢ_hat = S_hat(i, m, V; yϕ, Xϕ, Ω0, dP)

        log_marginalᵢ = sum(log.(Vᵢ))
        log_marginalᵢ += log(det(Kϕᵢ))
        log_marginalᵢ /= -2
        log_marginalᵢ += log(gamma(νᵢ + 0.5T))
        log_marginalᵢ += νᵢ * log(Sᵢ)
        log_marginalᵢ -= log(gamma(νᵢ))
        log_marginalᵢ -= (νᵢ + 0.5T) * log(Sᵢ_hat)

        log_marginal += log_marginalᵢ
    end
end

function ν(i; ν0, dP)
    return (ν0 + i - dP) / 2
end

function S(i; Ω0)
    return Ω0[i] / 2
end

"""
    *Input: V = var.([prior_ϕ0 prior_C])
            Xϕ = output of yϕ_Xϕ(PCs, macros, p) 
"""
function Kϕ(i, V; Xϕ, dP)
    Xϕᵢ = Xϕ[:, 1:(end-dP+i-1)]
    Vᵢ = V[i, 1:(end-dP+i-1)]
    return diagm(1 ./ Vᵢ) + Xϕᵢ'Xϕᵢ
end

"""
    *Input: m = mean.([prior_ϕ0 prior_C])
            V = var.([prior_ϕ0 prior_C])
            yϕ, Xϕ = output of yϕ_Xϕ(PCs, macros, p) 
"""
function ϕ_hat(i, m, V; yϕ, Xϕ, dP)
    Kϕᵢ = Kϕ(i, V; Xϕ, dP)
    Xϕᵢ = Xϕ[:, 1:(end-dP+i-1)]
    yϕᵢ = yϕ[:, i]
    mᵢ = m[i, 1:(end-dP+i-1)]
    Vᵢ = V[i, 1:(end-dP+i-1)]

    return Kϕᵢ \ (diagm(1 ./ Vᵢ) * mᵢ' + Xϕᵢ'yϕᵢ)
end

"""
    *Input: m = mean.([prior_ϕ0 prior_C])
            V = var.([prior_ϕ0 prior_C])
            yϕ, Xϕ = output of yϕ_Xϕ(PCs, macros, p) 
"""
function S_hat(i, m, V; yϕ, Xϕ, Ω0, dP)
    Sᵢ = S(i; Ω0)
    yϕᵢ = yϕ[:, i]
    mᵢ = m[i, 1:(end-dP+i-1)]
    Vᵢ = V[i, 1:(end-dP+i-1)]
    Kϕᵢ = Kϕ(i, V; Xϕ, dP)
    ϕᵢ_hat = ϕ_hat(i, m, V; yϕ, Xϕ, dP)

    Sᵢ_hat = yϕᵢ'yϕᵢ + mᵢ' * diagm(1 ./ Vᵢ) * mᵢ - ϕᵢ_hat' * Kϕᵢ * ϕᵢ_hat
    Sᵢ_hat /= 2

    return Sᵢ + Sᵢ_hat
end
