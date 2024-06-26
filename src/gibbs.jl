
"""
    post_kQ_infty(mean_kQ_infty, std_kQ_infty, yields, tau_n; kappaQ, phi, varFF, SigmaO, data_scale)
# Output
- Full conditional posterior distribution
"""
function post_kQ_infty(mean_kQ_infty, std_kQ_infty, yields, tau_n; kappaQ, phi, varFF, SigmaO, data_scale)

    dP = length(varFF)
    dZ = size(yields, 2) - length(tau_n)
    dQ = dimQ() + dZ
    p = Int(((size(phi, 2) - 1) / dP) - 1)
    yields = yields[p+1:end, :]

    N = length(tau_n) # of maturities
    T = size(yields, 1) # length of dependent variables
    PCs, OCs, Wₚ, Wₒ, mean_PCs = PCA(yields, 0; spanned=yields[:, end-dZ+1:end])

    bτ_ = bτ(tau_n[end]; kappaQ, dQ)
    Bₓ_ = Bₓ(bτ_, tau_n)
    T1X_ = T1X(Bₓ_, Wₚ)
    Bₚ_ = Bₚ(Bₓ_, T1X_, Wₒ)
    ΩPP = phi_varFF_2_ΩPP(; phi, varFF, dQ)

    a0 = zeros(tau_n[end])
    a1 = zeros(tau_n[end])
    for τ in 2:tau_n[end]
        a0[τ] = a0[τ-1] - jensens_inequality(τ, bτ_, T1X_; ΩPP, data_scale)
        if length(kappaQ) > 1
            a1[τ] = a1[τ-1] + (1 - (kappaQ[1]^(τ - 1))) / (1 - kappaQ[1])
        else
            a1[τ] = a1[τ-1] + (τ - 1)
        end
    end
    A0_kQ_infty = a0[tau_n] ./ tau_n
    A1_kQ_infty = a1[tau_n] ./ tau_n

    # Dependent variable
    y = vec(OCs')
    y -= kron(ones(T), Wₒ * (I(N) - Bₓ_ / T1X_ * Wₚ) * A0_kQ_infty + Wₒ * Bₓ_ / T1X_ * mean_PCs)
    y -= vec(Bₚ_ * PCs')
    y ./= kron(ones(T), sqrt.(SigmaO))

    # regressor
    X = Wₒ * (I(N) - Bₓ_ / T1X_ * Wₚ) * A1_kQ_infty
    X ./= sqrt.(SigmaO)
    X = kron(ones(T), X)

    kQ_infty_var = inv(X'X + (1 / (std_kQ_infty^2)))
    return Normal(kQ_infty_var * ((mean_kQ_infty / (std_kQ_infty^2)) + X'y), sqrt(kQ_infty_var))

end

"""
    post_kappaQ(yields, prior_kappaQ_, tau_n; kQ_infty, phi, varFF, SigmaO, data_scale)
# Input
- `prior_kappaQ_` is a output of function `prior_kappaQ`.
# Output 
- Full conditional posterior distribution
"""
function post_kappaQ(yields, prior_kappaQ_, tau_n; kQ_infty, phi, varFF, SigmaO, data_scale)

    kappaQ_candidate = support(prior_kappaQ_)

    kern = Vector{Float64}(undef, length(kappaQ_candidate)) # Posterior kernel

    for i in eachindex(kappaQ_candidate)
        # likelihood of the measurement eq
        kern[i] = loglik_mea(yields, tau_n; kappaQ=kappaQ_candidate[i], kQ_infty, phi, varFF, SigmaO, data_scale)
    end

    kern .-= maximum(kern)
    Pr = exp.(kern)
    Pr ./= sum(Pr)

    return DiscreteNonParametric(kappaQ_candidate, Pr)
end

"""
    post_kappaQ2(yields, prior_kappaQ_, tau_n; kappaQ, kQ_infty, phi, varFF, SigmaO, data_scale, x_mode, inv_x_hess)
It conducts the Metropolis-Hastings algorithm for the reparameterized `kappaQ` under the unrestricted JSZ form. `x_mode` and `inv_x_hess` constitute the mean and variance of the Normal proposal distribution.
- Reparameterization:
    kappaQ[1] = x[1]
    kappaQ[2] = x[1] + x[2]
    kappaQ[3] = x[1] + x[2] + x[3]
- Jacobian:
    [1 0 0
    1 1 0
    1 1 1]
- The determinant = 1
"""
function post_kappaQ2(yields, prior_kappaQ_, tau_n; kappaQ, kQ_infty, phi, varFF, SigmaO, data_scale, x_mode, inv_x_hess)

    function logpost(x)
        kappaQ = [x[1], x[1] + x[2], x[1] + x[2] + x[3]]
        loglik = loglik_mea(yields, tau_n; kappaQ, kQ_infty, phi, varFF, SigmaO, data_scale)
        logprior = 0.0
        for i in eachindex(prior_kappaQ_)
            logprior += logpdf(prior_kappaQ_[i], kappaQ[i])
        end
        return loglik + logprior
    end

    # AR step
    proposal_dist = MvNormal(x_mode, inv_x_hess)

    is_cond = true
    kappaQ_prop = similar(kappaQ)
    x_prop = similar(kappaQ)
    while is_cond
        x_prop = rand(proposal_dist)
        kappaQ_prop = [x_prop[1], x_prop[1] + x_prop[2], x_prop[1] + x_prop[2] + x_prop[3]]
        if sort(kappaQ_prop, rev=true) == kappaQ_prop && kappaQ_prop[1] < 1.0
            is_cond = false
        end
    end
    x = [kappaQ[1], kappaQ[2] - kappaQ[1], kappaQ[3] - kappaQ[2]]
    log_MHPr = min(0.0, logpost(x_prop) + logpdf(proposal_dist, kappaQ) - logpost(x) - logpdf(proposal_dist, kappaQ_prop))
    if log(rand()) < log_MHPr
        return kappaQ_prop, true
    else
        return kappaQ, false
    end
end

"""
    post_phi_varFF(yields, macros, mean_phi_const, rho, prior_kappaQ_, tau_n; phi, ψ, ψ0, varFF, q, nu0, Omega0, kappaQ, kQ_infty, SigmaO, fix_const_PC1, data_scale)
Full-conditional posterior sampler for `phi` and `varFF` 
# Input
- `prior_kappaQ_` is a output of function `prior_kappaQ`.
- When `fix_const_PC1==true`, the first element in a constant term in our orthogonalized VAR is fixed to its prior mean during the posterior sampling.
# Output(3) 
`phi`, `varFF`, `isaccept=Vector{Bool}(undef, dQ)`
- It gives a posterior sample.
"""
function post_phi_varFF(yields, macros, mean_phi_const, rho, prior_kappaQ_, tau_n; phi, ψ, ψ0, varFF, q, nu0, Omega0, kappaQ, kQ_infty, SigmaO, fix_const_PC1, data_scale)

    dZ = size(yields, 2) - length(tau_n)
    dQ = dimQ() + dZ
    dP = size(ψ, 1)
    p = Int(size(ψ)[2] / dP)
    PCs, ~, Wₚ = PCA(yields, p; spanned=yields[:, end-dZ+1:end])

    yphi, Xphi = yphi_Xphi(PCs, macros, p)
    prior_phi0_ = prior_phi0(mean_phi_const, rho, prior_kappaQ_, tau_n, Wₚ; ψ0, ψ, q, nu0, Omega0, fix_const_PC1)
    prior_phi_ = [prior_phi0_ prior_C(; Omega0)]
    prior_varFF_ = prior_varFF(; nu0, Omega0)

    isaccept = fill(false, dQ)
    for i in 1:dP
        if i <= dQ
            prop_phi = copy(phi) # proposal for C
            prop_varFF = copy(varFF) # proposal for varFF

            mᵢ = mean.(prior_phi_[i, 1:(1+p*dP+i-1)])
            Vᵢ = var.(prior_phi_[i, 1:(1+p*dP+i-1)])
            prop_phi[i, 1:(1+p*dP+i-1)], prop_varFF[i] = NIG_NIG(yphi[:, i], Xphi[:, 1:(end-dP+i-1)], mᵢ, diagm(Vᵢ), shape(prior_varFF_[i]), scale(prior_varFF_[i]))

            prob = loglik_mea(yields, tau_n; kappaQ, kQ_infty, phi=prop_phi, varFF=prop_varFF, SigmaO, data_scale)
            prob -= loglik_mea(yields, tau_n; kappaQ, kQ_infty, phi, varFF, SigmaO, data_scale)

            if rand() < min(1.0, exp(prob))
                phi = copy(prop_phi)
                varFF = copy(prop_varFF)
                isaccept[i] = true
            end
        else
            mᵢ = mean.(prior_phi_[i, 1:(1+p*dP+i-1)])
            Vᵢ = var.(prior_phi_[i, 1:(1+p*dP+i-1)])
            phi[i, 1:(1+p*dP+i-1)], varFF[i] = NIG_NIG(yphi[:, i], Xphi[:, 1:(end-dP+i-1)], mᵢ, diagm(Vᵢ), shape(prior_varFF_[i]), scale(prior_varFF_[i]))
        end
    end

    return phi, varFF, isaccept

end

"""
    NIG_NIG(y, X, β₀, B₀, α₀, δ₀)
Normal-InverseGamma-Normal-InverseGamma update
- prior: `β|σ² ~ MvNormal(β₀,σ²B₀)`, `σ² ~ InverseGamma(α₀,δ₀)`
- likelihood: `y|β,σ² = Xβ + MvNormal(zeros(T,1),σ²I(T))`
# Output(2)
`β`, `σ²`
- posterior sample
"""
function NIG_NIG(y, X, β₀, B₀, α₀, δ₀)

    T = length(y)

    inv_B₀ = inv(B₀)
    inv_B₁ = inv_B₀ + X'X
    B₁ = Symmetric(inv(inv_B₁))
    β₁ = B₁ * (inv_B₀ * β₀ + X'y)
    δ₁ = δ₀ + 0.5 * (y'y + β₀' * inv_B₀ * β₀ - β₁' * inv_B₁ * β₁)

    σ² = rand(InverseGamma(α₀ + 0.5T, δ₁))
    β = rand(MvNormal(β₁, σ² * B₁))

    return β, σ²
end

"""
    post_SigmaO(yields, tau_n; kappaQ, kQ_infty, ΩPP, gamma, p, data_scale)
Posterior sampler for the measurement errors
# Output
- `Vector{Dist}(IG, N-dQ)`
"""
function post_SigmaO(yields, tau_n; kappaQ, kQ_infty, ΩPP, gamma, p, data_scale)
    yields = yields[p+1:end, :]

    dZ = size(yields, 2) - length(tau_n)
    dQ = dimQ() + dZ
    N = length(tau_n)
    T = size(yields, 1)
    PCs, OCs, Wₚ, Wₒ, mean_PCs = PCA(yields, 0; spanned=yields[:, end-dZ+1:end])

    bτ_ = bτ(tau_n[end]; kappaQ, dQ)
    Bₓ_ = Bₓ(bτ_, tau_n)
    T1X_ = T1X(Bₓ_, Wₚ)
    Bₚ_ = Bₚ(Bₓ_, T1X_, Wₒ)

    aτ_ = aτ(tau_n[end], bτ_, tau_n, Wₚ; kQ_infty, ΩPP, data_scale)
    Aₓ_ = Aₓ(aτ_, tau_n)
    T0P_ = T0P(T1X_, Aₓ_, Wₚ, mean_PCs)
    Aₚ_ = Aₚ(Aₓ_, Bₓ_, T0P_, Wₒ)

    post_SigmaO_ = Vector{Any}(undef, N - dQ)
    for i in 1:N-dQ
        residuals = OCs[:, i] - (Aₚ_[i] .+ (Bₚ_[i, :]' * PCs')')
        post_SigmaO_[i] = InverseGamma(2 + 0.5T, gamma[i] + 0.5residuals'residuals)
    end

    return post_SigmaO_
end

"""
    post_gamma(; gamma_bar, SigmaO)
Posterior sampler for the population measurement error
# Output
- `Vector{Dist}(Gamma,length(SigmaO))`
"""
function post_gamma(; gamma_bar, SigmaO)

    N = length(SigmaO) # of measurement errors

    post_gamma_ = Vector{Any}(undef, N)
    for i in 1:N
        post_gamma_[i] = Gamma(3, 1 / (gamma_bar + (1 / SigmaO[i])))
    end

    return post_gamma_
end
