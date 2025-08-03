
"""
    post_kQ_infty(mean_kQ_infty, std_kQ_infty, yields, tau_n; kappaQ, phi, varFF, SigmaO, data_scale, pca_loadings)
# Output
- Full conditional posterior distribution
"""
function post_kQ_infty(mean_kQ_infty, std_kQ_infty, yields, tau_n; kappaQ, phi, varFF, SigmaO, data_scale, pca_loadings)

    dP = length(varFF)
    dQ = dimQ() + size(yields, 2) - length(tau_n)
    p = Int(((size(phi, 2) - 1) / dP) - 1)
    yields = yields[p+1:end, :]

    N = length(tau_n) # of maturities
    T = size(yields, 1) # length of dependent variables
    PCs, OCs, Wₚ, Wₒ, mean_PCs = PCA(yields, 0; pca_loadings)

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
    post_kappaQ(yields, prior_kappaQ_, tau_n; kQ_infty, phi, varFF, SigmaO, data_scale, pca_loadings)
# Input
- `prior_kappaQ_` is a output of function `prior_kappaQ`.
# Output 
- Full conditional posterior distribution
"""
function post_kappaQ(yields, prior_kappaQ_, tau_n; kQ_infty, phi, varFF, SigmaO, data_scale, pca_loadings)

    kappaQ_candidate = support(prior_kappaQ_)

    kern = Vector{Float64}(undef, length(kappaQ_candidate)) # Posterior kernel

    for i in eachindex(kappaQ_candidate)
        # likelihood of the measurement eq
        kern[i] = loglik_mea(yields, tau_n; kappaQ=kappaQ_candidate[i], kQ_infty, phi, varFF, SigmaO, data_scale, pca_loadings)
    end

    kern .-= maximum(kern)
    Pr = exp.(kern)
    Pr ./= sum(Pr)

    return DiscreteNonParametric(kappaQ_candidate, Pr)
end

"""
    post_kappaQ2(yields, prior_kappaQ_, tau_n; kappaQ, kQ_infty, phi, varFF, SigmaO, data_scale, x_mode, inv_x_hess, pca_loadings)
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
function post_kappaQ2(yields, prior_kappaQ_, tau_n; kappaQ, kQ_infty, phi, varFF, SigmaO, data_scale, x_mode, inv_x_hess, pca_loadings)

    function logpost(x)
        kappaQ = [x[1], x[1] + x[2], x[1] + x[2] + x[3]]
        loglik = loglik_mea(yields, tau_n; kappaQ, kQ_infty, phi, varFF, SigmaO, data_scale, pca_loadings)
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
    post_phi_varFF(yields, macros, mean_phi_const, rho, prior_kappaQ_, tau_n; phi, psi, psi_const, varFF, q, nu0, Omega0, kappaQ, kQ_infty, SigmaO, fix_const_PC1, data_scale, pca_loadings)
Full-conditional posterior sampler for `phi` and `varFF` 
# Input
- `prior_kappaQ_` is a output of function `prior_kappaQ`.
- When `fix_const_PC1==true`, the first element in a constant term in our orthogonalized VAR is fixed to its prior mean during the posterior sampling.
# Output(3) 
`phi`, `varFF`, `isaccept=Vector{Bool}(undef, dQ)`
- It gives a posterior sample.
"""
function post_phi_varFF(yields, macros, mean_phi_const, rho, prior_kappaQ_, tau_n; phi, psi, psi_const, varFF, q, nu0, Omega0, kappaQ, kQ_infty, SigmaO, fix_const_PC1, data_scale, pca_loadings)

    dQ = dimQ() + size(yields, 2) - length(tau_n)
    dP = size(psi, 1)
    p = Int(size(psi)[2] / dP)
    PCs, ~, Wₚ = PCA(yields, p; pca_loadings)

    yphi, Xphi = yphi_Xphi(PCs, macros, p)
    prior_phi0_ = prior_phi0(mean_phi_const, rho, prior_kappaQ_, tau_n, Wₚ; psi_const, psi, q, nu0, Omega0, fix_const_PC1)
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

            prob = loglik_mea(yields, tau_n; kappaQ, kQ_infty, phi=prop_phi, varFF=prop_varFF, SigmaO, data_scale, pca_loadings)
            prob -= loglik_mea(yields, tau_n; kappaQ, kQ_infty, phi, varFF, SigmaO, data_scale, pca_loadings)

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
    post_kappaQ_phi_varFF(yields, macros, mean_phi_const, rho, prior_diff_kappaQ, tau_n; phi, psi, psi_const, varFF, q, nu0, Omega0, kappaQ, kQ_infty, SigmaO, fix_const_PC1, data_scale, pca_loadings, sampler, chain, is_warmup)
Full-conditional posterior sampler for `kappaQ`, `phi` and `varFF` 
# Input
- `prior_diff_kappaQ` is a vector of the truncated normals(`Distributions.truncated(Distributions.Normal(), lower, upper)`). It has a prior for `[kappaQ[1]; diff(kappaQ)]`.
- When `fix_const_PC1==true`, the first element in a constant term in our orthogonalized VAR is fixed to its prior mean during the posterior sampling.
- `sampler` and `chain` are the objects in `Turing.jl`.
- If the current step is in the warming up phrase, set `is_warmup=true`.
# Output(3) 
`phi`, `varFF`, `isaccept=Vector{Bool}(undef, dQ)`
- It gives a posterior sample.
"""
function post_kappaQ_phi_varFF(yields, macros, mean_phi_const, rho, prior_diff_kappaQ, tau_n; phi, psi, psi_const, varFF, q, nu0, Omega0, kappaQ, kQ_infty, SigmaO, fix_const_PC1, data_scale, pca_loadings, sampler, chain, is_warmup)

    dQ = dimQ() + size(yields, 2) - length(tau_n)
    dP = size(psi, 1)
    p = Int(size(psi)[2] / dP)
    PCs, ~, Wₚ = PCA(yields, p; pca_loadings)
    dims_phi = [1 + p * dP + i - 1 for i in 1:dQ] |> cumsum

    initial_params = [kappaQ[1]; diff(kappaQ)]
    for i in 1:dQ
        initial_params = [initial_params; phi[i, 1:(1+p*dP+i-1)]]
    end
    initial_params = [initial_params; varFF[1:dQ]]

    yphi, Xphi = yphi_Xphi(PCs, macros, p)
    prior_kappaQ_ = mean.([prior_diff_kappaQ[i].untruncated for i in eachindex(prior_diff_kappaQ)]) |> cumsum |> x -> [Dirac(x[i]) for i in eachindex(x)]
    prior_phi0_ = prior_phi0(mean_phi_const, rho, prior_kappaQ_, tau_n, Wₚ; psi_const, psi, q, nu0, Omega0, fix_const_PC1)
    prior_phi_ = [prior_phi0_ prior_C(; Omega0)]
    prior_varFF_ = prior_varFF(; nu0, Omega0)

    NUTS_model_ = NUTS_model(yields, PCs, tau_n, macros, dQ, dP, p, dims_phi, prior_diff_kappaQ, prior_phi_, prior_varFF_; kQ_infty, phi, varFF, SigmaO, data_scale, pca_loadings)

    local current_chain
    if chain == []
        current_chain = Turing.sample(NUTS_model_, sampler, 1; initial_params, save_state=true, progress=false, verbose=false)
    else
        current_chain = Turing.AbstractMCMC.mcmcsample(
            Random.default_rng(),
            NUTS_model_,
            Turing.DynamicPPL.Sampler(sampler),
            1;
            chain_type=Turing.DynamicPPL.default_chain_type(Turing.DynamicPPL.Sampler(sampler)),
            initial_state=Turing.DynamicPPL.loadstate(chain),
            progress=Turing.PROGRESS[],
            nadapts=is_warmup ? Turing.DynamicPPL.loadstate(chain).i + 1 : 0,
            discard_adapt=false,
            discard_initial=0,
            save_state=true
        )
    end

    diff_kappaQ_chain = group(current_chain, :diff_kappaQ)
    phiQ_chain = group(current_chain, :phiQ)
    varFFQ_chain = group(current_chain, :varFFQ)

    kappaQ = diff_kappaQ_chain.value |> x -> x[end, :, 1] |> cumsum
    phiQ = phiQ_chain.value |> x -> x[end, :, 1]
    varFFQ = varFFQ_chain.value |> x -> x[end, :, 1]

    for i in 1:dP
        if i <= dQ
            if i == 1
                phi[i, 1:(1+p*dP+i-1)], varFF[i] = copy(phiQ[1:dims_phi[i]]), copy(varFFQ[i])
            else
                phi[i, 1:(1+p*dP+i-1)], varFF[i] = copy(phiQ[dims_phi[i-1]+1:dims_phi[i]]), copy(varFFQ[i])
            end
        else
            mᵢ = mean.(prior_phi_[i, 1:(1+p*dP+i-1)])
            Vᵢ = var.(prior_phi_[i, 1:(1+p*dP+i-1)])
            phi[i, 1:(1+p*dP+i-1)], varFF[i] = NIG_NIG(yphi[:, i], Xphi[:, 1:(end-dP+i-1)], mᵢ, diagm(Vᵢ), shape(prior_varFF_[i]), scale(prior_varFF_[i]))
        end
    end

    return current_chain, kappaQ, phi, varFF

end

"""
    function NUTS_model(yields, PCs, tau_n, macros, dQ, dP, p, dims_phi, prior_diff_kappaQ_, prior_phi_, prior_varFF_; kQ_infty, phi, varFF, SigmaO, data_scale, pca_loadings)
It makes a model in the syntax of `Turing.jl`.
"""

@model function NUTS_model(yields, PCs, tau_n, macros, dQ, dP, p, dims_phi, prior_diff_kappaQ_, prior_phi_, prior_varFF_; kQ_infty, phi, varFF, SigmaO, data_scale, pca_loadings)

    diff_kappaQ ~ product_distribution(prior_diff_kappaQ_)

    phiQ_mean = Vector{Float64}(undef, dims_phi[end])
    phiQ_var_diag = Vector{Float64}(undef, dims_phi[end])
    for i in 1:dQ
        mᵢ = mean.(prior_phi_[i, 1:(1+p*dP+i-1)])
        Vᵢ = var.(prior_phi_[i, 1:(1+p*dP+i-1)])
        start_idx = (i == 1) ? 1 : dims_phi[i-1] + 1
        end_idx = dims_phi[i]
        phiQ_mean[start_idx:end_idx] = mᵢ
        phiQ_var_diag[start_idx:end_idx] = Vᵢ
    end
    phiQ ~ MvNormal(phiQ_mean, diagm(phiQ_var_diag))
    varFFQ ~ product_distribution(prior_varFF_)

    log_lik = loglik_NUTS(yields, PCs, tau_n, macros, dims_phi, p; phiQ, varFFQ, diff_kappaQ, kQ_infty, phi, varFF, SigmaO, data_scale, pca_loadings)
    Turing.@addlogprob! log_lik

    return diff_kappaQ, phiQ, varFFQ
end

"""
    loglik_NUTS(yields, PCs, tau_n, macros, dims_phi, p; phiQ, varFFQ, diff_kappaQ, kQ_infty, phi, varFF, SigmaO, data_scale, pca_loadings)
The function calculate the likelihood of the NUTS block.
"""
function loglik_NUTS(yields, PCs, tau_n, macros, dims_phi, p; phiQ, varFFQ, diff_kappaQ, kQ_infty, phi, varFF, SigmaO, data_scale, pca_loadings)

    phi_full = similar(phi, promote_type(eltype(phi), eltype(phiQ))) |> x -> x .= 0.0
    varFF_full = similar(varFF, promote_type(eltype(varFF), eltype(varFFQ)))

    phi_full[length(varFFQ)+1:end, :] = phi[length(varFFQ)+1:end, :]
    varFF_full[length(varFFQ)+1:end] = varFF[length(varFFQ)+1:end]

    yphi, Xphi = yphi_Xphi(PCs, macros, p)

    T = size(yphi, 1)
    log_pdf = 0.0
    for i in eachindex(varFFQ)
        if i == 1
            phi_full[i, 1:dims_phi[i]] = phiQ[1:dims_phi[i]]
            varFF_full[i] = varFFQ[i]
            log_pdf += logpdf(MvNormal(Xphi * (phi_full[i, :]), varFF_full[i] * I(T)), yphi[:, i])
        else
            phi_full[i, 1:diff(dims_phi)[i-1]] = phiQ[dims_phi[i-1]+1:dims_phi[i]]
            varFF_full[i] = varFFQ[i]
            log_pdf += logpdf(MvNormal(Xphi * (phi_full[i, :]), varFF_full[i] * I(T)), yphi[:, i])
        end
    end
    log_pdf += loglik_mea_NUTS(yields, tau_n; kappaQ=cumsum(diff_kappaQ), kQ_infty, phi=phi_full, varFF=varFF_full, SigmaO, data_scale, pca_loadings)

    return log_pdf
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
    post_SigmaO(yields, tau_n; kappaQ, kQ_infty, ΩPP, gamma, p, data_scale, pca_loadings)
Posterior sampler for the measurement errors
# Output
- `Vector{Dist}(IG, N-dQ)`
"""
function post_SigmaO(yields, tau_n; kappaQ, kQ_infty, ΩPP, gamma, p, data_scale, pca_loadings)
    yields = yields[p+1:end, :]

    dQ = dimQ() + size(yields, 2) - length(tau_n)
    N = length(tau_n)
    T = size(yields, 1)
    PCs, OCs, Wₚ, Wₒ, mean_PCs = PCA(yields, 0; pca_loadings)

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