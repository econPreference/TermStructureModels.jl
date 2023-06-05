####################
## Gibbs sampling ##
####################
# This file contains the full-conditional posterior distribution of all parameters including the MH blocks.

"""
post_kQ_infty(μkQ_infty, yields, τₙ; κQ, ϕ, σ²FF, Σₒ)
* Input: yields should exclude initial observations. μkQ_infty is a prior variance.
* Output: Posterior distribution itself
"""
function post_kQ_infty(μkQ_infty, σkQ_infty, yields, τₙ; κQ, ϕ, σ²FF, Σₒ)

    N = length(τₙ) # of maturities
    T = size(yields, 1) # length of dependent variables
    PCs, OCs, Wₚ, Wₒ = PCA(yields, 0)

    bτ_ = bτ(τₙ[end]; κQ)
    Bₓ_ = Bₓ(bτ_, τₙ)
    T1X_ = T1X(Bₓ_, Wₚ)
    Bₚ_ = Bₚ(Bₓ_, T1X_, Wₒ)
    ΩPP = ϕ_σ²FF_2_ΩPP(; ϕ, σ²FF)

    a0 = zeros(τₙ[end])
    a1 = zeros(τₙ[end])
    for τ in 2:τₙ[end]
        a0[τ] = a0[τ-1] - jensens_inequality(τ, bτ_, T1X_; ΩPP)
        a1[τ] = a1[τ-1] + (τ - 1)
    end
    A0_kQ_infty = a0[τₙ] ./ τₙ
    A1_kQ_infty = a1[τₙ] ./ τₙ

    # Dependent variable
    y = vec(OCs')
    y -= kron(ones(T), Wₒ * (I(N) - Bₓ_ / T1X_ * Wₚ) * A0_kQ_infty)
    y -= vec(Bₚ_ * PCs')
    y = y ./ kron(ones(T), sqrt.(1 ./ Σₒ))

    # regressor
    X = Wₒ * (I(N) - Bₓ_ / T1X_ * Wₚ) * A1_kQ_infty
    X = X ./ (sqrt.(1 ./ Σₒ))
    X = kron(ones(T), X)

    kQ_infty_var = inv(X'X + (1 / (σkQ_infty^2)))
    return Normal(kQ_infty_var * ((μkQ_infty / (σkQ_infty^2)) + X'y), sqrt(kQ_infty_var))

end

"""
post_κQ(yields, prior_κQ_, τₙ; kQ_infty, ϕ, σ²FF, Σₒ)
* Input: data should exclude initial observations
* Output: Posterior distribution itself
"""
function post_κQ(yields, prior_κQ_, τₙ; kQ_infty, ϕ, σ²FF, Σₒ)
    κQ_candidate = support(prior_κQ_)

    kern = Vector{Float64}(undef, length(κQ_candidate)) # Posterior kernel

    for i in eachindex(κQ_candidate)
        # likelihood of the measurement eq
        kern[i] = loglik_mea(yields, τₙ; κQ=κQ_candidate[i], kQ_infty, ϕ, σ²FF, Σₒ)
    end

    kern .-= maximum(kern)
    Pr = exp.(kern)
    Pr = Pr / sum(Pr)

    return DiscreteNonParametric(κQ_candidate, Pr)
end

"""
post_σ²FF₁(yields, macros, τₙ, p; κQ, kQ_infty, ϕ, σ²FF, Σₒ, ν0, Ω0)
* Input: Data should contain initial observations.
* Output(2): σ²FF, isaccept
    - a posterior sample of σ²FF is returned, but only σ²FF[1] is updated
"""
function post_σ²FF₁(yields, macros, τₙ, p; κQ, kQ_infty, ϕ, σ²FF, Σₒ, ν0, Ω0)

    dP = length(Ω0)
    T = size(yields, 1) - p # length of dependent variable
    PCs = PCA(yields, p)[1]

    yϕ, Xϕ = yϕ_Xϕ(PCs, macros, p)
    y = yϕ[:, 1]
    fitted = Xϕ * (ϕ[1, :])
    RSS = (y - fitted)' * (y - fitted)

    prop_σ²FF = deepcopy(σ²FF) # proposal
    prop_σ²FF[1] = rand(InverseGamma(0.5 * (ν0 + 1 - dP + T), 0.5 * (Ω0[1] + RSS))) # a sample from the proposal distribution

    prob = loglik_mea(yields[(p+1):end, :], τₙ; κQ, kQ_infty, ϕ, σ²FF=prop_σ²FF, Σₒ)
    prob -= loglik_mea(yields[(p+1):end, :], τₙ; κQ, kQ_infty, ϕ, σ²FF, Σₒ)

    if rand() < min(1.0, exp(prob))
        return prop_σ²FF, true
    else
        return σ²FF, false
    end

end

"""
post_C_σ²FF_dQ(yields, macros, τₙ, p; κQ, kQ_infty, ϕ, σ²FF, Σₒ, ν0, Ω0)
* It make a posterior sample of components in ΩPP, except for σ²FF₁.
* Input: data should contain initial observations.
* Output(3): ϕ, σ²FF, isaccept
    - posterior samples of [ϕ0 C0], σ²FF are returned
    - 2~dQ rows of C0 and σ²FF are updated
"""
function post_C_σ²FF_dQ(yields, macros, τₙ, p; κQ, kQ_infty, ϕ, σ²FF, Σₒ, ν0, Ω0)

    dQ = dimQ()
    PCs = PCA(yields, p)[1]

    yϕ, ~, Xϕ0, XC = yϕ_Xϕ(PCs, macros, p)
    ϕ0, ~, C0 = ϕ_2_ϕ₀_C(; ϕ)
    prior_C_ = prior_C(; Ω0)
    prior_σ²FF_ = prior_σ²FF(; ν0, Ω0)

    isaccept = fill(false, dQ - 1)
    for i in 2:dQ
        prop_C0 = deepcopy(C0) # proposal for C
        prop_σ²FF = deepcopy(σ²FF) # proposal for σ²FF

        y = yϕ[:, i] - Xϕ0 * ϕ0[i, :]
        X = XC[:, 1:(i-1)]
        prop_C0[i, 1:(i-1)], prop_σ²FF[i] = NIG_NIG(y, X, zeros(i - 1), diagm(var.(prior_C_[i, 1:(i-1)])), shape(prior_σ²FF_[i]), scale(prior_σ²FF_[i]))

        prop_ϕ = [ϕ0 prop_C0] # proposal for ϕ

        prob = loglik_mea(yields[(p+1):end, :], τₙ; κQ, kQ_infty, ϕ=prop_ϕ, σ²FF=prop_σ²FF, Σₒ)
        prob -= loglik_mea(yields[(p+1):end, :], τₙ; κQ, kQ_infty, ϕ, σ²FF, Σₒ)

        if rand() < min(1.0, exp(prob))
            C0 = deepcopy(prop_C0)
            σ²FF = deepcopy(prop_σ²FF)
            isaccept[i-1] = true
        end

    end
    return [ϕ0 C0], σ²FF, isaccept

end

"""
post_ηψ(; ηψ, ψ, ψ0)
* Posterior sampler for the sparsity parameters
* Output(2): ηψ, isaccept
    - a sample from the MH algorithm.
"""
function post_ηψ(; ηψ, ψ, ψ0)

    dP = size(ψ)[1]
    p = Int(size(ψ)[2] / dP)

    obj(_ηψ) = dlogηψ_dηψ(_ηψ; ψ, ψ0)
    function find_ηψ(obj_, ηψ_)
        try
            ηψ_hat_ = fzero(obj_, ηψ_)
        catch
            ηψ_hat_ = ηψ_
        end
    end
    ηψ_hat = find_ηψ(obj, ηψ)
    ηψ_hess = d2logηψ_dηψ2(ηψ_hat; dP, p)

    function log_target(arg_ηψ; ψ, ψ0) # See the posterior kernel formula in terms of distributions not pdfs.
        logpdf_ = sum(logpdf.(Gamma(arg_ηψ, 1 / arg_ηψ), ψ))
        logpdf_ += sum(logpdf.(Gamma(arg_ηψ, 1 / arg_ηψ), ψ0))
        logpdf_ += logpdf(Gamma(1, 1), arg_ηψ)
        return logpdf_
    end
    proposal_dist = truncated(TDist(15, ηψ_hat, -1 / ηψ_hess); lower=0)
    prop_ηψ = rand(proposal_dist)

    prob = log_target(prop_ηψ; ψ, ψ0)
    prob -= logpdf(proposal_dist, prop_ηψ)
    prob += logpdf(proposal_dist, ηψ)
    prob -= log_target(ηψ; ψ, ψ0)

    if rand() < min(1.0, exp(prob))
        return prop_ηψ, true
    else
        return ηψ, false
    end
end
"""
TDist(df, μ, σ²) = μ + TDist(df) * sqrt(σ²)
* Generalized T distribution with scaling parameter σ² and location parameter μ
* mean: μ, var: df*σ²/(df-2)
"""
TDist(df, μ, σ²) = μ + TDist(df) * sqrt(σ²)

"""
dlogηψ_dηψ(ηψ; ψ, ψ0)
* It is a first derivative of the log posterior density w.r.t. ηψ
"""
function dlogηψ_dηψ(ηψ; ψ, ψ0)

    if ηψ < 0
        return -Inf
    end

    dP = size(ψ, 1)
    p = Int(size(ψ, 2) / dP)
    allψ = [ψ0 ψ] # align with ϕ

    deri = log(ηψ) + 1 - digamma(ηψ)
    deri *= p * (dP^2) + dP
    deri += sum(log.(allψ))
    deri -= sum(allψ) + 1

    return deri
end

"""
d2logηψ_dηψ2(ηψ; dP, p)
* It is a second derivative of the log posterior density w.r.t. ηψ
"""
function d2logηψ_dηψ2(ηψ; dP, p)

    deri = 1 / ηψ
    deri -= trigamma(ηψ)
    deri *= p * (dP^2) + dP

    return deri
end

"""
post_ϕ_σ²FF_remaining(PCs, macros, ρ, prior_κQ_, τₙ, Wₚ; ϕ, ψ, ψ0, σ²FF, q, ν0, Ω0)
* Posterior sampler for ϕ and σ²FF that are not sampled by the MH. 
* Input: data should contain initial observations.
* Output(2): ϕ, σ²FF
    - It gives a posterior sample, and it is updated for the remaining elements that are not in MH block.
"""
function post_ϕ_σ²FF_remaining(PCs, macros, μϕ_const, ρ, prior_κQ_, τₙ, Wₚ; ϕ, ψ, ψ0, σ²FF, q, ν0, Ω0)

    dQ = dimQ()
    dP = size(ψ, 1)
    p = Int(size(ψ)[2] / dP)

    yϕ, Xϕ, Xϕ0, XC = yϕ_Xϕ(PCs, macros, p)
    ~, ~, C0 = ϕ_2_ϕ₀_C(; ϕ)
    prior_ϕ0_ = prior_ϕ0(μϕ_const, ρ, prior_κQ_, τₙ, Wₚ; ψ0, ψ, q, ν0, Ω0)
    prior_ϕ_ = [prior_ϕ0_ prior_C(; Ω0)]
    prior_σ²FF_ = prior_σ²FF(; ν0, Ω0)

    # for i = 1
    mᵢ = mean.(prior_ϕ0_[1, :])
    Vᵢ = var.(prior_ϕ0_[1, :])
    ϕ[1, 1:(1+p*dP)] = Normal_Normal_in_NIG(yϕ[:, 1], Xϕ0, mᵢ, diagm(Vᵢ), σ²FF[1])

    for i in 2:dQ

        y = yϕ[:, i] - XC * C0[i, :]
        mᵢ = mean.(prior_ϕ0_[i, :])
        Vᵢ = var.(prior_ϕ0_[i, :])
        ϕ[i, 1:(1+p*dP)] = Normal_Normal_in_NIG(y, Xϕ0, mᵢ, diagm(Vᵢ), σ²FF[i])

    end
    for i in dQ+1:dP

        mᵢ = mean.(prior_ϕ_[i, 1:(1+p*dP+i-1)])
        Vᵢ = var.(prior_ϕ_[i, 1:(1+p*dP+i-1)])
        ϕ[i, 1:(1+p*dP+i-1)], σ²FF[i] = NIG_NIG(yϕ[:, i], Xϕ[:, 1:(end-dP+i-1)], mᵢ, diagm(Vᵢ), shape(prior_σ²FF_[i]), scale(prior_σ²FF_[i]))

    end

    return ϕ, σ²FF

end


"""
Normal_Normal_in_NIG(y, X, β₀, B₀, σ²)
* Normal-Normal update part in NIG-NIG update, given σ²
    - prior: β ~ MvNormal(β₀,σ²B₀)
    - likelihood: y|β = Xβ + MvNormal(zeros(T,1),σ²I(T)) 
* Output: posterior sample
"""
function Normal_Normal_in_NIG(y, X, β₀, B₀, σ²)

    R"library(MASS)"

    inv_B₀ = inv(B₀)
    B₁ = Symmetric(inv(inv_B₀ + X'X))
    β₁ = B₁ * (inv_B₀ * β₀ + X'y)

    return rcopy(Array, rcall(:mvrnorm, mu=β₁, Sigma=σ² * B₁))
end

# """
# Normal_Normal(y, X, β₀, B₀, σ²)
# * Normal-Normal update, given σ²
#     - prior: β ~ MvNormal(β₀,B₀)
#     - likelihood: y|β = Xβ + MvNormal(zeros(T,1),σ²I(T)) 
# * Output: posterior sample
# """
# function Normal_Normal(y, X, β₀, B₀, σ²)

#     B₁ = inv(inv(B₀) + (X'X / σ²))
#     B₁ = 0.5(B₁ + B₁')
#     β₁ = B₁ * (B₀ \ β₀ + (X'y / σ²))

#     return rand(MvNormal(β₁, B₁))
# end


"""
NIG_NIG(y, X, β₀, B₀, α₀, δ₀)
* Normal-InverseGamma-Normal-InverseGamma update
    - prior: β|σ² ~ MvNormal(β₀,σ²B₀), σ² ~ InverseGamma(α₀,δ₀)
    - likelihood: y|β,σ² = Xβ + MvNormal(zeros(T,1),σ²I(T)) 
* Output(2): β, σ²
    - posterior sample
"""
function NIG_NIG(y, X, β₀, B₀, α₀, δ₀)
    R"library(MASS)"
    T = length(y)

    inv_B₀ = inv(B₀)
    inv_B₁ = inv_B₀ + X'X
    B₁ = Symmetric(inv(inv_B₁))
    β₁ = B₁ * (inv_B₀ * β₀ + X'y)
    δ₁ = δ₀ + 0.5 * (y'y + β₀' * inv_B₀ * β₀ - β₁' * inv_B₁ * β₁)

    if δ₁ < eps() || isposdef(B₁) == false
        idx_deg = []
        diag_B₀ = diag(B₀)
        while δ₁ < eps() || isposdef(B₁) == false
            push!(idx_deg, findmin(diag_B₀)[2])
            diag_B₀[findmin(diag_B₀)[2]] = maximum(diag_B₀)
            idx = collect(1:length(β₀))
            for i in eachindex(idx_deg)
                aux_idx = similar(idx)
                aux_idx = collect(1:length(β₀)) .!= idx_deg[i]
                idx .*= aux_idx
            end
            idx = findall(idx .> 0)
            B₀_deg = B₀[idx, idx]
            β₀_deg = β₀[idx]
            X_deg = X[:, idx]
            y_deg = y - X[:, idx_deg] * β₀[idx_deg]

            inv_B₀ = inv(B₀_deg)
            inv_B₁ = inv_B₀ + X_deg'X_deg
            B₁ = Symmetric(inv(inv_B₁))
            β₁ = B₁ * (inv_B₀ * β₀_deg + X_deg'y_deg)
            δ₁ = δ₀ + 0.5 * (y_deg'y_deg + β₀_deg' * inv_B₀ * β₀_deg - β₁' * inv_B₁ * β₁)

            if δ₁ > eps() && isposdef(B₁)
                σ² = rand(InverseGamma(α₀ + 0.5T, δ₁))
                β = deepcopy(β₀)
                β[idx] = rcopy(Array, rcall(:mvrnorm, mu=β₁, Sigma=σ² * B₁))
                return β, σ²
            end
        end
    end

    σ² = rand(InverseGamma(α₀ + 0.5T, δ₁))
    β = rand(MvNormal(β₁, σ² * B₁))

    return β, σ²
end

"""
post_ψ_ψ0(ρ, prior_κQ_, τₙ, Wₚ; ϕ, ψ0, ψ, ηψ, q, σ²FF, ν0, Ω0)
* posterior sampler for the sparsity parameters
* Output(2): ψ0, ψ
    - posterior samples
"""
function post_ψ_ψ0(μϕ_const, ρ, prior_κQ_, τₙ, Wₚ; ϕ, ψ0, ψ, ηψ, q, σ²FF, ν0, Ω0)

    R"library(GIGrvg)"
    dP = size(ψ, 1)
    p = Int(size(ψ, 2) / dP)
    priormean_ϕ0_ = mean.(prior_ϕ0(μϕ_const, ρ, prior_κQ_, τₙ, Wₚ; ψ0, ψ, q, ν0, Ω0))
    post_ψ = similar(ψ)
    post_ψ0 = similar(ψ0)

    for l in 1:p, i in 1:dP, j in 1:dP # slope coefficients
        m = priormean_ϕ0_[i, 1+(l-1)dP+j]
        V = σ²FF[i] * Minnesota(l, i, j; q, ν0, Ω0)

        std_ϕ = (ϕ[i, 1+(l-1)dP+j] - m)^2
        std_ϕ /= V
        post_ψ[i, dP*(l-1)+j] = rcopy(rcall(:rgig, lambda=ηψ - 0.5, chi=max(eps(), std_ϕ), psi=2ηψ))
    end

    for i in 1:dP # intercepts
        m = priormean_ϕ0_[i, 1]
        if i < dimQ() + 1
            V = q[4, 1] * σ²FF[i]
        else
            V = q[4, 2] * σ²FF[i]
        end

        std_ϕ = (ϕ[i, 1] - m)^2
        std_ϕ /= V
        post_ψ0[i] = rcopy(rcall(:rgig, lambda=ηψ - 0.5, chi=max(eps(), std_ϕ), psi=2ηψ))
    end

    return post_ψ0, post_ψ
end

"""
post_Σₒ(yields, τₙ; κQ, kQ_infty, ΩPP, γ)
* Posterior sampler for the measurement errors
* Input: Data excludes initial observations
* Output: Vector{Dist}(IG, N-dQ)
"""
function post_Σₒ(yields, τₙ; κQ, kQ_infty, ΩPP, γ)

    dQ = dimQ()
    N = length(τₙ)
    T = size(yields, 1)
    PCs, OCs, Wₚ, Wₒ, mean_PCs = PCA(yields, 0)

    bτ_ = bτ(τₙ[end]; κQ)
    Bₓ_ = Bₓ(bτ_, τₙ)
    T1X_ = T1X(Bₓ_, Wₚ)
    Bₚ_ = Bₚ(Bₓ_, T1X_, Wₒ)

    aτ_ = aτ(τₙ[end], bτ_, τₙ, Wₚ; kQ_infty, ΩPP)
    Aₓ_ = Aₓ(aτ_, τₙ)
    T0P_ = T0P(T1X_, Aₓ_, Wₚ, mean_PCs)
    Aₚ_ = Aₚ(Aₓ_, Bₓ_, T0P_, Wₒ)

    post_Σₒ_ = Vector{Any}(undef, N - dQ)
    for i in 1:N-dQ
        residuals = OCs[:, i] - (Aₚ_[i] .+ (Bₚ_[i, :]' * PCs')')
        post_Σₒ_[i] = InverseGamma(2 + 0.5T, γ[i] + 0.5residuals'residuals)
    end

    return post_Σₒ_
end

"""
post_γ(; γ_bar, Σₒ)
* Posterior sampler for the population measurement error
* Input: γ_bar comes from function prior_γ.
* Output: Vector{Dist}(Gamma,length(Σₒ))
"""

function post_γ(; γ_bar, Σₒ)

    N = length(Σₒ) # of measurement errors

    post_γ_ = Vector{Any}(undef, N)
    for i in 1:N
        post_γ_[i] = Gamma(3, γ_bar + (1 / Σₒ[i]))
    end

    return post_γ_
end
