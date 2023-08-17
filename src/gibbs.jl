
"""
    post_kQ_infty(μkQ_infty, σkQ_infty, yields, τₙ; κQ, ϕ, σ²FF, Σₒ, data_scale)
# Output
- Full conditional posterior distribution
"""
function post_kQ_infty(μkQ_infty, σkQ_infty, yields, τₙ; κQ, ϕ, σ²FF, Σₒ, data_scale)

    dP = length(σ²FF)
    p = Int(((size(ϕ, 2) - 1) / dP) - 1)
    yields = yields[p+1:end, :]

    N = length(τₙ) # of maturities
    T = size(yields, 1) # length of dependent variables
    PCs, OCs, Wₚ, Wₒ, mean_PCs = PCA(yields, 0)

    bτ_ = bτ(τₙ[end]; κQ)
    Bₓ_ = Bₓ(bτ_, τₙ)
    T1X_ = T1X(Bₓ_, Wₚ)
    Bₚ_ = Bₚ(Bₓ_, T1X_, Wₒ)
    ΩPP = ϕ_σ²FF_2_ΩPP(; ϕ, σ²FF)

    a0 = zeros(τₙ[end])
    a1 = zeros(τₙ[end])
    for τ in 2:τₙ[end]
        a0[τ] = a0[τ-1] - jensens_inequality(τ, bτ_, T1X_; ΩPP, data_scale)
        a1[τ] = a1[τ-1] + (τ - 1)
    end
    A0_kQ_infty = a0[τₙ] ./ τₙ
    A1_kQ_infty = a1[τₙ] ./ τₙ

    # Dependent variable
    y = vec(OCs')
    y -= kron(ones(T), Wₒ * (I(N) - Bₓ_ / T1X_ * Wₚ) * A0_kQ_infty + Wₒ * Bₓ_ / T1X_ * mean_PCs)
    y -= vec(Bₚ_ * PCs')
    y ./= kron(ones(T), sqrt.(Σₒ))

    # regressor
    X = Wₒ * (I(N) - Bₓ_ / T1X_ * Wₚ) * A1_kQ_infty
    X ./= sqrt.(Σₒ)
    X = kron(ones(T), X)

    kQ_infty_var = inv(X'X + (1 / (σkQ_infty^2)))
    return Normal(kQ_infty_var * ((μkQ_infty / (σkQ_infty^2)) + X'y), sqrt(kQ_infty_var))

end

"""
    post_κQ(yields, prior_κQ_, τₙ; kQ_infty, ϕ, σ²FF, Σₒ, data_scale)
# Input
- `prior_κQ_` is a output of function `prior_κQ`.
# Output 
- Full conditional posterior distribution
"""
function post_κQ(yields, prior_κQ_, τₙ; kQ_infty, ϕ, σ²FF, Σₒ, data_scale)

    κQ_candidate = support(prior_κQ_)

    kern = Vector{Float64}(undef, length(κQ_candidate)) # Posterior kernel

    for i in eachindex(κQ_candidate)
        # likelihood of the measurement eq
        kern[i] = loglik_mea(yields, τₙ; κQ=κQ_candidate[i], kQ_infty, ϕ, σ²FF, Σₒ, data_scale)
    end

    kern .-= maximum(kern)
    Pr = exp.(kern)
    Pr ./= sum(Pr)

    return DiscreteNonParametric(κQ_candidate, Pr)
end

"""
    post_ϕ_σ²FF(yields, macros, μϕ_const, ρ, prior_κQ_, τₙ; ϕ, ψ, ψ0, σ²FF, q, ν0, Ω0, κQ, kQ_infty, Σₒ, fix_const_PC1, data_scale)
Full-conditional posterior sampler for `ϕ` and `σ²FF` 
# Input
- `prior_κQ_` is a output of function `prior_κQ`.
- When `fix_const_PC1==true`, the first element in a constant term in our orthogonalized VAR is fixed to its prior mean during the posterior sampling.
# Output(3) 
`ϕ`, `σ²FF`, `isaccept=Vector{Bool}(undef, dQ)`
- It gives a posterior sample.
"""
function post_ϕ_σ²FF(yields, macros, μϕ_const, ρ, prior_κQ_, τₙ; ϕ, ψ, ψ0, σ²FF, q, ν0, Ω0, κQ, kQ_infty, Σₒ, fix_const_PC1, data_scale)

    dQ = dimQ()
    dP = size(ψ, 1)
    p = Int(size(ψ)[2] / dP)
    PCs, ~, Wₚ = PCA(yields, p)

    yϕ, Xϕ = yϕ_Xϕ(PCs, macros, p)
    prior_ϕ0_ = prior_ϕ0(μϕ_const, ρ, prior_κQ_, τₙ, Wₚ; ψ0, ψ, q, ν0, Ω0, fix_const_PC1)
    prior_ϕ_ = [prior_ϕ0_ prior_C(; Ω0)]
    prior_σ²FF_ = prior_σ²FF(; ν0, Ω0)

    isaccept = fill(false, dQ)
    for i in 1:dP
        if i <= dQ
            prop_ϕ = deepcopy(ϕ) # proposal for C
            prop_σ²FF = deepcopy(σ²FF) # proposal for σ²FF

            mᵢ = mean.(prior_ϕ_[i, 1:(1+p*dP+i-1)])
            Vᵢ = var.(prior_ϕ_[i, 1:(1+p*dP+i-1)])
            prop_ϕ[i, 1:(1+p*dP+i-1)], prop_σ²FF[i] = NIG_NIG(yϕ[:, i], Xϕ[:, 1:(end-dP+i-1)], mᵢ, diagm(Vᵢ), shape(prior_σ²FF_[i]), scale(prior_σ²FF_[i]))

            prob = loglik_mea(yields, τₙ; κQ, kQ_infty, ϕ=prop_ϕ, σ²FF=prop_σ²FF, Σₒ, data_scale)
            prob -= loglik_mea(yields, τₙ; κQ, kQ_infty, ϕ, σ²FF, Σₒ, data_scale)

            if rand() < min(1.0, exp(prob))
                ϕ = deepcopy(prop_ϕ)
                σ²FF = deepcopy(prop_σ²FF)
                isaccept[i] = true
            end
        else
            mᵢ = mean.(prior_ϕ_[i, 1:(1+p*dP+i-1)])
            Vᵢ = var.(prior_ϕ_[i, 1:(1+p*dP+i-1)])
            ϕ[i, 1:(1+p*dP+i-1)], σ²FF[i] = NIG_NIG(yϕ[:, i], Xϕ[:, 1:(end-dP+i-1)], mᵢ, diagm(Vᵢ), shape(prior_σ²FF_[i]), scale(prior_σ²FF_[i]))
        end
    end

    return ϕ, σ²FF, isaccept

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
    post_Σₒ(yields, τₙ; κQ, kQ_infty, ΩPP, γ, p, data_scale)
Posterior sampler for the measurement errors
# Output
- `Vector{Dist}(IG, N-dQ)`
"""
function post_Σₒ(yields, τₙ; κQ, kQ_infty, ΩPP, γ, p, data_scale)
    yields = yields[p+1:end, :]

    dQ = dimQ()
    N = length(τₙ)
    T = size(yields, 1)
    PCs, OCs, Wₚ, Wₒ, mean_PCs = PCA(yields, 0)

    bτ_ = bτ(τₙ[end]; κQ)
    Bₓ_ = Bₓ(bτ_, τₙ)
    T1X_ = T1X(Bₓ_, Wₚ)
    Bₚ_ = Bₚ(Bₓ_, T1X_, Wₒ)

    aτ_ = aτ(τₙ[end], bτ_, τₙ, Wₚ; kQ_infty, ΩPP, data_scale)
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
Posterior sampler for the population measurement error
# Output
- `Vector{Dist}(Gamma,length(Σₒ))`
"""
function post_γ(; γ_bar, Σₒ)

    N = length(Σₒ) # of measurement errors

    post_γ_ = Vector{Any}(undef, N)
    for i in 1:N
        post_γ_[i] = Gamma(3, 1 / (γ_bar + (1 / Σₒ[i])))
    end

    return post_γ_
end
