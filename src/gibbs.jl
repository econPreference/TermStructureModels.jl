####################
## Gibbs sampling ##
####################
# This file contains the full-conditional posterior distribution of all parameters including the MH block.

"""
    * Input: yields should include initial conditions
===
"""
function post_kQ_infty(σ²kQ_infty, yields, τₙ, p; κQ, ϕ, σ²FF, Σₒ)

    dQ = dimQ()
    N = length(τₙ)
    T = size(yields, 1) - p
    PCs, OCs, Wₚ, Wₒ = PCA(yields, p)
    PCs = PCs[(p+1):end, :]
    OCs = OCs[(p+1):end, :]
    yields = yields[(p+1):end, :]

    bτ_ = bτ(τₙ[end]; κQ)
    Bₓ_ = Bₓ(bτ_, τₙ)
    T1X_ = T1X(Bₓ_; Wₚ)
    Bₚ_ = Bₚ(Bₓ_, T1X_; Wₒ)
    ΩPP = ϕ_σ²FF_2_ΩPP(; ϕ, σ²FF)

    a0 = zeros(τₙ[end])
    a1 = zeros(τₙ[end])
    for τ in 2:τₙ[end]
        a0[τ] = a0[τ-1] - Jensens(τ, bτ_, T1X_; ΩPP)
        a1[τ] = a1[τ-1] + (τ - 1)
    end
    A0_kQ_infty = a0[τₙ] ./ τₙ
    A1_kQ_infty = a1[τₙ] ./ τₙ

    # Dependent variable
    y = vec(OCs')
    y -= kron(ones(T), Wₒ * (I(N) - Bₓ_' / T1X_ * Wₚ) * A0_kQ_infty)
    y -= vec(Bₚ_ * PCs')
    y = y ./ kron(ones(T), sqrt.(1 ./ Σₒ))

    # regressor
    X = Wₒ * (I(N) - Bₓ_' / T1X_ * Wₚ) * A1_kQ_infty
    X = X ./ (sqrt.(1 ./ Σₒ))
    X = kron(ones(T), X)

    kQ_infty_var = inv(X'X + (1 / σ²kQ_infty))
    return Normal(kQ_infty_var * X'y, sqrt(kQ_infty_var))

end

"""
    * Input: data should exclude initial conditions
"""
function post_κQ(yields, prior_κQ_, τₙ, p; kQ_infty, ϕ, σ²FF, Σₒ)
    κQ_candidate = support(prior_κQ_)

    kern = Vector{Float64}(undef, length(κQ_candidate)) # Posterior kernel

    for i in eachindex(κQ_candidate)
        κQ = κQ_candidate[i]

        # likelihood
        kern[i] = loglik_mea(yields, τₙ, p; κQ, kQ_infty, ϕ, σ²FF, Σₒ)

    end

    kern .-= maximum(kern)
    Pr = exp.(kern)
    Pr = Pr / sum(Pr)

    return DiscreteNonParametric(κQ_candidate, Pr)
end

"""
Note that it does not return a distribution, but a sample from the MH algorithm. Also, data should contain initial conditions.

===
"""
function post_σ²FF₁(yields, macros, τₙ, p; κQ, kQ_infty, ϕ, σ²FF, Σₒ, ν0, Ω0)

    dQ = dimQ()
    dP = size(Ω0)[1]
    T = size(yields, 1)
    PCs = PCA(yields, p)[1]

    yϕ, Xϕ = yϕ_Xϕ(PCs, macros, p)
    y = yϕ[:, 1]
    fitted = Xϕ * (ϕ[1, :])
    RSS = (y - fitted)' * (y - fitted)

    prop_σ²FF = copy(σ²FF)
    prop_σ²FF[1] = rand(InverseGamma(0.5 * (ν0 + 1 - dP + T), 0.5 * (Ω0[1] + RSS))) # a sample from the proposal distribution

    prob = loglik_mea(yields, τₙ, p; κQ, kQ_infty, ϕ, σ²FF=prop_σ²FF, Σₒ)
    prob -= loglik_mea(yields, τₙ, p; κQ, kQ_infty, ϕ, σ²FF, Σₒ)

    if rand() < min(1.0, exp(prob))
        return prop_σ²FF
    else
        return σ²FF
    end

end

"""
It make a posterior sample of ΩPP, except for σ²FF₁. Note that it does not return a distribution, but a sample from the MH algorithm.

    * Input: data should contain initial conditions.
===
"""
function post_C_σ²FF_dQ(yields, macros, τₙ, p; κQ, kQ_infty, ϕ, σ²FF, Σₒ, ν0, Ω0)

    dQ = dimQ()
    dP = length(Ω0)

    std_yields = yields[p+1:end, :] .- mean(yields[p+1:end, :], dims=1)
    std_yields ./= std(yields[p+1:end, :], dims=1)
    V = reverse(eigen(cov(std_yields)).vectors, dims=2)
    Wₚ = V[:, 1:dQ]'
    PCs = (Wₚ * yields')'

    yϕ, ~, Xϕ0, XC = yϕ_Xϕ(PCs, macros, p)
    ϕ0, ~, C0 = ϕ_2_ϕ₀_C(; ϕ)

    for i in 2:dQ
        prop_C0 = copy(C0)
        prop_σ²FF = copy(σ²FF)

        y = yϕ[:, i] - Xϕ0 * ϕ0[i, :]
        X = XC[:, 1:(i-1)]
        prop_C0[i, 1:(i-1)], prop_σ²FF[i] = NIG_NIG(y, X, zeros(i - 1), diagm(1 ./ Ω0[1:(i-1)]), 0.5(ν0 + i - dP), 0.5Ω0[i])

        prop_ϕ = [ϕ0 prop_C0]

        prob = loglik_mea(yields, τₙ, p; κQ, kQ_infty, ϕ=prop_ϕ, σ²FF=prop_σ²FF, Σₒ)
        prob -= loglik_mea(yields, τₙ, p; κQ, kQ_infty, ϕ, σ²FF, Σₒ)

        if rand() < min(1.0, exp(prob))
            C0 = copy(prop_C0)
            σ²FF = copy(prop_σ²FF)
        end

    end
    return [ϕ0 C0], σ²FF

end

"""
Note that it does not return a distribution, but a sample from the MH algorithm.

===
"""
function post_ηψ(; ηψ, ψ, ψ0)

    dP = size(ψ)[1]
    p = size(ψ)[2] / dP

    obj(_ηψ) = dlogηψ_dηψ(_ηψ; ψ, ψ0)
    ηψ_hat = fzero(obj, ηψ)
    ηψ_hess = d2logηψ_dηψ2(ηψ_hat; dP, p)


    function log_target(ηψ; ψ, ψ0)
        logpdf_ = sum(logpdf.(Gamma(ηψ, 1 / ηψ), ψ))
        logpdf_ += sum(logpdf.(Gamma(ηψ, 1 / ηψ), ψ0))
        logpdf_ += logpdf(Gamma(1, 1), ηψ)
        return logpdf_
    end
    proposal_dist = truncated(TDist(15, ηψ_hat, -1 / ηψ_hess); lower=0)
    prop_ηψ = rand(proposal_dist)

    prob = log_target(prop_ηψ; ψ, ψ0)
    prob -= logpdf(proposal_dist, prop_ηψ)
    prob += logpdf(proposal_dist, ηψ)
    prob -= log_target(ηψ; ψ, ψ0)

    if rand() < min(1.0, exp(prob))
        return prop_ηψ
    else
        return ηψ
    end
end
"""
Generalized T distribution

===
"""
TDist(df, μ, σ²) = μ + TDist(df) * sqrt(σ²)

"""
    This part generate a full conditional distribution of ϕ and σ²FF that are not sampled by the MH. It gives a posterior sample.

    * Input: data should contain initial conditions.
===
"""
function post_ϕ_σ²FF_remaining(PCs, macros, ρ; ϕ, ψ, ψ0, σ²FF, q, ν0, Ω0)

    dQ = dimQ()
    dP = size(ψ)[1]
    p = Int(size(ψ)[2] / dP)

    yϕ, Xϕ, Xϕ0, XC = yϕ_Xϕ(PCs, macros, p)
    ~, ~, C0 = ϕ_2_ϕ₀_C(; ϕ)
    prior_ϕ0_ = prior_ϕ0(ρ; ψ0, ψ, q, ν0, Ω0)
    prior_ϕ_ = [prior_ϕ0_ prior_C(; Ω0)]

    # for i = 1
    y = yϕ[:, 1]
    X = Xϕ0
    mᵢ = mean.(prior_ϕ0_[1, :])
    Vᵢ = var.(prior_ϕ0_[1, :])
    ϕ[1, 1:(1+p*dP)] = rand(Normal_Normal_in_NIG(y, X, mᵢ, diagm(Vᵢ), σ²FF[1]))

    for i in 2:dQ

        y = yϕ[:, i] - XC * C0[i, :]
        X = Xϕ0
        mᵢ = mean.(prior_ϕ0_[i, :])
        Vᵢ = var.(prior_ϕ0_[i, :])
        ϕ[i, 1:(1+p*dP)] = rand(Normal_Normal_in_NIG(y, X, mᵢ, diagm(Vᵢ), σ²FF[i]))

    end
    for i in dQ+1:dP

        y = yϕ[:, i]
        X = Xϕ[:, 1:(end-dP+i-1)]
        mᵢ = mean.(prior_ϕ_[i, 1:(1+p*dP+i-1)])
        Vᵢ = var.(prior_ϕ_[i, 1:(1+p*dP+i-1)])
        ϕ[i, 1:(1+p*dP+i-1)], σ²FF[i] = NIG_NIG(y, X, mᵢ, diagm(Vᵢ), 0.5(ν0 + i - dP), 0.5Ω0[i])

    end

    return ϕ, σ²FF

end

function post_ψ_ψ0(ρ; ϕ, ψ0, ψ, ηψ, q, σ²FF, ν0, Ω0)

    R"library(GIGrvg)"
    dP = size(ϕ, 1)
    p = Int((size(ϕ, 2) - 1) / dP) - 1
    priormean_ϕ0_ = mean.(prior_ϕ0(ρ; ψ0, ψ, q, ν0, Ω0))
    post_ψ = Matrix{Float64}(undef, dP, dP * p)
    post_ψ0 = Vector{Float64}(undef, dP)

    for l in 1:p, i in 1:dP, j in 1:dP
        m = priormean_ϕ0_[i, 1+(l-1)dP+j]
        V = σ²FF[i] * Minnesota(l, i, j; q, ν0, Ω0)

        std_ϕ = (ϕ[i, 1+(l-1)dP+j] - m)^2
        std_ϕ /= V
        post_ψ[i, dP*(l-1)+j] = rcopy(rcall(:rgig, lambda=ηψ - 0.5, chi=std_ϕ, psi=2ηψ))
    end

    for i in 1:dP
        m = priormean_ϕ0_[i, 1]
        V = q[4] * σ²FF[i]

        std_ϕ = (ϕ[i, 1] - m)^2
        std_ϕ /= V
        post_ψ0[i] = rcopy(rcall(:rgig, lambda=ηψ - 0.5, chi=std_ϕ, psi=2ηψ))
    end

    return post_ψ, post_ψ0
end

"""
    * Input: Data excludes initial conditions

===
"""
function post_Σₒ(yields, τₙ; κQ, kQ_infty, ΩPP, γ)

    dQ = dimQ()
    N = length(τₙ)
    T = size(yields)[1]

    std_yields = yields .- mean(yields, dims=1)
    std_yields ./= std(yields, dims=1)
    V = reverse(eigen(cov(std_yields)).vectors, dims=2)
    Wₚ = V[:, 1:dQ]'
    Wₒ = V[:, (dQ+1):end]'
    PCs = (Wₚ * yields')'
    OCs = (Wₒ * yields')'

    bτ_ = bτ(τₙ[end]; κQ)
    Bₓ_ = Bₓ(bτ_, τₙ)
    T1X_ = T1X(Bₓ_; Wₚ)
    Bₚ_ = Bₚ(Bₓ_, T1X_; Wₒ)

    aτ_ = aτ(τₙ[end], bτ_, τₙ; kQ_infty, ΩPP, Wₚ)
    Aₓ_ = Aₓ(aτ_, τₙ)
    T0P_ = T0P(T1X_, Aₓ_; Wₚ)
    Aₚ_ = Aₚ(Aₓ_, Bₓ_, T0P_; Wₒ)

    post_Σₒ_ = Vector{Any}(undef, N)
    for i in 1:N-dQ
        residuals = OCs[:, i] - (Aₚ_[i] .+ (Bₚ_[i, :]' * PCs')')
        post_Σₒ_[i] = InverseGamma(2 + 0.5T, γ[i] + 0.5residuals'residuals)
    end

    return post_Σₒ_
end

function post_γ(γ_bar; Σₒ)

    N = length(Σₒ)

    post_γ_ = Vector{Any}(undef, N)
    for i in 1:N
        post_γ_[i] = Gamma(3, γ_bar + (1 / Σₒ[i]))
    end

    return post_γ_
end

"""
Normal-Normal update in NIG-NIG update

    Prior: ``\\beta|\\sigma^{2} & \\backsim\\mathcal{N}(\\beta_{0},\\sigma^{2}B_{0})``
===
"""
function Normal_Normal_in_NIG(y, X, β₀, B₀, σ²)

    B₁ = inv(inv(B₀) + X'X)
    B₁ = 0.5(B₁ + B₁')
    β₁ = B₁ * (B₀ \ β₀ + X'y)

    return MvNormal(β₁, σ² * B₁)
end

"""
Normal-Normal update

    Prior: ``\\beta & \\backsim\\mathcal{N}(\\beta_{0},B_{0})``
===
"""
function Normal_Normal(y, X, β₀, B₀, σ²)

    B₁ = inv(inv(B₀) + X'X / σ²)
    B₁ = 0.5(B₁ + B₁')
    β₁ = B₁ * (B₀ \ β₀ + X'y / σ²)

    return MvNormal(β₁, B₁)
end


"""
Normal-InverseGamma-Normal-InverseGamma update

    Prior: ``\\sigma^{2} & \\backsim\\mathcal{IG}(\\alpha_{0},\\delta_{0})``
=== 
"""
function NIG_NIG(y, X, β₀, B₀, α₀, δ₀)
    T = length(y)

    B₁ = inv(inv(B₀) + X'X)
    B₁ = 0.5(B₁ + B₁')
    β₁ = B₁ * (B₀ \ β₀ + X'y)
    δ₁ = δ₀
    δ₁ += 0.5 * (y'y + β₀' / B₀ * β₀ - β₁' / B₁ * β₁)

    σ² = rand(InverseGamma(α₀ + 0.5T, δ₁))
    β = rand(MvNormal(β₁, σ² * B₁))

    return β, σ²
end

"""
It is a first derivative of the log posterior density of ηψ
===
"""
function dlogηψ_dηψ(ηψ; ψ, ψ0)

    if ηψ < 0
        return -Inf
    end

    dP = size(ψ)[1]
    p = size(ψ)[2] / dP
    allψ = [ψ0 ψ]

    deri = log(ηψ) + 1 - digamma(ηψ)
    deri *= p * (dP^2) + dP
    deri += sum(log.(allψ))
    deri -= sum(allψ) + 1

    return deri
end

"""
It is a second derivative of the log posterior density of ηψ
===
"""
function d2logηψ_dηψ2(ηψ; dP, p)

    deri = 1 / ηψ
    deri -= trigamma(ηψ)
    deri *= p * (dP^2) + dP

    return deri
end
