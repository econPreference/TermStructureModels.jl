"""
loglik_mea(yields, τₙ; κQ, kQ_infty, ϕ, σ²FF, Σₒ)
* This function generate a log likelihood of the measurement equation.
* Inputs: yields should exclude their initial observations. τₙ is maturities in "yields".
* Output: the measurement equation part of the log likelihood
"""
function loglik_mea(yields, τₙ; κQ, kQ_infty, ϕ, σ²FF, Σₒ)

    PCs, OCs, Wₚ, Wₒ, mean_PCs = PCA(yields, 0)
    bτ_ = bτ(τₙ[end]; κQ)
    Bₓ_ = Bₓ(bτ_, τₙ)
    T1X_ = T1X(Bₓ_, Wₚ)
    Bₚ_ = Bₚ(Bₓ_, T1X_, Wₒ)

    ΩPP = ϕ_σ²FF_2_ΩPP(; ϕ, σ²FF)
    aτ_ = aτ(τₙ[end], bτ_, τₙ, Wₚ; kQ_infty, ΩPP)
    Aₓ_ = Aₓ(aτ_, τₙ)
    T0P_ = T0P(T1X_, Aₓ_, Wₚ, mean_PCs)
    Aₚ_ = Aₚ(Aₓ_, Bₓ_, T0P_, Wₒ)

    T = size(OCs, 1)
    dist_mea = MvNormal(diagm(Σₒ))
    residuals = (OCs' - (Aₚ_ .+ Bₚ_ * PCs'))'

    logpdf_ = 0
    for t = 1:T
        logpdf_ += logpdf(dist_mea, residuals[t, :])
    end

    return logpdf_
end

"""
loglik_tran(PCs, macros; ϕ, σ²FF)
* It calculate log likelihood of the transition equation. 
* Input: All data should contains initial observations.
* Output: log likelihood of the transition equation.
"""
function loglik_tran(PCs, macros; ϕ, σ²FF)

    dP = length(σ²FF)
    p = Int(((size(ϕ, 2) - 1) / dP) - 1)

    yϕ, Xϕ = yϕ_Xϕ(PCs, macros, p)

    T = size(yϕ, 1)
    logpdf_ = Vector{Float64}(undef, dP)
    for i in 1:dP
        logpdf_[i] = logpdf(MvNormal(Xϕ * (ϕ[i, :]), σ²FF[i] * I(T)), yϕ[:, i])
    end

    return sum(logpdf_)
end

"""
yϕ_Xϕ(PCs, macros, p)
* This function generate the dependent variable and the corresponding regressors in the orthogonal transition equation.
* Input: PCs and macros data should contain initial observations. p is the transition equation lag.
* Output(4): yϕ, Xϕ = [ones(T - p) Xϕ_lag Xϕ_contemporaneous], [ones(T - p) Xϕ_lag], Xϕ_contemporaneous
    - yϕ and Xϕ is a full matrix. For i'th equation, the dependent variable is yϕ[:,i] and the regressor is Xϕ. 
    - Xϕ is same to all orthogonalized transtion equations. The orthogonalized equations are different in terms of contemporaneous regressors. Therefore, the corresponding regressors in Xϕ should be excluded. The form of parameter ϕ do that task by setting the coefficients of the excluded regressors to zeros.
    - For last dP by dP block in ϕ, the diagonals and the upper diagonal elements should be zero. 
"""
function yϕ_Xϕ(PCs, macros, p)

    data = [PCs macros]
    T = size(data, 1) # length including initial observations
    dP = size(data, 2)

    yϕ = data[(p+1):T, :]

    Xϕ_lag = Matrix{Float64}(undef, T - p, dP * p)
    Xϕ_contem = Matrix{Float64}(undef, T - p, dP)
    for t in (p+1):T
        Xϕ_lag[t-p, :] = vec(data[(t-1):-1:(t-p), :]')'
        Xϕ_contem[t-p, :] = -data[t, :]
    end
    Xϕ = [ones(T - p) Xϕ_lag Xϕ_contem]

    return yϕ, Xϕ, [ones(T - p) Xϕ_lag], Xϕ_contem
end

"""
LDL(X)
* This function generate a matrix decomposition, called LDLt. X = L*D*L', where L is a lower triangular matrix and D is a diagonal. How to conduct it can be found at [Wikipedia](https://en.wikipedia.org/wiki/Cholesky_decomposition#LDL_decomposition).
* Input: Decomposed Object, X
* Output(2): L, D
    - Decomposed result is X = L*D*L'
"""
function LDL(X)
    C = cholesky(X).L
    S = Diagonal(diagm(diag(C)))
    L = C / S
    D = S^2

    return L, D
end

"""
ϕ_σ²FF_2_ΩPP(; ϕ, σ²FF)
* It construct ΩPP from statistical parameters.
* Output: ΩPP
"""
function ϕ_σ²FF_2_ΩPP(; ϕ, σ²FF)

    dQ = dimQ()
    ~, C = ϕ_2_ϕ₀_C(; ϕ)

    CQQ = C[1:dQ, 1:dQ]
    return (CQQ \ diagm(σ²FF[1:dQ])) / CQQ'

end

"""
ϕ_2_ϕ₀_C(; ϕ)
* It divide ϕ into the lagged regressor part and the contemporaneous regerssor part.
* Output(3): ϕ0, C = C0 + I, C0
    * ϕ0: coefficients for the lagged regressors
    * C: coefficients for the dependent variables when all contemporaneous variables are in the LHS of the orthogonalized equations. Therefore, the diagonals of C is ones. 
        - Note that since the contemporaneous variables get negative signs when they are at the RHS, the signs of C do not change whether they are at the RHS or LHS. 
"""
function ϕ_2_ϕ₀_C(; ϕ)

    dP = size(ϕ, 1)
    ϕ0 = ϕ[:, 1:(end-dP)]
    C0 = ϕ[:, (end-dP+1):end]
    C = C0 + I(dP)

    return ϕ0, C, C0
end

"""
ϕ\\_σ²FF\\_2\\_ΩFF(; ϕ, σ²FF)
* It construct ΩFF from statistical parameters.
* Output: ΩFF
"""
function ϕ_σ²FF_2_ΩFF(; ϕ, σ²FF)

    C = ϕ_2_ϕ₀_C(; ϕ)[2]
    return (C \ diagm(σ²FF)) / C'
end

"""
isstationary(GₚFF)
* It checks whether a reduced VAR matrix has unit roots. If there is at least one unit root, return is false.
* GₚFF should not include intercepts. Also, GₚFF is dP by dP*p matrix that the coefficient at lag 1 comes first, and the lag p slope matrix comes last. 
* Output: boolean
"""
function isstationary(GₚFF)
    dP = size(GₚFF, 1)
    p = Int(size(GₚFF, 2) / dP)

    G = [GₚFF
        I(dP * (p - 1)) zeros(dP * (p - 1), dP)]

    return maximum(abs.(eigen(G).values)) < 1 || maximum(abs.(eigen(G).values)) ≈ 1
end

""" 
stationary\\_θ(saved_θ)
* It filters out posterior samples that implies an unit root VAR system. Only stationary posterior samples remain.
* Output(2): stationary samples, acceptance rate(%)
    - The second output indicates how many posterior samples remain.
    - "κQ", "kQ_infty", "ϕ", "σ²FF" , "ηψ", "ψ", "ψ0","Σₒ", "γ" ∈ Output[i]
"""
function stationary_θ(saved_θ)

    iteration = length(saved_θ)
    stationary_saved_θ = Vector{Parameter}(undef, 0)
    prog = Progress(iteration; dt=5, desc="Filtering...")
    Threads.@threads for iter in 1:iteration

        κQ = saved_θ[:κQ][iter]
        kQ_infty = saved_θ[:kQ_infty][iter]
        ϕ = saved_θ[:ϕ][iter]
        σ²FF = saved_θ[:σ²FF][iter]
        Σₒ = saved_θ[:Σₒ][iter]
        γ = saved_θ[:γ][iter]

        ϕ0, C = ϕ_2_ϕ₀_C(; ϕ)
        ϕ0 = C \ ϕ0
        GₚFF = ϕ0[:, 2:end]

        if isstationary(GₚFF)
            push!(stationary_saved_θ, Parameter(κQ=κQ, kQ_infty=kQ_infty, ϕ=ϕ, σ²FF=σ²FF, Σₒ=Σₒ, γ=γ))
        end
        next!(prog)
    end
    finish!(prog)

    return stationary_saved_θ, 100length(stationary_saved_θ) / iteration
end

"""
reducedform(saved_θ, yields, macros, τₙ)
* It generate posterior samples of the statistical parameters in struct "ReducedForm". 
* Input: "saved_θ" comes from function "posterior_sampler".
    - Input data includes initial observations.
* Output: the market prices of risks does not have initial observations
"""
function reducedform(saved_θ, yields, macros, τₙ)

    dQ = dimQ()
    dP = size(saved_θ[:ϕ][1], 1)
    p = Int((size(saved_θ[:ϕ][1], 2) - 1) / dP - 1)
    PCs, ~, Wₚ, ~, mean_PCs = PCA(yields, p)
    factors = [PCs macros]

    iteration = length(saved_θ)
    reduced_θ = Vector{ReducedForm}(undef, iteration)
    prog = Progress(iteration; dt=5, desc="Moving to the reduced form...")
    Threads.@threads for iter in 1:iteration

        κQ = saved_θ[:κQ][iter]
        kQ_infty = saved_θ[:kQ_infty][iter]
        ϕ = saved_θ[:ϕ][iter]
        σ²FF = saved_θ[:σ²FF][iter]
        Σₒ = saved_θ[:Σₒ][iter]

        ϕ0, C = ϕ_2_ϕ₀_C(; ϕ)
        ϕ0 = C \ ϕ0
        KₚF = ϕ0[:, 1]
        GₚFF = ϕ0[:, 2:end]
        ΩFF = (C \ diagm(σ²FF)) / C' |> Symmetric


        bτ_ = bτ(τₙ[end]; κQ)
        Bₓ_ = Bₓ(bτ_, τₙ)
        T1X_ = T1X(Bₓ_, Wₚ)
        aτ_ = aτ(τₙ[end], bτ_, τₙ, Wₚ; kQ_infty, ΩPP=ΩFF[1:dQ, 1:dQ])
        Aₓ_ = Aₓ(aτ_, τₙ)
        T0P_ = T0P(T1X_, Aₓ_, Wₚ, mean_PCs)

        KₓQ = zeros(dQ)
        KₓQ[1] = kQ_infty
        KₚQ = T1X_ * (KₓQ + (GQ_XX(; κQ) - I(dQ)) * T0P_)
        GQPF = similar(GₚFF[1:dQ, :]) |> (x -> x .= 0)
        GQPF[:, 1:dQ] = T1X_ * GQ_XX(; κQ) / T1X_
        λP = KₚF[1:dQ] - KₚQ
        ΛPF = GₚFF[1:dQ, :] - GQPF

        mpr = Matrix{Float64}(undef, size(factors, 1) - p, dP)
        for t in p+1:size(factors, 1)
            Ft = factors'[:, t:-1:t-p+1] |> vec
            mpr[t-p, :] = cholesky(ΩFF).L \ [λP + ΛPF * Ft; zeros(dP - dQ)]
        end
        reduced_θ[iter] = ReducedForm(κQ=κQ, kQ_infty=kQ_infty, KₚF=KₚF, GₚFF=GₚFF, ΩFF=ΩFF, Σₒ=Σₒ, λP=λP, ΛPF=ΛPF, mpr=mpr)

        next!(prog)
    end
    finish!(prog)

    return reduced_θ
end

function calibration_μϕ_const(μkQ_infty, σkQ_infty, τ, yields, τₙ, p; μϕ_const_PCs=[], medium_τ=12 * [2, 2.5, 3, 3.5, 4, 4.5, 5], iteration=1000)

    dQ = dimQ()
    PCs, ~, Wₚ, ~, mean_PCs = PCA(yields, p)
    ΩPP = diagm([AR_res_var(PCs[:, i], p)[1] for i in 1:dQ])
    if isempty(μϕ_const_PCs)
        μϕ_const_PCs = zeros(dQ)
    end

    prior_TP = Vector{Float64}(undef, iteration)
    prior_λₚ = Matrix{Float64}(undef, iteration, dQ)
    for i in 1:iteration
        κQ = prior_κQ(medium_τ) |> rand
        kQ_infty = Normal(μkQ_infty, σkQ_infty) |> rand

        bτ_ = bτ(τₙ[end]; κQ)
        Bₓ_ = Bₓ(bτ_, τₙ)
        T1X_ = T1X(Bₓ_, Wₚ)

        aτ_ = aτ(τₙ[end], bτ_, τₙ, Wₚ; kQ_infty, ΩPP)
        Aₓ_ = Aₓ(aτ_, τₙ)
        T0P_ = T0P(T1X_, Aₓ_, Wₚ, mean_PCs)

        # Jensen's Ineqaulity term
        jensen = 0
        for i = 1:(τ-1)
            jensen += jensens_inequality(i + 1, bτ_, T1X_; ΩPP)
        end
        jensen /= -τ

        # Constant term
        KₓQ = zeros(dQ)
        KₓQ[1] = kQ_infty
        KₚQ = T1X_ * (KₓQ + (GQ_XX(; κQ) - I(dQ)) * T0P_)
        λₚ = μϕ_const_PCs - KₚQ
        prior_λₚ[i, :] = μϕ_const_PCs - KₚQ

        prior_TP[i] = sum(bτ_[:, 1:(τ-1)], dims=2)' * (T1X_ \ λₚ) |> x -> (-x[1] / τ) + jensen
    end

    return prior_TP, prior_λₚ

end

function prior_const_TP(tuned, τ, yields, τₙ, ρ; medium_τ=12 * [2, 2.5, 3, 3.5, 4, 4.5, 5], iteration=100)

    (; p, q, ν0, Ω0, μkQ_infty, σkQ_infty, μϕ_const, fix_const_PC1) = tuned
    ~, ~, Wₚ, ~, mean_PCs = PCA(yields, p)
    dP = length(Ω0)
    dQ = dimQ()

    prior_σ²FF_ = prior_σ²FF(; ν0, Ω0)
    prior_C_ = prior_C(; Ω0)
    prior_κQ_ = prior_κQ(medium_τ)
    prior_ϕ0_ = prior_ϕ0(μϕ_const, ρ, prior_κQ_, τₙ, Wₚ; ψ0=ones(dP), ψ=ones(dP, dP * p), q, ν0, Ω0, fix_const_PC1)
    kQ_infty_dist = Normal(μkQ_infty, σkQ_infty)

    prior_TP = Vector{Float64}(undef, iteration)
    for iter in 1:iteration

        σ²FF = rand.(MersenneTwister(1 + iter), prior_σ²FF_)
        C = rand.(MersenneTwister(2 + iter), prior_C_)
        for i in 2:dP, j in 1:(i-1)
            C[i, j] = Normal(0, sqrt(σ²FF[i] * var(prior_C_[i, j]))) |> x -> rand(MersenneTwister(3 + iter * i * j), x)
        end
        ΩFF = (C \ diagm(σ²FF)) / C' |> Symmetric
        ϕ0 = rand.(MersenneTwister(4 + iter), [Normal(mean(prior_ϕ0_[i, j]), sqrt(σ²FF[i] * var(prior_ϕ0_[i, j]))) for i in 1:dP, j in 1:(dP*p+1)])

        ϕ0 = C \ ϕ0
        KₚF = ϕ0[:, 1]

        κQ = rand(MersenneTwister(5 + iter), prior_κQ_)
        bτ_ = bτ(τₙ[end]; κQ)
        Bₓ_ = Bₓ(bτ_, τₙ)
        T1X_ = T1X(Bₓ_, Wₚ)

        kQ_infty = rand(MersenneTwister(6 + iter), kQ_infty_dist)
        aτ_ = aτ(τₙ[end], bτ_, τₙ, Wₚ; kQ_infty, ΩPP=ΩFF[1:dQ, 1:dQ])
        Aₓ_ = Aₓ(aτ_, τₙ)
        T0P_ = T0P(T1X_, Aₓ_, Wₚ, mean_PCs)

        # Jensen's Ineqaulity term
        jensen = 0
        for i = 1:(τ-1)
            jensen += jensens_inequality(i + 1, bτ_, T1X_; ΩPP=ΩFF[1:dQ, 1:dQ])
        end
        jensen /= -τ

        # Constant term
        KₓQ = zeros(dQ)
        KₓQ[1] = kQ_infty
        KₚQ = T1X_ * (KₓQ + (GQ_XX(; κQ) - I(dQ)) * T0P_)
        λₚ = KₚF[1:dQ] - KₚQ

        prior_TP[iter] = sum(bτ_[:, 1:(τ-1)], dims=2)' * (T1X_ \ λₚ) |> x -> (-x[1] / τ) + jensen
    end

    return prior_TP

end
