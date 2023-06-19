
"""
tuning_hyperparameter(yields, macros, τₙ, ρ; gradient=false)
* It derives the hyperparameters that maximize the marginal likelhood. First, the generating set search algorithm detemines the search range that do not make a final solution as a corner solution. Second, the evolutionary algorithm and Nelder-Mead algorithm find the global optimum. Lastly, the LBFGS algorithm calibrate the global optimum. 
* Input: Data should contain initial observations.
    - ρ = Vector{Float64}(0 or ≈1, dP-dQ). Usually, 0 for growth macro variables and 1 (or 0.9) for level macro variables.
    - If gradient == true, the LBFGS method is applied at the last.
* Output: struct Hyperparameter
"""
function tuning_hyperparameter(yields, macros, τₙ, ρ; populationsize=30, maxiter=0, medium_τ=12 * [2, 2.5, 3, 3.5, 4, 4.5, 5], lag=1, upper_q=[1 1; 1 1; 10 10; 100 100], μkQ_infty=0, σkQ_infty=1, mSR_const=Inf, mSR_ftn=x -> mean(x), initial=[], upper_ν0=[], μϕ_const=[], fix_const_PC1=false, mSR_param=[])

    if isempty(upper_ν0) == true
        upper_ν0 = size(yields, 1)
    end

    dQ = dimQ()
    dP = dQ + size(macros, 2)
    PCs, ~, Wₚ, ~, mean_PCs = PCA(yields, lag)
    lx = [0.0; 0.0; 1; 0.0; 0.0; 0.0; 1; 0.0; 1]
    ux = 0.0 .+ [vec(upper_q); upper_ν0 - (dP + 1)]
    AR_re_var_vec = [AR_res_var([PCs macros][:, i], lag)[1] for i in 1:dP]
    if isempty(μϕ_const) == true
        μϕ_const = zeros(dP)
    end

    ## for calculating mSR###
    factors = [PCs macros]
    T = size(factors, 1)
    yϕ, Xϕ = yϕ_Xϕ(PCs, macros, lag)
    Xϕ0 = Xϕ[:, (end-dP+1):end]
    prior_κQ_ = prior_κQ(medium_τ)

    if mSR_param |> isempty

        σ²FF = AR_re_var_vec
        C = I(dP)
        CQQ = C[1:dQ, 1:dQ]
        ΩFF = diagm(σ²FF)
        κQ = mean(prior_κQ_)
        kQ_infty = Normal(μkQ_infty, σkQ_infty) |> mean

        bτ_ = bτ(τₙ[end]; κQ)
        Bₓ_ = Bₓ(bτ_, τₙ)
        T1X_ = T1X(Bₓ_, Wₚ)
        aτ_ = aτ(τₙ[end], bτ_, τₙ, Wₚ; kQ_infty, ΩPP=ΩFF[1:dQ, 1:dQ])
        Aₓ_ = Aₓ(aτ_, τₙ)
        T0P_ = T0P(T1X_, Aₓ_, Wₚ, mean_PCs)

        KₓQ = zeros(dQ)
        KₓQ[1] = kQ_infty
        KₚQ = T1X_ * (KₓQ + (GQ_XX(; κQ) - I(dQ)) * T0P_)
        GQPP = T1X_ * GQ_XX(; κQ) / T1X_

        CQQ_KₚQ = CQQ * KₚQ
        CQQ_GQPP = CQQ * GQPP
    else
        (; σ²FF, C, κQ, kQ_infty) = mSR_param
        CQQ = C[1:dQ, 1:dQ]
        ΩPP = (CQQ \ diagm(σ²FF[1:dQ])) / CQQ'

        bτ_ = bτ(τₙ[end]; κQ)
        Bₓ_ = Bₓ(bτ_, τₙ)
        T1X_ = T1X(Bₓ_, Wₚ)
        aτ_ = aτ(τₙ[end], bτ_, τₙ, Wₚ; kQ_infty, ΩPP)
        Aₓ_ = Aₓ(aτ_, τₙ)
        T0P_ = T0P(T1X_, Aₓ_, Wₚ, mean_PCs)

        KₓQ = zeros(dQ)
        KₓQ[1] = kQ_infty
        KₚQ = T1X_ * (KₓQ + (GQ_XX(; κQ) - I(dQ)) * T0P_)
        GQPP = T1X_ * GQ_XX(; κQ) / T1X_

        CQQ_KₚQ = CQQ * KₚQ
        CQQ_GQPP = CQQ * GQPP
    end

    function maximum_SR_inner(Hyperparameter_::Hyperparameter)

        (; p, q, ν0, Ω0, μϕ_const, fix_const_PC1) = Hyperparameter_

        prior_ϕ0_ = prior_ϕ0(μϕ_const, ρ, prior_κQ_, τₙ, Wₚ; ψ0=ones(dP), ψ=ones(dP, dP * p), q, ν0, Ω0, fix_const_PC1)

        Λ_i_mean = Matrix{Float64}(undef, 1 + dP * p, dQ)
        ϕ_i_var = Array{Float64}(undef, 1 + dP * p, 1 + dP * p, dQ)
        for i in 1:dQ
            mᵢ = mean.(prior_ϕ0_[i, 1:(1+p*dP)])
            Vᵢ = var.(prior_ϕ0_[i, 1:(1+p*dP)])
            Λ_i_mean[:, i], ϕ_i_var[:, :, i] = Normal_Normal_in_NIG(-Xϕ0 * C[i, :], Xϕ[:, 1:(end-dP)], mᵢ, diagm(Vᵢ), σ²FF[i])
            Λ_i_mean[1, i] -= CQQ_KₚQ[i]
            Λ_i_mean[2:dQ+1, i] .-= CQQ_GQPP[i, :]
        end

        mSR = Vector{Float64}(undef, T - p)
        for t in p+1:T
            Ft = factors'[:, t:-1:t-p+1] |> vec |> x -> [1; x]

            λ_dist = Vector{Normal}(undef, dQ)
            for i in 1:dQ
                λ_dist[i] = Normal(Λ_i_mean[:, i]'Ft, sqrt(Ft' * ϕ_i_var[:, :, i] * Ft)) / sqrt(σ²FF[i])
            end

            mSR[t-p] = var.(λ_dist)' * (((mean.(λ_dist)) .^ 2) .+ 1) # mean of Generalized chi-squared distribution
        end

        return mSR
    end
    ###

    function negative_log_marginal(input)

        # parameters
        q = [input[1] input[5]
            input[2] input[6]
            input[3] input[7]
            input[4] input[8]]
        q[2, :] = q[1, :] .* q[2, :]
        ν0 = input[9] + dP + 1
        Ω0 = AR_re_var_vec * input[9]

        if minimum([vec(q); ν0 - dP + 1; Ω0]) <= 0
            return Inf
        end

        tuned = Hyperparameter(p=lag, q=q, ν0=ν0, Ω0=Ω0, μkQ_infty=μkQ_infty, σkQ_infty=σkQ_infty, μϕ_const=μϕ_const, fix_const_PC1=fix_const_PC1)
        return -log_marginal(PCs, macros, ρ, tuned, τₙ, Wₚ; medium_τ)

        # Although the input data should contains initial observations, the argument of the marginal likelihood should be the same across the candidate models. Therefore, we should align the length of the dependent variable across the models.

    end
    function constraint(input)

        # parameters
        q = [input[1] input[5]
            input[2] input[6]
            input[3] input[7]
            input[4] input[8]]
        q[2, :] = q[1, :] .* q[2, :]
        ν0 = input[9] + dP + 1
        Ω0 = AR_re_var_vec * input[9]

        if minimum([vec(q); ν0 - dP + 1; Ω0]) <= 0
            return Inf
        end

        tuned = Hyperparameter(p=lag, q=q, ν0=ν0, Ω0=Ω0, μkQ_infty=μkQ_infty, σkQ_infty=σkQ_infty, μϕ_const=μϕ_const, fix_const_PC1=fix_const_PC1)
        if isinf(mSR_const)
            return 0.0
        else
            return mSR_ftn(maximum_SR_inner(tuned))
        end

    end

    algo = WOA(; N=populationsize, options=Options(debug=true, iterations=maxiter))

    bounds = boxconstraints(lb=lx, ub=ux)
    function obj(input)
        fx, gx = negative_log_marginal(input), constraint(input)
        fx, [gx - mSR_const], zeros(1)
    end
    if !isempty(initial)
        set_user_solutions!(algo, initial, obj)
    end
    opt = Metaheuristics.optimize(obj, bounds, algo)

    q = [minimizer(opt)[1] minimizer(opt)[5]
        minimizer(opt)[2] minimizer(opt)[6]
        minimizer(opt)[3] minimizer(opt)[7]
        minimizer(opt)[4] minimizer(opt)[8]]
    q[2, :] = q[1, :] .* q[2, :]
    ν0 = minimizer(opt)[9] + dP + 1
    Ω0 = AR_re_var_vec * minimizer(opt)[9]

    return Hyperparameter(p=lag, q=q, ν0=ν0, Ω0=Ω0, μkQ_infty=μkQ_infty, σkQ_infty=σkQ_infty, μϕ_const=μϕ_const, fix_const_PC1=fix_const_PC1), opt

end

"""
tuning_hyperparameter_mSR(yields, macros, τₙ, ρ; medium_τ=12 * [1.5, 2, 2.5, 3, 3.5], maxstep=10_000, mSR_scale=1.0, mSR_mean=1.0, upper_lag=9, upper_q1=1, upper_q45=100, μkQ_infty=1)
"""
function tuning_hyperparameter_MOEA(yields, macros, τₙ, ρ; populationsize=100, maxiter=0, medium_τ=12 * [2, 2.5, 3, 3.5, 4, 4.5, 5], lag=1, upper_q=[1 1; 1 1; 10 10; 100 100], μkQ_infty=0, σkQ_infty=1, μϕ_const=[], upper_ν0=[], fix_const_PC1=false, mSR_param=[], mSR_ftn=x -> mean(x))

    if isempty(upper_ν0) == true
        upper_ν0 = size(yields, 1)
    end

    dQ = dimQ()
    dP = dQ + size(macros, 2)
    PCs, ~, Wₚ, ~, mean_PCs = PCA(yields, lag)
    lx = [0.0; 0.0; 1; 0.0; 0.0; 0.0; 1; 0.0; 1]
    ux = 0.0 .+ [vec(upper_q); upper_ν0 - (dP + 1)]
    AR_re_var_vec = [AR_res_var([PCs macros][:, i], lag)[1] for i in 1:dP]
    if isempty(μϕ_const) == true
        μϕ_const = zeros(dP)
    end

    ## for calculating mSR###
    factors = [PCs macros]
    T = size(factors, 1)
    yϕ, Xϕ = yϕ_Xϕ(PCs, macros, lag)
    Xϕ0 = Xϕ[:, (end-dP+1):end]
    prior_κQ_ = prior_κQ(medium_τ)

    if mSR_param |> isempty

        σ²FF = AR_re_var_vec
        C = I(dP)
        CQQ = C[1:dQ, 1:dQ]
        ΩFF = diagm(σ²FF)
        κQ = mean(prior_κQ_)
        kQ_infty = Normal(μkQ_infty, σkQ_infty) |> mean

        bτ_ = bτ(τₙ[end]; κQ)
        Bₓ_ = Bₓ(bτ_, τₙ)
        T1X_ = T1X(Bₓ_, Wₚ)
        aτ_ = aτ(τₙ[end], bτ_, τₙ, Wₚ; kQ_infty, ΩPP=ΩFF[1:dQ, 1:dQ])
        Aₓ_ = Aₓ(aτ_, τₙ)
        T0P_ = T0P(T1X_, Aₓ_, Wₚ, mean_PCs)

        KₓQ = zeros(dQ)
        KₓQ[1] = kQ_infty
        KₚQ = T1X_ * (KₓQ + (GQ_XX(; κQ) - I(dQ)) * T0P_)
        GQPP = T1X_ * GQ_XX(; κQ) / T1X_

        CQQ_KₚQ = CQQ * KₚQ
        CQQ_GQPP = CQQ * GQPP
    else
        (; σ²FF, C, κQ, kQ_infty) = mSR_param
        CQQ = C[1:dQ, 1:dQ]
        ΩPP = (CQQ \ diagm(σ²FF[1:dQ])) / CQQ'

        bτ_ = bτ(τₙ[end]; κQ)
        Bₓ_ = Bₓ(bτ_, τₙ)
        T1X_ = T1X(Bₓ_, Wₚ)
        aτ_ = aτ(τₙ[end], bτ_, τₙ, Wₚ; kQ_infty, ΩPP)
        Aₓ_ = Aₓ(aτ_, τₙ)
        T0P_ = T0P(T1X_, Aₓ_, Wₚ, mean_PCs)

        KₓQ = zeros(dQ)
        KₓQ[1] = kQ_infty
        KₚQ = T1X_ * (KₓQ + (GQ_XX(; κQ) - I(dQ)) * T0P_)
        GQPP = T1X_ * GQ_XX(; κQ) / T1X_

        CQQ_KₚQ = CQQ * KₚQ
        CQQ_GQPP = CQQ * GQPP
    end

    function maximum_SR_inner(Hyperparameter_::Hyperparameter)

        (; p, q, ν0, Ω0, μϕ_const, fix_const_PC1) = Hyperparameter_

        prior_ϕ0_ = prior_ϕ0(μϕ_const, ρ, prior_κQ_, τₙ, Wₚ; ψ0=ones(dP), ψ=ones(dP, dP * p), q, ν0, Ω0, fix_const_PC1)

        Λ_i_mean = Matrix{Float64}(undef, 1 + dP * p, dQ)
        ϕ_i_var = Array{Float64}(undef, 1 + dP * p, 1 + dP * p, dQ)
        for i in 1:dQ
            mᵢ = mean.(prior_ϕ0_[i, 1:(1+p*dP)])
            Vᵢ = var.(prior_ϕ0_[i, 1:(1+p*dP)])
            Λ_i_mean[:, i], ϕ_i_var[:, :, i] = Normal_Normal_in_NIG(-Xϕ0 * C[i, :], Xϕ[:, 1:(end-dP)], mᵢ, diagm(Vᵢ), σ²FF[i])
            Λ_i_mean[1, i] -= CQQ_KₚQ[i]
            Λ_i_mean[2:dQ+1, i] .-= CQQ_GQPP[i, :]
        end

        mSR = Vector{Float64}(undef, T - p)
        for t in p+1:T
            Ft = factors'[:, t:-1:t-p+1] |> vec |> x -> [1; x]

            λ_dist = Vector{Normal}(undef, dQ)
            for i in 1:dQ
                λ_dist[i] = Normal(Λ_i_mean[:, i]'Ft, sqrt(Ft' * ϕ_i_var[:, :, i] * Ft)) / sqrt(σ²FF[i])
            end

            mSR[t-p] = var.(λ_dist)' * (((mean.(λ_dist)) .^ 2) .+ 1) # mean of Generalized chi-squared distribution
        end

        return mSR
    end
    ###


    function negative_log_marginal(input)

        # parameters
        q = [input[1] input[5]
            input[2] input[6]
            input[3] input[7]
            input[4] input[8]]
        q[2, :] = q[1, :] .* q[2, :]
        ν0 = input[9] + dP + 1
        Ω0 = AR_re_var_vec * input[9]

        if minimum([vec(q); ν0 - dP + 1; Ω0]) <= 0
            return [Inf, Inf]
        end

        tuned = Hyperparameter(p=lag, q=q, ν0=ν0, Ω0=Ω0, μkQ_infty=μkQ_infty, σkQ_infty=σkQ_infty, μϕ_const=μϕ_const, fix_const_PC1=fix_const_PC1)
        return [-log_marginal(PCs, macros, ρ, tuned, τₙ, Wₚ; medium_τ), mSR_ftn(maximum_SR_inner(tuned))]
        # Although the input data should contains initial observations, the argument of the marginal likelihood should be the same across the candidate models. Therefore, we should align the length of the dependent variable across the models.

    end

    bounds = boxconstraints(lb=lx, ub=ux)
    function obj(input)
        return negative_log_marginal(input), zeros(1), zeros(1)
    end
    opt = Metaheuristics.optimize(obj, bounds, NSGA3(; N=populationsize, options=Options(; verbose=true, iterations=maxiter)))

    pf = pareto_front(opt)
    pf_input = Vector{Hyperparameter}(undef, size(pf, 1))
    for i in eachindex(pf_input)
        input = opt.population[i].x
        q = [input[1] input[5]
            input[2] input[6]
            input[3] input[7]
            input[4] input[8]]
        q[2, :] = q[1, :] .* q[2, :]
        ν0 = input[9] + dP + 1
        Ω0 = AR_re_var_vec * input[9]

        pf_input[i] = Hyperparameter(p=lag, q=q, ν0=ν0, Ω0=Ω0, μkQ_infty=μkQ_infty, σkQ_infty=σkQ_infty, μϕ_const=μϕ_const, fix_const_PC1=fix_const_PC1)
    end

    return [-pf[:, 1], pf[:, 2]], pf_input, opt

end

"""
AR_res_var(TS::Vector, p)
* It derives an MLE error variance estimate of an AR(p) model
* Input: univariate time series TS and the lag p
* output: residual variance estimate
"""
function AR_res_var(TS::Vector, p)
    Y = TS[(p+1):end]
    T = length(Y)
    X = ones(T)
    for i in 1:p
        X = hcat(X, TS[p+1-i:end-i])
    end

    β = (X'X) \ (X'Y)
    return var(Y - X * β), β
end

"""
posterior_sampler(yields, macros, τₙ, ρ, iteration, Hyperparameter_; sparsity=false, medium_τ=12 * [1.5, 2, 2.5, 3, 3.5])
* This is a posterior distribution sampler. It needs data and hyperparameters. 
* Input: Data should include initial observations. τₙ is a vector that contains observed maturities.
    - ρ = Vector{Float64}(0 or ≈1, dP-dQ). Usually, 0 for growth macro variables and 1 (or 0.9) for level macro variables. 
    - iteration: # of posterior samples
* Output(3): Vector{Parameter}(posterior, iteration), acceptPr_C_σ²FF, acceptPr_ηψ 
"""
function posterior_sampler(yields, macros, τₙ, ρ, iteration, Hyperparameter_::Hyperparameter; sparsity=false, medium_τ=12 * [2, 2.5, 3, 3.5, 4, 4.5, 5], init_param=[])

    (; p, q, ν0, Ω0, μkQ_infty, σkQ_infty, μϕ_const, fix_const_PC1) = Hyperparameter_
    N = size(yields, 2) # of maturities
    dQ = dimQ()
    dP = dQ + size(macros, 2)
    Wₚ = PCA(yields, p)[3]
    prior_κQ_ = prior_κQ(medium_τ)
    γ_bar = prior_γ(yields[(p+1):end, :])

    if typeof(init_param) == Parameter
        (; κQ, kQ_infty, ϕ, σ²FF, ηψ, ψ, ψ0, Σₒ, γ) = init_param
    else
        ## initial parameters ##
        κQ = 0.0609
        kQ_infty = 0.0
        ϕ = [zeros(dP) diagm([0.9ones(dQ); ρ]) zeros(dP, dP * (p - 1)) zeros(dP, dP)] # The last dP by dP block matrix in ϕ should always be a lower triangular matrix whose diagonals are also always zero.
        bτ_ = bτ(τₙ[end]; κQ)
        Bₓ_ = Bₓ(bτ_, τₙ)
        T1X_ = T1X(Bₓ_, Wₚ)
        ϕ[1:dQ, 2:(dQ+1)] = T1X_ * GQ_XX(; κQ) / T1X_
        σ²FF = [Ω0[i] / (ν0 + i - dP) for i in eachindex(Ω0)]
        ηψ = 1
        ψ = ones(dP, dP * p)
        ψ0 = ones(dP)
        Σₒ = 1 ./ fill(γ_bar, N - dQ)
        γ = 1 ./ fill(γ_bar, N - dQ)
        ########################
    end

    isaccept_C_σ²FF = zeros(dQ)
    isaccept_ηψ = 0
    saved_θ = Vector{Parameter}(undef, iteration)
    @showprogress 1 "Sampling the posterior..." for iter in 1:iteration
        κQ = rand(post_κQ(yields[(p+1):end, :], prior_κQ_, τₙ; kQ_infty, ϕ, σ²FF, Σₒ))

        kQ_infty = rand(post_kQ_infty(μkQ_infty, σkQ_infty, yields[(p+1):end, :], τₙ; κQ, ϕ, σ²FF, Σₒ))

        # σ²FF, isaccept = post_σ²FF₁(yields, macros, τₙ, p; κQ, kQ_infty, ϕ, σ²FF, Σₒ, ν0, Ω0)
        # isaccept_C_σ²FF[1] += isaccept

        # ϕ, σ²FF, isaccept = post_C_σ²FF_dQ(yields, macros, τₙ, p; κQ, kQ_infty, ϕ, σ²FF, Σₒ, ν0, Ω0)
        # isaccept_C_σ²FF[2:end] += isaccept

        if sparsity == true
            ηψ, isaccept = post_ηψ(; ηψ, ψ, ψ0)
            isaccept_ηψ += isaccept
        end

        ϕ, σ²FF, isaccept = post_ϕ_σ²FF(yields, macros, μϕ_const, ρ, prior_κQ_, τₙ; ϕ, ψ, ψ0, σ²FF, q, ν0, Ω0, κQ, kQ_infty, Σₒ, fix_const_PC1)
        isaccept_C_σ²FF += isaccept

        if sparsity == true
            ψ0, ψ = post_ψ_ψ0(μϕ_const, ρ, prior_κQ_, τₙ, Wₚ; ϕ, ψ0, ψ, ηψ, q, σ²FF, ν0, Ω0, fix_const_PC1)
        end

        Σₒ = rand.(post_Σₒ(yields[(p+1):end, :], τₙ; κQ, kQ_infty, ΩPP=ϕ_σ²FF_2_ΩPP(; ϕ, σ²FF), γ))

        γ = rand.(post_γ(; γ_bar, Σₒ))


        saved_θ[iter] = Parameter(κQ=κQ, kQ_infty=kQ_infty, ϕ=ϕ, σ²FF=σ²FF, ηψ=ηψ, ψ=ψ, ψ0=ψ0, Σₒ=Σₒ, γ=γ)

    end

    return saved_θ, 100isaccept_C_σ²FF / iteration, 100isaccept_ηψ / iteration
end

"""
sparse_prec(saved_θ, yields, macros, τₙ)
* It conduct the glasso of Friedman, Hastie, and Tibshirani (2022) using the method of Hauzenberger, Huber and Onorante. 
* That is, the posterior samples of ΩFF is penalized with L1 norm to impose a sparsity on the precision.
* Input: "saved\\_θ" from function posterior_sampler, and the data should contain initial observations.
* Output(3): sparse_θ, trace_λ, trace_sparsity
    - sparse_θ: sparsified posterior samples
    - trace_λ: a vector that contains an optimal lasso parameters in iterations
    - trace_sparsity: a vector that contains degree of freedoms of inv(ΩFF) in iterations
"""
function sparse_prec(saved_θ, T; lower_penalty=1e-2, nlambda=100)

    R"library(qgraph)"
    ϕ = saved_θ[:ϕ][1]
    dP = size(ϕ, 1)

    iteration = length(saved_θ)
    sparse_θ = Vector{Parameter}(undef, iteration)
    trace_sparsity = Vector{Float64}(undef, iteration)
    @showprogress 1 "Imposing sparsity on precision..." for iter in 1:iteration

        κQ = saved_θ[:κQ][iter]
        kQ_infty = saved_θ[:kQ_infty][iter]
        ϕ = saved_θ[:ϕ][iter]
        σ²FF = saved_θ[:σ²FF][iter]
        ηψ = saved_θ[:ηψ][iter]
        ψ = saved_θ[:ψ][iter]
        ψ0 = saved_θ[:ψ0][iter]
        Σₒ = saved_θ[:Σₒ][iter]
        γ = saved_θ[:γ][iter]
        ϕ0, C = ϕ_2_ϕ₀_C(; ϕ)

        ΩFF_ = (C \ diagm(σ²FF)) / C'
        ΩFF_ = 0.5(ΩFF_ + ΩFF_')

        std_ = sqrt.(diag(ΩFF_))
        # inv_corr_ΩFF = diagm(1 ./ std_) * ΩFF_ * diagm(1 ./ std_) |> inv
        glasso_results = rcopy(rcall(:EBICglasso, ΩFF_, T, returnAllResults=true, var"lambda.min.ratio"=lower_penalty, nlambda=nlambda))#penalizeMatrix=abs.(inv_corr_ΩFF) .^ (-kappa / 2)))
        sparse_prec = glasso_results[:optwi]
        sparse_cov = diagm(std_) * inv(sparse_prec) * diagm(std_) |> Symmetric

        sparsity = sum(abs.(sparse_prec) .<= eps())
        trace_sparsity[iter] = sparsity
        inv_sparse_C, diagm_σ²FF = LDL(sparse_cov)
        ϕ = [ϕ0 (inv(inv_sparse_C) - I(dP))]
        σ²FF = diag(diagm_σ²FF)

        sparse_θ[iter] = Parameter(κQ=κQ, kQ_infty=kQ_infty, ϕ=ϕ, σ²FF=σ²FF, ηψ=ηψ, ψ=ψ, ψ0=ψ0, Σₒ=Σₒ, γ=γ)
    end

    return sparse_θ, trace_sparsity
end

function sparse_coef(saved_θ, yields, macros, τₙ; zeta=2, lambda=1)

    dP = size(saved_θ[:ϕ][1], 1)
    p = Int((size(saved_θ[:ϕ][1], 2) - 1) / dP - 1)
    PCs = PCA(yields, p)[1]
    factors = [PCs macros]
    T = size(factors, 1)
    X = Matrix{Float64}(undef, T - p, 1 + dP * p)
    for t = p+1:T
        X[t-p, :] = [vec(factors[t-1:-1:t-p, :]'); 1]
    end
    Z = kron(I(dP), X)

    iteration = length(saved_θ)
    sparse_θ = Vector{Parameter}(undef, iteration)
    trace_sparsity = Vector{Float64}(undef, iteration)
    trace_lik = Vector{Float64}(undef, iteration)
    @showprogress 1 "Imposing sparsity on coefs..." for iter in 1:iteration

        κQ = saved_θ[:κQ][iter]
        kQ_infty = saved_θ[:kQ_infty][iter]
        ϕ = saved_θ[:ϕ][iter]
        σ²FF = saved_θ[:σ²FF][iter]
        ηψ = saved_θ[:ηψ][iter]
        ψ = saved_θ[:ψ][iter]
        ψ0 = saved_θ[:ψ0][iter]
        Σₒ = saved_θ[:Σₒ][iter]
        γ = saved_θ[:γ][iter]

        ϕ0, C, C0 = ϕ_2_ϕ₀_C(; ϕ)
        ϕ0 = C \ ϕ0
        KₚF = ϕ0[:, 1]
        GₚFF = ϕ0[:, 2:end]
        coefs_hat = [GₚFF KₚF]' |> vec

        # ind_diag = I(dP)
        # ind_lag = ones(dP, dP)
        # for i in 2:p
        #     ind_diag = [ind_diag I(dP)]
        #     ind_lag = [ind_lag i * ones(dP, dP)]
        # end
        # ind_diag = [ind_diag zeros(dP)] |> x -> vec(x')
        # ind_lag = [ind_lag zeros(dP)] |> x -> vec(x')

        coefs = similar(coefs_hat)
        sparsity = 0
        for j in eachindex(coefs)
            j_lambda = lambda
            # if ind_diag[j] == 1
            #     j_lambda = lambda * (ind_lag[j] - 1)^2
            # elseif ind_lag[j] == 0
            #     j_lambda = 0
            # else
            #     j_lambda = lambda * ind_lag[j]^2
            # end

            coef_hat = coefs_hat[j]
            thres = j_lambda / (abs(coef_hat)^zeta)
            trunc_coef = max(abs(coef_hat) * (norm(Z[:, j])^2) - thres, 0)
            coefs[j] = sign(coef_hat) * trunc_coef / (norm(Z[:, j])^2)
            if trunc_coef == 0
                sparsity += 1
            end
        end
        reshape_coef = reshape(coefs, 1 + dP * p, dP)'
        KₚF = reshape_coef[:, end]
        GₚFF = reshape_coef[:, 1:end-1]
        ϕ0 = C * [KₚF GₚFF]
        ϕ = [ϕ0 C0]

        sparse_θ[iter] = Parameter(κQ=κQ, kQ_infty=kQ_infty, ϕ=ϕ, σ²FF=σ²FF, ηψ=ηψ, ψ=ψ, ψ0=ψ0, Σₒ=Σₒ, γ=γ)
        trace_sparsity[iter] = sparsity
        trace_lik[iter] = loglik_mea(yields[p+1:end, :], τₙ; κQ, kQ_infty, ϕ, σ²FF, Σₒ) + loglik_tran(PCs, macros; ϕ, σ²FF)
    end

    return sparse_θ, trace_sparsity, trace_lik
end

function sparse_prec_coef(saved_θ, yields, macros, τₙ; zeta=2, lower_penalty=1e-2, nlambda=100, lambda=1)
    R"library(qgraph)"

    dP = size(saved_θ[:ϕ][1], 1)
    p = Int((size(saved_θ[:ϕ][1], 2) - 1) / dP - 1)
    PCs = PCA(yields, p)[1]
    factors = [PCs macros]
    T = size(factors, 1)
    X = Matrix{Float64}(undef, T - p, 1 + dP * p)
    for t = p+1:T
        X[t-p, :] = [vec(factors[t-1:-1:t-p, :]'); 1]
    end
    Z = kron(I(dP), X)

    iteration = length(saved_θ)
    sparse_θ = Vector{Parameter}(undef, iteration)
    trace_sparsity_prec = Vector{Float64}(undef, iteration)
    trace_sparsity_coef = Vector{Float64}(undef, iteration)
    trace_lik = Vector{Float64}(undef, iteration)
    @showprogress 1 "Imposing sparsity on coefs..." for iter in 1:iteration

        κQ = saved_θ[:κQ][iter]
        kQ_infty = saved_θ[:kQ_infty][iter]
        ϕ = saved_θ[:ϕ][iter]
        σ²FF = saved_θ[:σ²FF][iter]
        ηψ = saved_θ[:ηψ][iter]
        ψ = saved_θ[:ψ][iter]
        ψ0 = saved_θ[:ψ0][iter]
        Σₒ = saved_θ[:Σₒ][iter]
        γ = saved_θ[:γ][iter]

        ϕ0, C = ϕ_2_ϕ₀_C(; ϕ)
        ΩFF_ = (C \ diagm(σ²FF)) / C'
        ΩFF_ = 0.5(ΩFF_ + ΩFF_')

        std_ = sqrt.(diag(ΩFF_))
        # inv_corr_ΩFF = diagm(1 ./ std_) * ΩFF_ * diagm(1 ./ std_) |> inv
        glasso_results = rcopy(rcall(:EBICglasso, ΩFF_, T - p, returnAllResults=true, var"lambda.min.ratio"=lower_penalty, nlambda=nlambda))#, penalizeMatrix=abs.(inv_corr_ΩFF) .^ (-kappa / 2)))
        sparse_prec = glasso_results[:optwi]
        sparse_cov = diagm(std_) * inv(sparse_prec) * diagm(std_) |> Symmetric

        sparsity_prec = sum(abs.(sparse_prec) .<= eps())
        trace_sparsity_prec[iter] = sparsity_prec
        inv_sparse_C, diagm_σ²FF = LDL(sparse_cov)
        C = inv(inv_sparse_C)
        C0 = C - I(dP)
        σ²FF = diag(diagm_σ²FF)

        ϕ0 = C \ ϕ0
        KₚF = ϕ0[:, 1]
        GₚFF = ϕ0[:, 2:end]
        coefs_hat = [GₚFF KₚF]' |> vec

        # ind_diag = I(dP)
        # ind_lag = ones(dP, dP)
        # for i in 2:p
        #     ind_diag = [ind_diag I(dP)]
        #     ind_lag = [ind_lag i * ones(dP, dP)]
        # end
        # ind_diag = [ind_diag zeros(dP)] |> x -> vec(x')
        # ind_lag = [ind_lag zeros(dP)] |> x -> vec(x')

        coefs = similar(coefs_hat)
        sparsity_coef = 0
        for j in eachindex(coefs)
            j_lambda = lambda
            # if ind_diag[j] == 1
            #     j_lambda = lambda * (ind_lag[j] - 1)^2
            # elseif ind_lag[j] == 0
            #     j_lambda = 0
            # else
            #     j_lambda = lambda * ind_lag[j]^2
            # end

            coef_hat = coefs_hat[j]
            thres = j_lambda / (abs(coef_hat)^zeta)
            trunc_coef = max(abs(coef_hat) * (norm(Z[:, j])^2) - thres, 0)
            coefs[j] = sign(coef_hat) * trunc_coef / (norm(Z[:, j])^2)
            if trunc_coef == 0
                sparsity_coef += 1
            end
        end
        reshape_coef = reshape(coefs, 1 + dP * p, dP)'
        KₚF = reshape_coef[:, end]
        GₚFF = reshape_coef[:, 1:end-1]
        ϕ0 = C * [KₚF GₚFF]
        trace_sparsity_coef[iter] = sparsity_coef
        ϕ = [ϕ0 C0]

        sparse_θ[iter] = Parameter(κQ=κQ, kQ_infty=kQ_infty, ϕ=ϕ, σ²FF=σ²FF, ηψ=ηψ, ψ=ψ, ψ0=ψ0, Σₒ=Σₒ, γ=γ)
        trace_lik[iter] = loglik_mea(yields[p+1:end, :], τₙ; κQ, kQ_infty, ϕ, σ²FF, Σₒ) + loglik_tran(PCs, macros; ϕ, σ²FF)
    end

    return sparse_θ, trace_sparsity_prec, trace_sparsity_coef, trace_lik
end

"""
generative(T, dP, τₙ, p; κQ, kQ_infty, KₚXF, GₚXFXF, ΩXFXF)
* This function generate a simulation data given parameters. 
    -Note that all parameters are the things in the latent factor state space. There is some differences in notations because it is hard to express mathcal letters in VScode. So, mathcal{F} in my paper is expressed in F in the VScode. And, "F" in my paper is expressed as XF.
* Input: p is a lag of transition VAR, τₙ is a set of observed maturities
* Output(3): yields, latents, macros
    - yields = Matrix{Float64}(obs,T,length(τₙ))
    - latents = Matrix{Float64}(obs,T,dimQ())
    - macros = Matrix{Float64}(obs,T,dP - dimQ())
"""
function generative(T, dP, τₙ, p; κQ, kQ_infty, KₚXF, GₚXFXF, ΩXFXF)
    N = length(τₙ) # of observed maturities
    dQ = dimQ() # of latent factors

    # Generating factors XF, where latents & macros ∈ XF
    XF = randn(p, dP)
    for horizon = 1:(round(Int, 1.5T))
        regressors = vec(XF[1:p, :]')
        samples = KₚXF + GₚXFXF * regressors + rand(MvNormal(zeros(dP), ΩXFXF))
        XF = vcat(samples', XF)
    end
    XF = reverse(XF, dims=1)
    XF = XF[end-T+1:end, :]

    # Generating yields
    bτ_ = bτ(τₙ[end]; κQ)
    Bₓ_ = Bₓ(bτ_, τₙ)

    ΩXX = ΩXFXF[1:dQ, 1:dQ]
    aτ_ = aτ(τₙ[end], bτ_; kQ_infty, ΩXX)
    Aₓ_ = Aₓ(aτ_, τₙ)

    yields = Matrix{Float64}(undef, T, N)
    for t = 1:T
        yields[t, :] = (Aₓ_ + Bₓ_ * XF[t, 1:dQ])' + rand(Normal(0, sqrt(0.01)), N)'
    end

    return yields, XF[:, 1:dQ], XF[:, (dQ+1):end]
end

"""
ineff_factor(saved_θ)
* It returns inefficiency factors of each parameter
* Input: posterior sample matrix from the Gibbs sampler
* Output: Vector{Float64}(inefficiency factors, # of parameters)
"""
function ineff_factor(saved_θ; fix_const_PC1=false)

    iteration = length(saved_θ)

    κQ = saved_θ[:κQ][1]
    kQ_infty = saved_θ[:kQ_infty][1]
    if fix_const_PC1
        ϕ = saved_θ[:ϕ][1] |> x -> vec(x)[2:end]
    else
        ϕ = saved_θ[:ϕ][1] |> x -> vec(x)
    end
    σ²FF = saved_θ[:σ²FF][1]
    ηψ = saved_θ[:ηψ][1]
    ψ = saved_θ[:ψ][1]
    ψ0 = saved_θ[:ψ0][1]
    Σₒ = saved_θ[:Σₒ][1]
    γ = saved_θ[:γ][1]

    initial_θ = [κQ; kQ_infty; ηψ; γ; Σₒ; σ²FF; ψ0; vec(ψ); ϕ]
    vec_saved_θ = Matrix{Float64}(undef, iteration, length(initial_θ))

    vec_saved_θ[1, :] = initial_θ
    @showprogress 1 "Vectorizing posterior samples..." for iter in 2:iteration
        κQ = saved_θ[:κQ][iter]
        kQ_infty = saved_θ[:kQ_infty][iter]
        if fix_const_PC1
            ϕ = saved_θ[:ϕ][iter] |> x -> vec(x)[2:end]
        else
            ϕ = saved_θ[:ϕ][iter] |> x -> vec(x)
        end
        σ²FF = saved_θ[:σ²FF][iter]
        ηψ = saved_θ[:ηψ][iter]
        ψ = saved_θ[:ψ][iter]
        ψ0 = saved_θ[:ψ0][iter]
        Σₒ = saved_θ[:Σₒ][iter]
        γ = saved_θ[:γ][iter]

        vec_saved_θ[iter, :] = [κQ; kQ_infty; ηψ; γ; Σₒ; σ²FF; ψ0; vec(ψ); ϕ]
    end
    vec_saved_θ = vec_saved_θ[:, findall(!iszero, var(vec_saved_θ, dims=1)[1, :])]

    ineff = Vector{Float64}(undef, size(vec_saved_θ)[2])
    kernel = QuadraticSpectralKernel{Andrews}()
    @showprogress 1 "Calculating Ineff factors..." for i in axes(vec_saved_θ, 2)
        object = Matrix{Float64}(undef, iteration, 1)
        object[:] = vec_saved_θ[:, i]
        bw = CovarianceMatrices.optimalbandwidth(kernel, object, prewhite=false)
        ineff[i] = Matrix(lrvar(QuadraticSpectralKernel(bw), object, scale=iteration / (iteration - 1)) / var(object))[1]
    end

    return ineff
end

# """
# load\\_object(saved\\_θ, object::String)
# * It derives an object in Vector "saved\\_θ" = Vector{Dict}(name => value, length(saved_θ))
# * Input: "object" is the name of the object of interest
# * Output: return[i] shows i'th iteration sample of "object" in saved_θ
# """
# function load_object(saved_θ, object::String)
#     return [saved_θ[i][object] for i in eachindex(saved_θ)]
# end
