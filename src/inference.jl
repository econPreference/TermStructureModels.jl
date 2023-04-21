
"""
tuning_hyperparameter(yields, macros, τₙ, ρ; gradient=false)
* It derives the hyperparameters that maximize the marginal likelhood. First, the generating set search algorithm detemines the search range that do not make a final solution as a corner solution. Second, the evolutionary algorithm and Nelder-Mead algorithm find the global optimum. Lastly, the LBFGS algorithm calibrate the global optimum. 
* Input: Data should contain initial observations.
    - ρ = Vector{Float64}(0 or ≈1, dP-dQ). Usually, 0 for growth macro variables and 1 (or 0.9) for level macro variables.
    - If gradient == true, the LBFGS method is applied at the last.
* Output: struct HyperParameter
"""
function tuning_hyperparameter(yields, macros, τₙ, ρ; populationsize=50, maxstep=10_000, medium_τ=12 * [1.5, 2, 2.5, 3, 3.5], upper_lag=12, upper_q1=1, upper_q4=100, upper_q5=100, σ²kQ_infty=1, weight=[], mSR_mean=Inf, AR_res_lag=4)

    dQ = dimQ()
    dP = dQ + size(macros, 2)
    PCs, ~, Wₚ = PCA(yields, upper_lag)
    AR_res_var_vec = Vector{Float64}(undef, dP)
    for i in eachindex(AR_res_var_vec)
        AR_res_var_vec[i] = AR_res_var([PCs macros][:, i], AR_res_lag)
    end

    starting = [1, upper_q1 / 2, 1, 2, upper_q4 / 2, upper_q5 / 2, 1]
    if isempty(weight)
        p = Int(starting[1])
        q = starting[2:6]
        q[2] = q[1] * q[2]
        ν0 = starting[7] + dP + 1
        Ω0 = AR_res_var_vec * starting[7]

        weight = 10^(ndigits(Int(floor(Int, -log_marginal(PCs[(upper_lag-p)+1:end, :], macros[(upper_lag-p)+1:end, :], ρ, HyperParameter(p=p, q=q, ν0=ν0, Ω0=Ω0, σ²kQ_infty=σ²kQ_infty), τₙ, Wₚ; medium_τ)))) + 1)
    end

    function negative_log_marginal(input)

        # parameters
        p = Int(input[1])
        q = input[2:6]
        q[2] = q[1] * q[2]
        ν0 = input[7] + dP + 1
        Ω0 = AR_res_var_vec * input[7]

        tuned = HyperParameter(p=p, q=q, ν0=ν0, Ω0=Ω0, σ²kQ_infty=σ²kQ_infty)
        if isinf(mSR_mean)
            return -log_marginal(PCs[(upper_lag-p)+1:end, :], macros[(upper_lag-p)+1:end, :], ρ, tuned, τₙ, Wₚ; medium_τ)
        else
            return -log_marginal(PCs[(upper_lag-p)+1:end, :], macros[(upper_lag-p)+1:end, :], ρ, tuned, τₙ, Wₚ; medium_τ) + weight * max(0.0, mean(maximum_SR(yields, macros, tuned, τₙ, ρ)) - mSR_mean)
        end
        # Although the input data should contains initial observations, the argument of the marginal likelihood should be the same across the candidate models. Therefore, we should align the length of the dependent variable across the models.

    end

    lx = 0.0 .+ [1; 0; 0; 2; 0; 0; 1]
    ux = 0.0 .+ [upper_lag; upper_q1; 1; 2; upper_q4; upper_q5; size(yields, 1)]
    ss = MixedPrecisionRectSearchSpace(lx, ux, [0; -1ones(Int64, 6)])
    EA_opt = bboptimize(bbsetup(negative_log_marginal; SearchSpace=ss, MaxSteps=maxstep, Workers=workers(), PopulationSize=populationsize, CallbackInterval=10, CallbackFunction=x -> println("Current Best: p = $(Int(best_candidate(x)[1])), q = $(best_candidate(x)[2:6].*[1,best_candidate(x)[2],1,1,1]), ν0 = $(best_candidate(x)[7] + dP + 1)")), starting)

    p = best_candidate(EA_opt)[1] |> Int
    q = best_candidate(EA_opt)[2:6]
    q[2] = q[1] * q[2]
    ν0 = best_candidate(EA_opt)[7] + dP + 1
    Ω0 = AR_res_var_vec * best_candidate(EA_opt)[7]

    return HyperParameter(p=p, q=q, ν0=ν0, Ω0=Ω0, σ²kQ_infty=σ²kQ_infty)

end

"""
tuning_hyperparameter_MOEA(yields, macros, τₙ, ρ; medium_τ=12 * [1.5, 2, 2.5, 3, 3.5], maxstep=10_000, mSR_scale=1.0, mSR_mean=1.0, upper_lag=9, upper_q1=1, upper_q45=100, σ²kQ_infty=1)
"""
function tuning_hyperparameter_MOEA(yields, macros, τₙ, ρ; populationsize=50, maxstep=10_000, medium_τ=12 * [1.5, 2, 2.5, 3, 3.5], weight=[], mSR_mean=1.0, upper_lag=12, upper_q1=1, upper_q4=100, upper_q5=100, σ²kQ_infty=1, AR_res_lag=4)

    dQ = dimQ()
    dP = dQ + size(macros, 2)
    PCs, ~, Wₚ = PCA(yields, upper_lag)
    AR_res_var_vec = Vector{Float64}(undef, dP)
    for i in eachindex(AR_res_var_vec)
        AR_res_var_vec[i] = AR_res_var([PCs macros][:, i], AR_res_lag)
    end

    starting = [1, upper_q1 / 2, 1, 2, upper_q4 / 2, upper_q5 / 2, 1]
    if isempty(weight)
        p = Int(starting[1])
        q = starting[2:6]
        q[2] = q[1] * q[2]
        ν0 = starting[7] + dP + 1
        Ω0 = AR_res_var_vec * starting[7]

        weight = 10^(ndigits(Int(floor(Int, -log_marginal(PCs[(upper_lag-p)+1:end, :], macros[(upper_lag-p)+1:end, :], ρ, HyperParameter(p=p, q=q, ν0=ν0, Ω0=Ω0, σ²kQ_infty=σ²kQ_infty), τₙ, Wₚ; medium_τ)))) + 1)
    end

    function negative_log_marginal(input)

        # parameters
        p = Int(input[1])
        q = input[2:6]
        q[2] = q[1] * q[2]
        ν0 = input[7] + dP + 1
        Ω0 = AR_res_var_vec * input[7]

        tuned = HyperParameter(p=p, q=q, ν0=ν0, Ω0=Ω0, σ²kQ_infty=σ²kQ_infty)
        return (-log_marginal(PCs[(upper_lag-p)+1:end, :], macros[(upper_lag-p)+1:end, :], ρ, tuned, τₙ, Wₚ; medium_τ), mean(maximum_SR(yields, macros, tuned, τₙ, ρ; iteration=100))) # Although the input data should contains initial observations, the argument of the marginal likelihood should be the same across the candidate models. Therefore, we should align the length of the dependent variable across the models.

    end

    lx = 0.0 .+ [1; 0; 0; 2; 0; 0; 1]
    ux = 0.0 .+ [upper_lag; upper_q1; 1; 2; upper_q4; upper_q5; size(yields, 1)]
    ss = MixedPrecisionRectSearchSpace(lx, ux, [0; -1ones(Int64, 6)])
    weightedfitness(f) = f[1] + weight * f[2]
    EA_opt = bboptimize(negative_log_marginal, starting; Method=:borg_moea, SearchSpace=ss, MaxSteps=maxstep, FitnessScheme=ParetoFitnessScheme{2}(is_minimizing=true, aggregator=weightedfitness), PopulationSize=populationsize)

    pf = pareto_frontier(EA_opt)
    best_obj1, idx_obj1 = findmin(map(elm -> abs(fitness(elm)[2] - mSR_mean), pf))
    bo1_solution = BlackBoxOptim.params(pf[idx_obj1])
    println("deviations from the target mSR(= $(mSR_mean)): $best_obj1")

    p = bo1_solution[1] |> Int
    q = bo1_solution[2:6]
    q[2] = q[1] * q[2]
    ν0 = bo1_solution[7] + dP + 1
    Ω0 = AR_res_var_vec * bo1_solution[7]

    return HyperParameter(p=p, q=q, ν0=ν0, Ω0=Ω0, σ²kQ_infty=σ²kQ_infty), EA_opt

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
    M = I(T) - X * ((X'X) \ X')
    return var(M * Y)
end

"""
mSR_ML_frontier(EA_opt, dM; mSR_mean=1.0, σ²kQ_infty=1)
"""
function mSR_ML_frontier(EA_opt, yields, macros; mSR_mean=1.0, σ²kQ_infty=1, upper_lag=9, AR_res_lag=4)

    dP = size(macros, 2) + dimQ()
    pf = pareto_frontier(EA_opt)
    best_obj1, idx_obj1 = findmin(map(elm -> abs(fitness(elm)[2] - mSR_mean), pf))
    bo1_solution = BlackBoxOptim.params(pf[idx_obj1])
    println("deviations from the target mSR(= $(mSR_mean)): $best_obj1")

    p = bo1_solution[1] |> Int
    q = bo1_solution[2:6]
    q[2] = q[1] * q[2]
    ν0 = bo1_solution[7] + dP + 1

    PCs = PCA(yields, upper_lag)[1]
    AR_res_var_vec = Vector{Float64}(undef, dP)
    for i in eachindex(AR_res_var_vec)
        AR_res_var_vec[i] = AR_res_var([PCs macros][:, i], AR_res_lag)
    end
    Ω0 = AR_res_var_vec * bo1_solution[7]

    set_fits = Matrix{Float64}(undef, length(pf), 2)
    for i in axes(set_fits, 1)
        set_fits[i, :] = [fitness(pf[i])[1] fitness(pf[i])[2]]
    end

    scat = scatter(set_fits[:, 2], -set_fits[:, 1], ylabel="marginal likelhood", xlabel="E[maximum SR]", label="")
    return HyperParameter(p=p, q=q, ν0=ν0, Ω0=Ω0, σ²kQ_infty=σ²kQ_infty), scat

end

"""
posterior_sampler(yields, macros, τₙ, ρ, iteration, HyperParameter_; sparsity=false, medium_τ=12 * [1.5, 2, 2.5, 3, 3.5])
* This is a posterior distribution sampler. It needs data and hyperparameters. 
* Input: Data should include initial observations. τₙ is a vector that contains observed maturities.
    - ρ = Vector{Float64}(0 or ≈1, dP-dQ). Usually, 0 for growth macro variables and 1 (or 0.9) for level macro variables. 
    - iteration: # of posterior samples
* Output(3): Vector{Parameter}(posterior, iteration), acceptPr_C_σ²FF, acceptPr_ηψ 
"""
function posterior_sampler(yields, macros, τₙ, ρ, iteration, HyperParameter_::HyperParameter; sparsity=false, medium_τ=12 * [1.5, 2, 2.5, 3, 3.5], init_param=[])

    (; p, q, ν0, Ω0, σ²kQ_infty) = HyperParameter_
    N = size(yields, 2) # of maturities
    dQ = dimQ()
    dP = dQ + size(macros, 2)
    PCs, ~, Wₚ = PCA(yields, p)
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

        kQ_infty = rand(post_kQ_infty(σ²kQ_infty, yields[(p+1):end, :], τₙ; κQ, ϕ, σ²FF, Σₒ))

        σ²FF, isaccept = post_σ²FF₁(yields, macros, τₙ, p; κQ, kQ_infty, ϕ, σ²FF, Σₒ, ν0, Ω0)
        isaccept_C_σ²FF[1] += isaccept

        ϕ, σ²FF, isaccept = post_C_σ²FF_dQ(yields, macros, τₙ, p; κQ, kQ_infty, ϕ, σ²FF, Σₒ, ν0, Ω0)
        isaccept_C_σ²FF[2:end] += isaccept

        if sparsity == true
            ηψ, isaccept = post_ηψ(; ηψ, ψ, ψ0)
            isaccept_ηψ += isaccept
        end

        ϕ, σ²FF = post_ϕ_σ²FF_remaining(PCs, macros, ρ, prior_κQ_, τₙ, Wₚ; ϕ, ψ, ψ0, σ²FF, q, ν0, Ω0)

        if sparsity == true
            ψ0, ψ = post_ψ_ψ0(ρ, prior_κQ_, τₙ, Wₚ; ϕ, ψ0, ψ, ηψ, q, σ²FF, ν0, Ω0)
        end

        Σₒ = rand.(post_Σₒ(yields[(p+1):end, :], τₙ; κQ, kQ_infty, ΩPP=ϕ_σ²FF_2_ΩPP(; ϕ, σ²FF), γ))

        γ = rand.(post_γ(; γ_bar, Σₒ))


        saved_θ[iter] = Parameter(κQ=κQ, kQ_infty=kQ_infty, ϕ=ϕ, σ²FF=σ²FF, ηψ=ηψ, ψ=ψ, ψ0=ψ0, Σₒ=Σₒ, γ=γ)

    end

    return saved_θ, 100isaccept_C_σ²FF / iteration, 100isaccept_ηψ / iteration
end

"""
sparse_precision(saved_θ, yields, macros, τₙ)
* It conduct the glasso of Friedman, Hastie, and Tibshirani (2022) using the method of Hauzenberger, Huber and Onorante. 
* That is, the posterior samples of ΩFF is penalized with L1 norm to impose a sparsity on the precision.
* Input: "saved\\_θ" from function posterior_sampler, and the data should contain initial observations.
* Output(3): sparse_θ, trace_λ, trace_sparsity
    - sparse_θ: sparsified posterior samples
    - trace_λ: a vector that contains an optimal lasso parameters in iterations
    - trace_sparsity: a vector that contains degree of freedoms of inv(ΩFF) in iterations
"""
function sparse_precision(saved_θ, T; lower_penalty=1e-2, nlambda=100)

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
        glasso_results = rcopy(rcall(:EBICglasso, ΩFF_, T, returnAllResults=true, var"lambda.min.ratio"=lower_penalty, nlambda=nlambda))
        sparse_prec = glasso_results[:optwi]
        sparse_cov = diagm(std_) * inv(sparse_prec) * diagm(std_) |> Symmetric

        sparsity = sum(abs.(sparse_prec) .> eps())
        trace_sparsity[iter] = sparsity
        inv_sparse_C, diagm_σ²FF = LDL(sparse_cov)
        ϕ = [ϕ0 (inv(inv_sparse_C) - I(dP))]
        σ²FF = diag(diagm_σ²FF)

        sparse_θ[iter] = Parameter(κQ=κQ, kQ_infty=kQ_infty, ϕ=ϕ, σ²FF=σ²FF, ηψ=ηψ, ψ=ψ, ψ0=ψ0, Σₒ=Σₒ, γ=γ)
    end

    return sparse_θ, trace_sparsity
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
function ineff_factor(saved_θ)

    iteration = length(saved_θ)

    κQ = saved_θ[:κQ][1]
    kQ_infty = saved_θ[:kQ_infty][1]
    ϕ = saved_θ[:ϕ][1]
    σ²FF = saved_θ[:σ²FF][1]
    ηψ = saved_θ[:ηψ][1]
    ψ = saved_θ[:ψ][1]
    ψ0 = saved_θ[:ψ0][1]
    Σₒ = saved_θ[:Σₒ][1]
    γ = saved_θ[:γ][1]

    initial_θ = [κQ; kQ_infty; vec(ϕ); σ²FF; ηψ; vec(ψ); ψ0; Σₒ; γ]
    vec_saved_θ = Matrix{Float64}(undef, iteration, length(initial_θ))

    vec_saved_θ[1, :] = initial_θ
    for iter in 2:iteration
        κQ = saved_θ[:κQ][iter]
        kQ_infty = saved_θ[:kQ_infty][iter]
        ϕ = saved_θ[:ϕ][iter]
        σ²FF = saved_θ[:σ²FF][iter]
        ηψ = saved_θ[:ηψ][iter]
        ψ = saved_θ[:ψ][iter]
        ψ0 = saved_θ[:ψ0][iter]
        Σₒ = saved_θ[:Σₒ][iter]
        γ = saved_θ[:γ][iter]

        vec_saved_θ[iter, :] = [κQ; kQ_infty; vec(ϕ); σ²FF; ηψ; vec(ψ); ψ0; Σₒ; γ]
    end

    ineff = Vector{Float64}(undef, length(initial_θ))
    kernel = QuadraticSpectralKernel{Andrews}()
    for i in axes(vec_saved_θ, 2)
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
