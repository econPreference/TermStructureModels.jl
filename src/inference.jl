
"""
tuning_hyperparameter(yields, macros, τₙ, ρ; gradient=false)
* It derives the hyperparameters that maximize the marginal likelhood. First, the generating set search algorithm detemines the search range that do not make a final solution as a corner solution. Second, the evolutionary algorithm and Nelder-Mead algorithm find the global optimum. Lastly, the LBFGS algorithm calibrate the global optimum. 
* Input: Data should contain initial observations.
    - ρ = Vector{Float64}(0 or ≈1, dP-dQ). Usually, 0 for growth macro variables and 1 (or 0.9) for level macro variables.
    - If gradient == true, the LBFGS method is applied at the last.
* Output: struct Hyperparameter
"""
function tuning_hyperparameter(yields, macros, τₙ, ρ; populationsize=50, maxiter=10_000, medium_τ=12 * [2, 2.5, 3, 3.5, 4, 4.5, 5], upper_q=[1 1; 1 1; 10 10; 100 100], μkQ_infty=0, σkQ_infty=0.1, upper_ν0=[], μϕ_const=[], fix_const_PC1=false, upper_lag=6, μϕ_const_PC1=[])

    if isempty(upper_ν0) == true
        upper_ν0 = size(yields, 1)
    end

    dQ = dimQ()
    dP = dQ + size(macros, 2)
    lx = [0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 1; 1]
    ux = 0.0 .+ [vec(upper_q); upper_ν0 - (dP + 1); upper_lag]
    if isempty(μϕ_const)
        μϕ_const = Matrix{Float64}(undef, dP, upper_lag)
        for i in axes(μϕ_const, 2)
            μϕ_const_PCs = -calibration_μϕ_const(μkQ_infty, σkQ_infty, 120, yields[upper_lag-i+1:end, :], τₙ, i; medium_τ, iteration=10_000)[2] |> x -> mean(x, dims=1)[1, :]
            if !isempty(μϕ_const_PC1)
                μϕ_const_PCs = [μϕ_const_PC1, μϕ_const_PCs[2], μϕ_const_PCs[3]]
            end
            μϕ_const[:, i] = [μϕ_const_PCs; zeros(size(macros, 2))]
            @show calibration_μϕ_const(μkQ_infty, σkQ_infty, 120, yields[upper_lag-i+1:end, :], τₙ, i; medium_τ, μϕ_const_PCs, iteration=10_000)[1] |> mean
        end
    end
    starting = (lx + ux) ./ 2
    starting[end] = 1

    function negative_log_marginal(input)

        # parameters
        q = [input[1] input[5]
            input[2] input[6]
            input[3] input[7]
            input[4] input[8]]
        ν0 = input[9] + dP + 1
        lag = Int(input[10])

        PCs, ~, Wₚ = PCA(yields[(upper_lag-lag)+1:end, :], lag)
        factors = [PCs macros[(upper_lag-lag)+1:end, :]]
        Ω0 = Vector{Float64}(undef, dP)
        for i in eachindex(Ω0)
            Ω0[i] = (AR_res_var(factors[:, i], lag)[1]) * input[9]
        end

        tuned = Hyperparameter(p=lag, q=q, ν0=ν0, Ω0=Ω0, μkQ_infty=μkQ_infty, σkQ_infty=σkQ_infty, μϕ_const=μϕ_const[:, lag], fix_const_PC1=fix_const_PC1)
        return -log_marginal(factors[:, 1:dQ], factors[:, dQ+1:end], ρ, tuned, τₙ, Wₚ; medium_τ)

        # Although the input data should contains initial observations, the argument of the marginal likelihood should be the same across the candidate models. Therefore, we should align the length of the dependent variable across the models.

    end

    ss = MixedPrecisionRectSearchSpace(lx, ux, [-1ones(Int64, 9); 0])
    opt = bboptimize(negative_log_marginal, starting; SearchSpace=ss, MaxSteps=maxiter, PopulationSize=populationsize, CallbackInterval=10, CallbackFunction=x -> println("Current Best: p = $(Int(best_candidate(x)[10])), q = $(best_candidate(x)[1:8]), ν0 = $(best_candidate(x)[9] + dP + 1)"))

    q = [best_candidate(opt)[1] best_candidate(opt)[5]
        best_candidate(opt)[2] best_candidate(opt)[6]
        best_candidate(opt)[3] best_candidate(opt)[7]
        best_candidate(opt)[4] best_candidate(opt)[8]]
    ν0 = best_candidate(opt)[9] + dP + 1
    p = best_candidate(opt)[10] |> Int

    PCs = PCA(yields[(upper_lag-p)+1:end, :], p)[1]
    factors = [PCs macros[(upper_lag-p)+1:end, :]]
    Ω0 = Vector{Float64}(undef, dP)
    for i in eachindex(Ω0)
        Ω0[i] = (AR_res_var(factors[:, i], p)[1]) * best_candidate(opt)[9]
    end

    return Hyperparameter(p=p, q=q, ν0=ν0, Ω0=Ω0, μkQ_infty=μkQ_infty, σkQ_infty=σkQ_infty, μϕ_const=μϕ_const[:, p], fix_const_PC1=fix_const_PC1), opt

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
function posterior_sampler(yields, macros, τₙ, ρ, iteration, Hyperparameter_::Hyperparameter; medium_τ=12 * [2, 2.5, 3, 3.5, 4, 4.5, 5], init_param=[], ψ=[], ψ0=[])

    (; p, q, ν0, Ω0, μkQ_infty, σkQ_infty, μϕ_const, fix_const_PC1) = Hyperparameter_
    N = size(yields, 2) # of maturities
    dQ = dimQ()
    dP = dQ + size(macros, 2)
    Wₚ = PCA(yields, p)[3]
    prior_κQ_ = prior_κQ(medium_τ)
    γ_bar = prior_γ(yields[(p+1):end, :])

    if typeof(init_param) == Parameter
        (; κQ, kQ_infty, ϕ, σ²FF, Σₒ, γ) = init_param
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
        Σₒ = 1 ./ fill(γ_bar, N - dQ)
        γ = 1 ./ fill(γ_bar, N - dQ)
        ########################
    end
    if isempty(ψ)
        ψ = ones(dP, dP * p)
    end
    if isempty(ψ0)
        ψ0 = ones(dP)
    end

    isaccept_MH = zeros(dQ)
    saved_θ = Vector{Parameter}(undef, iteration)
    @showprogress 1 "Sampling the posterior..." for iter in 1:iteration

        κQ = rand(post_κQ(yields[(p+1):end, :], prior_κQ_, τₙ; kQ_infty, ϕ, σ²FF, Σₒ))

        kQ_infty = rand(post_kQ_infty(μkQ_infty, σkQ_infty, yields[(p+1):end, :], τₙ; κQ, ϕ, σ²FF, Σₒ))

        ϕ, σ²FF, isaccept = post_ϕ_σ²FF(yields, macros, μϕ_const, ρ, prior_κQ_, τₙ; ϕ, ψ, ψ0, σ²FF, q, ν0, Ω0, κQ, kQ_infty, Σₒ, fix_const_PC1)
        isaccept_MH += isaccept

        Σₒ = rand.(post_Σₒ(yields[(p+1):end, :], τₙ; κQ, kQ_infty, ΩPP=ϕ_σ²FF_2_ΩPP(; ϕ, σ²FF), γ))

        γ = rand.(post_γ(; γ_bar, Σₒ))

        saved_θ[iter] = Parameter(κQ=κQ, kQ_infty=kQ_infty, ϕ=ϕ, σ²FF=σ²FF, Σₒ=Σₒ, γ=γ)

    end

    return saved_θ, 100isaccept_MH / iteration
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
    if fix_const_PC1
        init_ϕ = saved_θ[:ϕ][1] |> x -> vec(x)[2:end] |> x -> [randn(); x]
    else
        init_ϕ = saved_θ[:ϕ][1] |> x -> vec(x)
    end
    initial_θ = [saved_θ[:κQ][1]; saved_θ[:kQ_infty][1]; saved_θ[:γ][1]; saved_θ[:Σₒ][1]; saved_θ[:σ²FF][1]; init_ϕ]
    vec_saved_θ = Matrix{Float64}(undef, iteration, length(initial_θ))
    vec_saved_θ[1, :] = initial_θ
    p = Progress(iteration - 1; dt=5, desc="Vectorizing posterior samples...")
    Threads.@threads for iter in 2:iteration
        κQ = saved_θ[:κQ][iter]
        kQ_infty = saved_θ[:kQ_infty][iter]
        if fix_const_PC1
            ϕ = saved_θ[:ϕ][iter] |> x -> vec(x)[2:end] |> x -> [randn(); x]
        else
            ϕ = saved_θ[:ϕ][iter] |> x -> vec(x)
        end
        σ²FF = saved_θ[:σ²FF][iter]
        Σₒ = saved_θ[:Σₒ][iter]
        γ = saved_θ[:γ][iter]

        vec_saved_θ[iter, :] = [κQ; kQ_infty; γ; Σₒ; σ²FF; ϕ]
        next!(p)
    end
    finish!(p)

    ineff = Vector{Float64}(undef, size(vec_saved_θ)[2])
    kernel = QuadraticSpectralKernel{Andrews}()
    p = Progress(size(vec_saved_θ, 2); dt=5, desc="Calculating Ineff factors...")
    Threads.@threads for i in axes(vec_saved_θ, 2)
        object = Matrix{Float64}(undef, iteration, 1)
        object[:] = vec_saved_θ[:, i]
        bw = CovarianceMatrices.optimalbandwidth(kernel, object, prewhite=false)
        ineff[i] = Matrix(lrvar(QuadraticSpectralKernel(bw), object, scale=iteration / (iteration - 1)) / var(object))[1]
        next!(p)
    end
    finish!(p)

    return (
        κQ=ineff[1],
        kQ_infty=ineff[2],
        γ=ineff[2+1:2+length(γ)],
        Σₒ=ineff[2+length(γ)+1:2+length(γ)+length(Σₒ)],
        σ²FF=ineff[2+length(γ)+length(Σₒ)+1:2+length(γ)+length(Σₒ)+length(σ²FF)],
        ϕ=ineff[2+length(γ)+length(Σₒ)+length(σ²FF)+1:end] |> x -> reshape(x, size(ϕ, 1), size(ϕ, 2))
    )
end
