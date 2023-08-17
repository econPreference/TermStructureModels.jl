
"""
    tuning_hyperparameter(yields, macros, τₙ, ρ; populationsize=50, maxiter=10_000, medium_τ=12 * [2, 2.5, 3, 3.5, 4, 4.5, 5], upper_q=[1 1; 1 1; 10 10; 100 100], μkQ_infty=0, σkQ_infty=0.1, upper_ν0=[], μϕ_const=[], fix_const_PC1=false, upper_p=18, μϕ_const_PC1=[], data_scale=1200, medium_τ_pr=[])
It optimizes our hyperparameters by maximizing the marginal likelhood of the transition equation. Our optimizer is a differential evolutionary algorithm that utilizes bimodal movements in the eigen-space(Wang, Li, Huang, and Li, 2014) and the trivial geography(Spector and Klein, 2006).
# Input
- `populationsize` and `maxiter` is a option for the optimizer.
- The lower bounds for `q` and `ν0` are `0` and `dP+2`. 
- The upper bounds for `q`, `ν0` and VAR lag can be set by `upper_q`, `upper_ν0`, `upper_p`.
    - Our default option for `upper_ν0` is the time-series length of the data.
- If you use our default option for `μϕ_const`,
    1. `μϕ_const[dQ+1:end]` is a zero vector.
    2. `μϕ_const[1:dQ]` is calibrated to make a prior mean of `λₚ` a zero vector.
    3. After step 2, `μϕ_const[1]` is replaced with `μϕ_const_PC1` if it is not empty.
- `μϕ_const = Matrix(your prior, dP, upper_p)` 
- `μϕ_const[:,i]` is a prior mean for the VAR(`i`) constant. Therefore μϕ_const is a matrix only in this function. In other functions, `μϕ_const` is a vector for the orthogonalized VAR system with your selected lag.
- When `fix_const_PC1==true`, the first element in a constant term in our orthogonalized VAR is fixed to its prior mean during the posterior sampling.
- `data_scale::scalar`: In typical affine term structure model, theoretical yields are in decimal and not annualized. But, for convenience(public data usually contains annualized percentage yields) and numerical stability, we sometimes want to scale up yields, so want to use (`data_scale`*theoretical yields) as variable `yields`. In this case, you can use `data_scale` option. For example, we can set `data_scale = 1200` and use annualized percentage monthly yields as `yields`.
# Output(2)
optimized Hyperparameter, optimization result
- Be careful that we minimized the negative log marginal likelihood, so the second output is about the minimization problem.
"""
function tuning_hyperparameter(yields, macros, τₙ, ρ; populationsize=50, maxiter=10_000, medium_τ=12 * [2, 2.5, 3, 3.5, 4, 4.5, 5], upper_q=[1 1; 1 1; 10 10; 100 100], μkQ_infty=0, σkQ_infty=0.1, upper_ν0=[], μϕ_const=[], fix_const_PC1=false, upper_p=18, μϕ_const_PC1=[], data_scale=1200, medium_τ_pr=[])

    if isempty(upper_ν0) == true
        upper_ν0 = size(yields, 1)
    end

    dQ = dimQ()
    if isempty(macros)
        dP = dQ
    else
        dP = dQ + size(macros, 2)
    end
    if isempty(medium_τ_pr)
        medium_τ_pr = length(medium_τ) |> x -> ones(x) / x
    end

    lx = [0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 1; 1]
    ux = 0.0 .+ [vec(upper_q); upper_ν0 - (dP + 1); upper_p]
    if isempty(μϕ_const)
        μϕ_const = Matrix{Float64}(undef, dP, upper_p)
        for i in axes(μϕ_const, 2)
            μϕ_const_PCs = -calibration_μϕ_const(μkQ_infty, σkQ_infty, 120, yields[upper_p-i+1:end, :], τₙ, i; medium_τ, iteration=10_000, data_scale, medium_τ_pr)[2] |> x -> mean(x, dims=1)[1, :]
            if !isempty(μϕ_const_PC1)
                μϕ_const_PCs = [μϕ_const_PC1, μϕ_const_PCs[2], μϕ_const_PCs[3]]
            end
            if isempty(macros)
                μϕ_const[:, i] = deepcopy(μϕ_const_PCs)
            else
                μϕ_const[:, i] = [μϕ_const_PCs; zeros(size(macros, 2))]
            end
            @show calibration_μϕ_const(μkQ_infty, σkQ_infty, 120, yields[upper_p-i+1:end, :], τₙ, i; medium_τ, μϕ_const_PCs, iteration=10_000, data_scale, medium_τ_pr)[1] |> mean
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
        p = Int(input[10])

        PCs, ~, Wₚ = PCA(yields[(upper_p-p)+1:end, :], p)
        if isempty(macros)
            factors = PCs
        else
            factors = [PCs macros[(upper_p-p)+1:end, :]]
        end
        Ω0 = Vector{Float64}(undef, dP)
        for i in eachindex(Ω0)
            Ω0[i] = (AR_res_var(factors[:, i], p)[1]) * input[9]
        end

        tuned = Hyperparameter(p=p, q=q, ν0=ν0, Ω0=Ω0, μkQ_infty=μkQ_infty, σkQ_infty=σkQ_infty, μϕ_const=μϕ_const[:, p], fix_const_PC1=fix_const_PC1)
        if isempty(macros)
            return -log_marginal(factors, macros, ρ, tuned, τₙ, Wₚ; medium_τ, medium_τ_pr)
        else
            return -log_marginal(factors[:, 1:dQ], factors[:, dQ+1:end], ρ, tuned, τₙ, Wₚ; medium_τ, medium_τ_pr)
        end

        # Although the input data should contains initial observations, the argument of the marginal likelihood should be the same across the candidate models. Therefore, we should align the length of the dependent variable across the models.

    end

    ss = MixedPrecisionRectSearchSpace(lx, ux, [-1ones(Int64, 9); 0])
    opt = bboptimize(negative_log_marginal, starting; SearchSpace=ss, MaxSteps=maxiter, PopulationSize=populationsize, CallbackInterval=10, CallbackFunction=x -> println("Current Best: p = $(Int(best_candidate(x)[10])), q[:,1] = $(best_candidate(x)[1:4]), q[:,2] = $(best_candidate(x)[5:8]), ν0 = $(best_candidate(x)[9] + dP + 1)"))

    q = [best_candidate(opt)[1] best_candidate(opt)[5]
        best_candidate(opt)[2] best_candidate(opt)[6]
        best_candidate(opt)[3] best_candidate(opt)[7]
        best_candidate(opt)[4] best_candidate(opt)[8]]
    ν0 = best_candidate(opt)[9] + dP + 1
    p = best_candidate(opt)[10] |> Int

    PCs = PCA(yields[(upper_p-p)+1:end, :], p)[1]
    if isempty(macros)
        factors = PCs
    else
        factors = [PCs macros[(upper_p-p)+1:end, :]]
    end
    Ω0 = Vector{Float64}(undef, dP)
    for i in eachindex(Ω0)
        Ω0[i] = (AR_res_var(factors[:, i], p)[1]) * best_candidate(opt)[9]
    end

    return Hyperparameter(p=p, q=q, ν0=ν0, Ω0=Ω0, μkQ_infty=μkQ_infty, σkQ_infty=σkQ_infty, μϕ_const=μϕ_const[:, p], fix_const_PC1=fix_const_PC1, data_scale=data_scale), opt

end

"""
    AR_res_var(TS::Vector, p)
It derives an MLE error variance estimate of an AR(`p`) model
# Input
- univariate time series `TS` and the lag `p`
# output(2)
residual variance estimate, AR(p) coefficients
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
    posterior_sampler(yields, macros, τₙ, ρ, iteration, tuned::Hyperparameter; medium_τ=12 * [2, 2.5, 3, 3.5, 4, 4.5, 5], init_param=[], ψ=[], ψ0=[], γ_bar=[], medium_τ_pr=[])
This is a posterior distribution sampler.
# Input
- `iteration`: # of posterior samples
- `tuned`: optimized hyperparameters used during estimation
- `init_param`: starting point of the sampler. It should be a type of Parameter.
# Output(2)
`Vector{Parameter}(posterior, iteration)`, acceptance rate of the MH algorithm
"""
function posterior_sampler(yields, macros, τₙ, ρ, iteration, tuned::Hyperparameter; medium_τ=12 * [2, 2.5, 3, 3.5, 4, 4.5, 5], init_param=[], ψ=[], ψ0=[], γ_bar=[], medium_τ_pr=[])

    (; p, q, ν0, Ω0, μkQ_infty, σkQ_infty, μϕ_const, fix_const_PC1, data_scale) = tuned
    N = size(yields, 2) # of maturities
    dQ = dimQ()
    if isempty(macros)
        dP = dQ
    else
        dP = dQ + size(macros, 2)
    end
    if isempty(medium_τ_pr)
        medium_τ_pr = length(medium_τ) |> x -> ones(x) / x
    end
    Wₚ = PCA(yields, p)[3]
    prior_κQ_ = prior_κQ(medium_τ, medium_τ_pr)
    if isempty(γ_bar)
        γ_bar = prior_γ(yields, p)
    end

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
    @showprogress 5 "Sampling the posterior..." for iter in 1:iteration

        κQ = rand(post_κQ(yields, prior_κQ_, τₙ; kQ_infty, ϕ, σ²FF, Σₒ, data_scale))

        kQ_infty = rand(post_kQ_infty(μkQ_infty, σkQ_infty, yields, τₙ; κQ, ϕ, σ²FF, Σₒ, data_scale))

        ϕ, σ²FF, isaccept = post_ϕ_σ²FF(yields, macros, μϕ_const, ρ, prior_κQ_, τₙ; ϕ, ψ, ψ0, σ²FF, q, ν0, Ω0, κQ, kQ_infty, Σₒ, fix_const_PC1, data_scale)
        isaccept_MH += isaccept

        Σₒ = rand.(post_Σₒ(yields, τₙ; κQ, kQ_infty, ΩPP=ϕ_σ²FF_2_ΩPP(; ϕ, σ²FF), γ, p, data_scale))

        γ = rand.(post_γ(; γ_bar, Σₒ))

        saved_θ[iter] = Parameter(κQ=κQ, kQ_infty=kQ_infty, ϕ=ϕ, σ²FF=σ²FF, Σₒ=Σₒ, γ=γ)

    end

    return saved_θ, 100isaccept_MH / iteration
end

"""
    generative(T, dP, τₙ, p, noise::Float64; κQ, kQ_infty, KₚXF, GₚXFXF, ΩXFXF, data_scale=1200)
This function generate a simulation data given parameters. Note that all parameters are the things in the latent factor state space (that is, parameters in struct LatentSpace). There is some differences in notations because it is hard to express mathcal letters in VScode. So, mathcal{F} in my paper is expressed in `F` in the VScode. And, "F" in my paper is expressed as `XF`.
# Input: 
- noise = variance of the measurement errors
# Output(3)
`yields`, `latents`, `macros`
- `yields = Matrix{Float64}(obs,T,length(τₙ))`
- `latents = Matrix{Float64}(obs,T,dimQ())`
- `macros = Matrix{Float64}(obs,T,dP - dimQ())`
"""
function generative(T, dP, τₙ, p, noise::Float64; κQ, kQ_infty, KₚXF, GₚXFXF, ΩXFXF, data_scale=1200)
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
    aτ_ = aτ(τₙ[end], bτ_; kQ_infty, ΩXX, data_scale)
    Aₓ_ = Aₓ(aτ_, τₙ)

    yields = Matrix{Float64}(undef, T, N)
    for t = 1:T
        yields[t, :] = (Aₓ_ + Bₓ_ * XF[t, 1:dQ])' + rand(Normal(0, sqrt(noise)), N)'
    end

    return yields, XF[:, 1:dQ], XF[:, (dQ+1):end]
end

"""
    ineff_factor(saved_θ)
It returns inefficiency factors of each parameter
# Input
- `Vector{Parameter}` from posterior_sampler
# Output
- Estimated inefficiency factors are in Tuple(`κQ`, `kQ_infty`, `γ`, `Σₒ`, `σ²FF`, `ϕ`). For example, if you want to load an inefficiency factor of `ϕ`, you can use `Output.ϕ`.
- If `fix_const_PC1==true` in your optimized struct Hyperparameter, `Output.ϕ[1,1]` can be weird. So you should ignore it.
"""
function ineff_factor(saved_θ)

    iteration = length(saved_θ)

    init_κQ = saved_θ[:κQ][1]
    init_kQ_infty = saved_θ[:kQ_infty][1]
    init_ϕ = saved_θ[:ϕ][1] |> vec
    init_σ²FF = saved_θ[:σ²FF][1]
    init_Σₒ = saved_θ[:Σₒ][1]
    init_γ = saved_θ[:γ][1]

    initial_θ = [init_κQ; init_kQ_infty; init_γ; init_Σₒ; init_σ²FF; init_ϕ]
    vec_saved_θ = Matrix{Float64}(undef, iteration, length(initial_θ))
    vec_saved_θ[1, :] = initial_θ
    prog = Progress(iteration - 1; dt=5, desc="Vectorizing posterior samples...")
    Threads.@threads for iter in 2:iteration
        κQ = saved_θ[:κQ][iter]
        kQ_infty = saved_θ[:kQ_infty][iter]
        ϕ = saved_θ[:ϕ][iter] |> vec
        σ²FF = saved_θ[:σ²FF][iter]
        Σₒ = saved_θ[:Σₒ][iter]
        γ = saved_θ[:γ][iter]

        vec_saved_θ[iter, :] = [κQ; kQ_infty; γ; Σₒ; σ²FF; ϕ]
        next!(prog)
    end
    finish!(prog)

    ineff = Vector{Float64}(undef, size(vec_saved_θ)[2])
    kernel = QuadraticSpectralKernel{Andrews}()
    prog = Progress(size(vec_saved_θ, 2); dt=5, desc="Calculating Ineff factors...")
    Threads.@threads for i in axes(vec_saved_θ, 2)
        object = Matrix{Float64}(undef, iteration, 1)
        object[:] = vec_saved_θ[:, i]
        bw = CovarianceMatrices.optimalbandwidth(kernel, object, prewhite=false)
        ineff[i] = Matrix(lrvar(QuadraticSpectralKernel(bw), object, scale=iteration / (iteration - 1)) / var(object))[1]
        next!(prog)
    end
    finish!(prog)

    ϕ_ineff = ineff[2+length(init_γ)+length(init_Σₒ)+length(init_σ²FF)+1:end] |> x -> reshape(x, size(saved_θ[:ϕ][1], 1), size(saved_θ[:ϕ][1], 2))
    dP = size(ϕ_ineff, 1)
    for i in 1:dP, j in i:dP
        ϕ_ineff[i, end-dP+j] = 0
    end
    return (;
        κQ=ineff[1],
        kQ_infty=ineff[2],
        γ=ineff[2+1:2+length(init_γ)],
        Σₒ=ineff[2+length(init_γ)+1:2+length(init_γ)+length(init_Σₒ)],
        σ²FF=ineff[2+length(init_γ)+length(init_Σₒ)+1:2+length(init_γ)+length(init_Σₒ)+length(init_σ²FF)],
        ϕ=ϕ_ineff
    )
end
