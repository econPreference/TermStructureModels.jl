module GDTSM

using LinearAlgebra, Statistics, Distributions, SpecialFunctions, Roots, Optim, LineSearches, CovarianceMatrices, ProgressMeter
import Distributions: TDist
using BlackBoxOptim: bboptimize, best_candidate, MixedPrecisionRectSearchSpace

##When install RCall
# 1. Install R form internet
# 2. In R, run " R.home() " and copy the home address
# 3. In R, run " install.packages("GIGrvg") " and " install.packages("glasso") " to install the packages
# 3. In Juila, run  " ENV["R_HOME"]="" "
# 4. In Juila, run  " ENV["PATH"]="...the address in step 2..." "
# 5. In Juila, run " using Pkg " and " Pkg.add("RCall") "
using RCall

include("Theoretical.jl") # Theoretical results in GDTSM
include("priors.jl") # Contains prior distributions of statistical parameters
include("EB_marginal.jl") # Calculate the marginal likelihood of the transition VAR equation.
include("Empirical.jl") # Other statistical results not related to prior, posteior, and the marginal likelihood
include("Gibbs.jl") # posterior sampler.
include("scenario.jl") # scenario analysis

"""
Tuning_Hyperparameter(yields, macros, ρ; gradient=false)
* It derives the hyperparameters that maximize the marginal likelhood. First, the generating set search algorithm detemines the search range that do not make a final solution as a corner solution. Second, the evolutionary algorithm and Nelder-Mead algorithm find the global optimum. Lastly, the LBFGS algorithm calibrate the global optimum. 
* Input: Data should contain initial conditions.
    - ρ = Vector{Float64}(0 or ≈1, dP-dQ). Usually, 0 for growth macro variables and 1 (or 0.9) for level macro variables.
    - If gradient == true, the LBFGS method is applied at the last.
* Output(4): hyper-parameters, p, q, ν0, Ω0
"""
function Tuning_Hyperparameter(yields, macros, ρ; gradient=false)

    dQ = dimQ()
    dP = dQ + size(macros, 2)
    p_max = 4 # initial guess for the maximum lag

    function negative_log_marginal(input, p_max_)

        # parameters
        PCs = PCA(yields, p_max_)[1]

        p = Int(input[1])
        if p < 1
            return Inf
        end
        q = input[2:5]
        ν0 = input[6] + dP + 1
        Ω0 = input[7:end]

        ψ = ones(dP, dP * p)
        ψ0 = ones(dP)

        return -_log_marginal(PCs[(p_max_-p)+1:end, :], macros[(p_max_-p)+1:end, :], ρ; p, ν0, Ω0, q, ψ, ψ0) # Although the input data should contains initial conditions, the argument of the marginal likelihood should be the same across the candidate models. Therefore, we should align the length of the dependent variable across the models.

    end

    PCs = PCA(yields, p_max)[1]
    starting = [1, 0.1, 0.1, 2, 2, 1]
    for i in 1:dP
        push!(starting, AR_res_var([PCs macros][:, i], p_max))
    end
    lx = 0.0 .+ [1; zeros(4); 0; zeros(dP)]
    ux = 0.0 .+ [p_max; [1, 1, 4, 10]; 0.5size(macros, 1); 10starting[7:end]]

    ss = MixedPrecisionRectSearchSpace(lx, ux, [0; -1ones(Int64, 5 + dP)])
    obj_GSS0(x) = negative_log_marginal(x, Int(ux[1]))
    LS_opt = bboptimize(obj_GSS0, starting; SearchSpace=ss, Method=:generating_set_search, MaxTime=10)
    corner_idx = findall([false; best_candidate(LS_opt)[2:end] .> 0.9ux[2:end]])
    corner_p = best_candidate(LS_opt)[1] == ux[1]

    while ~isempty(corner_idx) || corner_p
        if ~isempty(corner_idx)
            ux[corner_idx] += ux[corner_idx]
        else
            ux[1] += 1
        end
        ss = MixedPrecisionRectSearchSpace(lx, ux, [0; -1ones(Int64, 5 + dP)])
        obj_GSS(x) = negative_log_marginal(x, Int(ux[1]))
        LS_opt = bboptimize(obj_GSS, best_candidate(LS_opt); SearchSpace=ss, Method=:generating_set_search, MaxTime=10)

        corner_idx = findall([false; best_candidate(LS_opt)[2:end] .> 0.9ux[2:end]])
        corner_p = best_candidate(LS_opt)[1] == ux[1]
    end
    obj_EA(x) = negative_log_marginal(x, Int(ux[1]))
    EA_opt = bboptimize(obj_EA, best_candidate(LS_opt); SearchSpace=ss)

    obj_NM(x) = negative_log_marginal([abs(ceil(Int, x[1])); abs.(x[2:end])], Int(ux[1]))
    NM_opt = optimize(obj_NM, best_candidate(EA_opt), NelderMead(), Optim.Options(show_trace=true))

    if gradient == true
        function negative_log_marginal_p(input, p)

            # parameters
            PCs = PCA(yields, p)[1]

            q = input[1:4]
            ν0 = input[5] + dP + 1
            Ω0 = input[6:end]

            ψ = ones(dP, dP * p)
            ψ0 = ones(dP)

            return -_log_marginal(PCs, macros, ρ; p, ν0, Ω0, q, ψ, ψ0) # the function should contains the initial conditions

        end

        p = abs(ceil(Int, NM_opt.minimizer[1]))
        obj_NT(x) = negative_log_marginal_p(abs.(x), p)
        NT_opt = optimize(obj_NT, NM_opt.minimizer[2:end], LBFGS(; linesearch=LineSearches.BackTracking()), Optim.Options(show_trace=true))
        solution = abs.(NT_opt.minimizer)

        q = solution[1:4]
        ν0 = solution[5] + dP + 1
        Ω0 = solution[6:end]
    else
        solution = abs.(NM_opt.minimizer)
        p = abs(ceil(Int, NM_opt.minimizer[1]))
        q = solution[2:5]
        ν0 = solution[6] + dP + 1
        Ω0 = solution[7:end]
    end

    return p, q, ν0, Ω0

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
posterior_sampler(yields, macros, τₙ, ρ, iteration; p, q, ν0, Ω0)
* This is a posterior distribution sampler. It needs data and hyperparameters. 
* Input: Data should include initial conditions. τₙ is a vector that contains observed maturities.
    - ρ = Vector{Float64}(0 or ≈1, dP-dQ). Usually, 0 for growth macro variables and 1 (or 0.9) for level macro variables. 
    - iteration: # of posterior samples
* Output: Vector{Dict}(, iteration)
    - "κQ", "kQ_infty", "ϕ", "σ²FF" , "ηψ", "ψ", "ψ0","Σₒ", "γ" ∈ Output[i]
    - Each element in the vector is a dictionary that "parameter"=>posterior sample value. Function load_object(output of the sampling function, "parameter") gives a vector of posteior samples of "parameter" in which which element has a numerical posteior sample value.
"""
function posterior_sampler(yields, macros, τₙ, ρ, iteration; p, q, ν0, Ω0)

    N = size(yields, 2) # of maturities
    dQ = dimQ()
    dP = dQ + size(macros, 2)
    PCs = PCA(yields, p)[1]

    ## additinoal hyperparameters ##
    γ_bar = prior_γ(yields[(p+1):end, :])
    σ²kQ_infty = 100 # prior variance of kQ_infty
    ################################

    ## initial parameters ##
    κQ = 0.0609
    kQ_infty = 0.0
    ϕ = [zeros(dP) diagm([0.9ones(dQ); ρ]) zeros(dP, dP * (p - 1)) zeros(dP, dP)] # The last dP by dP block matrix in ϕ should always be a lower triangular matrix whose diagonals are also always zero.
    ϕ[1:dQ, 2:(dQ+1)] = GQ_XX(; κQ)
    σ²FF = [Ω0[i] / (ν0 + i - dP) for i in eachindex(Ω0)]
    ηψ = 1
    ψ = ones(dP, dP * p)
    ψ0 = ones(dP)
    Σₒ = 1 ./ fill(γ_bar, N - dQ)
    γ = 1 ./ fill(γ_bar, N - dQ)
    ########################

    saved_θ = []
    @showprogress 1 "Sampling the posterior..." for iter in 1:iteration
        κQ = rand(post_κQ(yields[(p+1):end, :], prior_κQ(), τₙ; kQ_infty, ϕ, σ²FF, Σₒ))

        kQ_infty = rand(post_kQ_infty(σ²kQ_infty, yields[(p+1):end, :], τₙ; κQ, ϕ, σ²FF, Σₒ))

        σ²FF = post_σ²FF₁(yields, macros, τₙ, p; κQ, kQ_infty, ϕ, σ²FF, Σₒ, ν0, Ω0)

        ϕ, σ²FF = post_C_σ²FF_dQ(yields, macros, τₙ, p; κQ, kQ_infty, ϕ, σ²FF, Σₒ, ν0, Ω0)

        ηψ = post_ηψ(; ηψ, ψ, ψ0)

        ϕ, σ²FF = post_ϕ_σ²FF_remaining(PCs, macros, ρ; ϕ, ψ, ψ0, σ²FF, q, ν0, Ω0)

        ψ0, ψ = post_ψ_ψ0(ρ; ϕ, ψ0, ψ, ηψ, q, σ²FF, ν0, Ω0)

        Σₒ = rand.(post_Σₒ(yields[(p+1):end, :], τₙ; κQ, kQ_infty, ΩPP=ϕ_σ²FF_2_ΩPP(; ϕ, σ²FF), γ))

        γ = rand.(post_γ(; γ_bar, Σₒ))

        push!(saved_θ,
            Dict(
                "κQ" => κQ,
                "kQ_infty" => kQ_infty,
                "ϕ" => ϕ,
                "σ²FF" => σ²FF,
                "ηψ" => ηψ,
                "ψ" => ψ,
                "ψ0" => ψ0,
                "Σₒ" => Σₒ,
                "γ" => γ
            ))
    end

    return saved_θ
end

"""
sparsify_precision(saved_θ, yields, macros, τₙ)
* It conduct the glasso of Friedman, Hastie, and Tibshirani (2022) using the method of Hauzenberger, Huber and Onorante. 
* That is, the posterior samples of ΩFF is penalized with L1 norm to get a sparsity on the precision.
* Input: "saved\\_θ" from function posterior_sampler, and the data should contain initial conditions.
* Output(3): sparse_θ, trace_λ, trace_sparsity
    - sparse_θ: sparsified posterior samples
    - trace_λ: a vector that contains an optimal lasso parameters in iterations
    - trace_sparsity: a vector that contains degree of freedoms of inv(ΩFF) in iterations
"""
function sparsify_precision(saved_θ, yields, macros, τₙ)

    R"library(glasso)"
    ψ = saved_θ[1]["ψ"]
    dP = size(ψ, 1)
    p = Int(size(ψ, 2) / dP)
    T = size(yields, 1) - p
    PCs = PCA(yields, p)[1]

    iteration = length(saved_θ)
    iter_print = 1
    sparse_θ = []
    trace_λ = Vector{Float64}(undef, iteration)
    trace_sparsity = Vector{Float64}(undef, iteration)
    for iter in 1:iteration
        if iter == iter_print
            println("iteration = $iter")
            iter_print += 20
        end

        κQ = saved_θ[iter]["κQ"]
        kQ_infty = saved_θ[iter]["kQ_infty"]
        ϕ = saved_θ[iter]["ϕ"]
        σ²FF = saved_θ[iter]["σ²FF"]
        ηψ = saved_θ[iter]["ηψ"]
        ψ = saved_θ[iter]["ψ"]
        ψ0 = saved_θ[iter]["ψ0"]
        Σₒ = saved_θ[iter]["Σₒ"]
        γ = saved_θ[iter]["γ"]
        ϕ0, C = ϕ_2_ϕ₀_C(; ϕ)

        ΩFF_ = (C \ diagm(σ²FF)) / C'
        ΩFF_ = 0.5(ΩFF_ + ΩFF_')
        function glasso(λ)

            glasso_results = rcopy(rcall(:glasso, s=ΩFF_, rho=abs.(λ ./ inv(ΩFF_))))
            sparse_cov = glasso_results[:w]
            sparse_prec = glasso_results[:wi]

            inv_sparse_C, sparse_σ²FF = LDLt(sparse_cov)
            sparse_ϕ = [ϕ0 (inv(inv_sparse_C) - I(dP))]

            BIC_ = loglik_mea(yields[(p+1):end, :], τₙ; κQ, kQ_infty, ϕ=sparse_ϕ, σ²FF=diag(sparse_σ²FF), Σₒ)
            BIC_ += loglik_tran(PCs, macros; ϕ=sparse_ϕ, σ²FF=diag(sparse_σ²FF))
            BIC_ *= -2
            sparsity = sum(abs.(sparse_prec) .> eps())
            BIC_ += sparsity * log(T)

            return sparse_cov, BIC_, sparsity
        end

        obj(x) = glasso(abs(x[1]))[2]
        if iter > 1
            optim = optimize(obj, [trace_λ[iter-1]], NelderMead())
            λ_best = abs(optim.minimizer[1])
        else
            optim = bboptimize(obj; SearchRange=(-10.0, 10.0), NumDimensions=1)
            λ_best = abs(best_candidate(optim)[1])
        end
        trace_λ[iter] = λ_best

        sparse_cov, ~, sparsity = glasso(λ_best)
        trace_sparsity[iter] = sparsity
        inv_sparse_C, diagm_σ²FF = LDLt(sparse_cov)
        ϕ = [ϕ0 (inv(inv_sparse_C) - I(dP))]
        σ²FF = diag(diagm_σ²FF)

        push!(sparse_θ,
            Dict(
                "κQ" => κQ,
                "kQ_infty" => kQ_infty,
                "ϕ" => ϕ,
                "σ²FF" => σ²FF,
                "ηψ" => ηψ,
                "ψ" => ψ,
                "ψ0" => ψ0,
                "Σₒ" => Σₒ,
                "γ" => γ
            ))
    end

    return sparse_θ, trace_λ, trace_sparsity
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

    κQ = saved_θ[1]["κQ"]
    kQ_infty = saved_θ[1]["kQ_infty"]
    ϕ = saved_θ[1]["ϕ"]
    σ²FF = saved_θ[1]["σ²FF"]
    ηψ = saved_θ[1]["ηψ"]
    ψ = saved_θ[1]["ψ"]
    ψ0 = saved_θ[1]["ψ0"]
    Σₒ = saved_θ[1]["Σₒ"]
    γ = saved_θ[1]["γ"]

    initial_θ = [κQ; kQ_infty; vec(ϕ); σ²FF; ηψ; vec(ψ); ψ0; Σₒ; γ]
    vec_saved_θ = Matrix{Float64}(undef, iteration, length(initial_θ))

    vec_saved_θ[1, :] = initial_θ
    for iter in 2:iteration
        κQ = saved_θ[iter]["κQ"]
        kQ_infty = saved_θ[iter]["kQ_infty"]
        ϕ = saved_θ[iter]["ϕ"]
        σ²FF = saved_θ[iter]["σ²FF"]
        ηψ = saved_θ[iter]["ηψ"]
        ψ = saved_θ[iter]["ψ"]
        ψ0 = saved_θ[iter]["ψ0"]
        Σₒ = saved_θ[iter]["Σₒ"]
        γ = saved_θ[iter]["γ"]

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

"""
load\\_object(saved\\_θ, object::String)
* It derives an object in Vector "saved\\_θ" = Vector{Dict}(name => value, length(saved_θ))
* Input: "object" is the name of the object of interest
* Output: return[i] shows i'th iteration sample of "object" in saved_θ
"""
function load_object(saved_θ, object::String)
    return [saved_θ[i][object] for i in eachindex(saved_θ)]
end


export
    # EB_margianl.jl
    log_marginal,

    # Empirical.jl
    loglik_mea,
    loglik_tran,
    isstationary,
    stationary_θ,
    LDLt,
    ϕ_σ²FF_2_ΩFF,

    # GDTSM.jl
    Tuning_Hyperparameter,
    Tuning_Hyperparameter_compact,
    AR_res_var,
    generative,
    ineff_factor,
    posterior_sampler,
    sparsify_precision,
    load_object,

    # priors.jl
    prior_κQ,
    dcurvature_dτ,

    # scenario.jl
    scenario_sampler,

    # Theoretical.jl
    GQ_XX,
    dimQ,
    TP,
    PCs_2_latents,
    PCA
end


## Trash codes

# reduced_log_marginal(x) = log_marginal_outer([Int(round(x[1])); abs.(x[2:6])])
# PSO_opt = optimize(reduced_log_marginal, lx, ux, ParticleSwarm(; lower=lx, upper=ux), Optim.Options(show_trace=true))
# SA_opt = optimize(reduced_log_marginal, lx, ux, best_candidate(LS_opt), SAMIN(), Optim.Options(show_trace=true))

# so = pyimport("scipy.optimize")
# reduced_log_marginal(x) = log_marginal_outer([best_candidate(GA_optimum)[1]; abs.(x[1:5])])
# DE_opt = so.differential_evolution(reduced_log_marginal, [lx[2:end] ux[2:end]])
# DA_opt = so.dual_annealing(reduced_log_marginal, [lx[2:end] ux[2:end]])

# grid = [1:1:p_max, # lag p
# (0:0.01:0.1) .+ 0.0001, # q1
# (0:0.01:0.1) .+ 0.0001, # q2
# 0:2:4,         # q3
# (0:2:6) .+ 0.0001,# q4
# 0:0.1T:0.3T]   # ν0

# Grid_input = []
# Grid_log_marginal = []
# @showprogress 1 "Maximizing ML on a sparse grid" for (l, q1, q2, q3, q4, ν0) in Iterators.product(grid[1], grid[2], grid[3], grid[4], grid[5], grid[6])
# input = [l, q1, q2, q3, q4, ν0]
# push!(Grid_input, input)
# push!(Grid_log_marginal, log_marginal_outer(input))
# end
# Grid_input = Grid_input[findall(isfinite.(Grid_log_marginal))]
# Grid_log_marginal = Grid_log_marginal[findall(isfinite.(Grid_log_marginal))]
# Grid_optimum = Grid_input[findmin(Grid_log_marginal)[2]]
# Grid_optimum = floor.(Grid_optimum; digits=3)
