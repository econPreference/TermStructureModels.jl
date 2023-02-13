module GDTSM

using LinearAlgebra, Statistics, Distributions, Parameters, SpecialFunctions, Roots, Optim, LineSearches, CovarianceMatrices, ProgressMeter
import Distributions: TDist
using BlackBoxOptim: bboptimize, best_candidate, MixedPrecisionRectSearchSpace

##When install RCall
# 1. Install R form internet
# 2. In R, run " R.home() " and copy the home address
# 3. In R, run " install.packages("GIGrvg") " and install the package
# 3. In Juila, run  " ENV["R_HOME"]="" "
# 4. In Juila, run  " ENV["PATH"]="...the address in step 2..." "
# 5. In Juila, run " using Pkg " and " Pkg.add("RCall") "
using RCall

include("Theoretical.jl")
include("priors.jl")
include("EB_marginal.jl")
include("Empirical.jl")
include("gibbs.jl")

"""
This function generate a simulation data given parameters. Note that all parameters are the things in the latent factor state space. There is some differences in notations because it is hard to express mathcal letters in VScode. So, mathcal{F} in my paper is expressed in F in the VScode. And, "F" in my paper is expressed as XF. This confusion was made because XF factors are a little important for the research.

===
"""

function Tuning_Hyperparameter(yields, macros, ρ)

    dQ = dimQ()
    dP = dQ + size(macros)[2]
    T = size(macros)[1]
    p_max = 4

    function log_marginal_outer(input, p_max_)

        # parameters
        PCs = PCA(yields, p_max_)[1]

        p = Int(input[1])
        q = input[2:5]
        ν0 = input[6] + dP + 1
        Ω0 = Vector{Float64}(undef, dP)
        for i in 1:dP
            Ω0[i] = AR_res_var([PCs macros][(p_max_-p)+1:end, i], p)
        end

        ψ = ones(dP, dP * p)
        ψ0 = ones(dP)

        return -log_marginal(PCs[(p_max_-p)+1:end, :], macros[(p_max_-p)+1:end, :]; p, ν0, Ω0, q, ψ, ψ0, ρ)

    end

    lx = 0.0 .+ [1; zeros(4); 0]
    ux = 0.0 .+ [p_max; [1, 1, 4, 10]; 0.5T]
    starting = [1, 0.1, 0.1, 2, 2, 1]

    ss = MixedPrecisionRectSearchSpace(lx, ux, [0; -1ones(Int64, 5)])
    log_marginal_outer_(x) = log_marginal_outer(x, Int(ux[1]))
    LS_opt = bboptimize(log_marginal_outer_, starting; SearchSpace=ss, Method=:generating_set_search, MaxTime=10)
    corner_idx = findall([false; best_candidate(LS_opt)[2:end] .> 0.9ux[2:end]])
    corner_p = best_candidate(LS_opt)[1] == ux[1]

    while ~isempty(corner_idx) || corner_p
        if ~isempty(corner_idx)
            ux[corner_idx] += 1.2ux[corner_idx]
        else
            ux[1] += 1
        end
        ss = MixedPrecisionRectSearchSpace(lx, ux, [0; -1ones(Int64, 5)])
        log_marginal_outer_iter(x) = log_marginal_outer(x, Int(ux[1]))
        LS_opt = bboptimize(log_marginal_outer_iter, best_candidate(LS_opt); SearchSpace=ss, Method=:generating_set_search, MaxTime=10)

        corner_idx = findall([false; best_candidate(LS_opt)[2:end] .> 0.9ux[2:end]])
        corner_p = best_candidate(LS_opt)[1] == ux[1]
    end

    ss = MixedPrecisionRectSearchSpace(lx, ux, [0; -1ones(Int64, 5)])
    GA_opt = bboptimize(log_marginal_outer_, best_candidate(LS_opt); SearchSpace=ss)

    abs_log_marginal_outer(x) = log_marginal_outer([abs(round(Int, x[1])); abs.(x[2:6])], Int(ux[1]))
    NM_opt = optimize(abs_log_marginal_outer, best_candidate(GA_opt), NelderMead(), Optim.Options(show_trace=true))

    function log_marginal_outer2(input, p)

        # parameters
        PCs = PCA(yields, p)[1]

        p = Int(input[1])
        q = input[2:5]
        ν0 = input[6] + dP + 1
        Ω0 = Vector{Float64}(undef, dP)
        for i in 1:dP
            Ω0[i] = AR_res_var([PCs macros][:, i], p)
        end

        ψ = ones(dP, dP * p)
        ψ0 = ones(dP)

        return -log_marginal(PCs[(p+1):end, :], macros[(p+1):end, :]; p, ν0, Ω0, q, ψ, ψ0, ρ)

    end

    p = abs(round(Int, NM_opt.minimizer[1]))
    reduced_log_marginal_outer(x) = log_marginal_outer2([p; abs.(x)], p)
    NT_opt = optimize(reduced_log_marginal_outer, NM_opt.minimizer[2:6], LBFGS(; linesearch=LineSearches.BackTracking()), Optim.Options(show_trace=true))
    solution = abs.(NT_opt.minimizer)

    q = solution[1:4]
    ν0 = solution[5] + dP + 1
    Ω0 = Vector{Float64}(undef, dP)
    PCs = PCA(yields, p)[1]
    for i in 1:dP
        Ω0[i] = AR_res_var([PCs macros][:, i], p)
    end

    return p, q, ν0, Ω0

end

"""
    This is a posterior distribution sampler. It needs data and hyperparameters. Data should include initial conditions.
"""
function sampling_GDTSM(yields, macros, τₙ, ρ, iteration; p, q, ν0, Ω0)

    N = size(yields)[2]
    dQ = dimQ()
    dP = dQ + size(macros)[2]
    PCs = PCA(yields, p)[1]

    # additinoal hyperparameters
    γ_bar = prior_γ(yields[p+1:end, :])
    σ²kQ_infty = 100

    κQ = 0.0609 # initial parameters
    kQ_infty = 0.0
    ϕ = [zeros(dP) diagm([0.9ones(dQ); ρ]) zeros(dP, dP * (p - 1)) zeros(dP, dP)]
    σ²FF = [Ω0[i] / (ν0 + i - dP) for i in eachindex(Ω0)]
    ηψ = 1
    ψ = ones(dP, dP * p)
    ψ0 = ones(dP)
    Σₒ = 1 ./ fill(γ_bar, N - dQ)
    γ = 1 ./ fill(γ_bar, N - dQ)

    saved_θ = []
    @showprogress 1 "Sampling the posterior..." for iter in 1:iteration
        κQ = rand(post_κQ(yields[p+1:end, :], prior_κQ(), τₙ, p; kQ_infty, ϕ, σ²FF, Σₒ))

        kQ_infty = rand(post_kQ_infty(σ²kQ_infty, yields, τₙ, p; κQ, ϕ, σ²FF, Σₒ))

        σ²FF = post_σ²FF₁(yields, macros, τₙ, p; κQ, kQ_infty, ϕ, σ²FF, Σₒ, ν0, Ω0)

        ϕ, σ²FF = post_C_σ²FF_dQ(yields, macros, τₙ, p; κQ, kQ_infty, ϕ, σ²FF, Σₒ, ν0, Ω0)

        ηψ = post_ηψ(; ηψ, ψ, ψ0)

        ϕ, σ²FF = post_ϕ_σ²FF_remaining(PCs, macros, ρ; ϕ, ψ, ψ0, σ²FF, q, ν0, Ω0)

        ψ, ψ0 = post_ψ_ψ0(ρ; ϕ, ψ0, ψ, ηψ, q, σ²FF, ν0, Ω0)

        ΩPP = ϕ_σ²FF_2_ΩPP(; ϕ, σ²FF)
        Σₒ = [rand(post_Σₒ(yields[p+1:end, :], τₙ; κQ, kQ_infty, ΩPP, γ)[i]) for i in 1:N-dQ]

        γ = rand.(post_γ(γ_bar; Σₒ))

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

function AR_res_var(TS, p)
    Y = TS[p+1:end]
    T = length(Y)
    X = ones(T)
    for i in 1:p
        X = hcat(X, TS[p+1-i:end-i])
    end
    M = I(T) - X * ((X'X) \ X')
    return var(M * Y)
end

function generative(T, dP, τₙ, p; κQ, kQ_infty, KₚXF, GₚXFXF, ΩXFXF)
    N = length(τₙ)
    dQ = dimQ()

    # Generating latent factors
    XF = randn(p, dP)
    for horizon = 1:(1.5T)
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
        yields[t, :] = (Aₓ_ + Bₓ_' * XF[t, 1:dQ])' + rand(Normal(0, 0.1), N)'
    end

    return yields, XF[:, 1:dQ], XF[:, (dQ+1):end]
end

function effective_sample(saved_θ)

    iteration = size(saved_θ)[1]
    @unpack κQ, kQ_infty, σ²FF, ηψ, ψ, ψ0, Σₒ, γ = saved_θ[1]
    initial_θ = [κQ; kQ_infty; σ²FF; ηψ; vec(ψ); ψ0; Σₒ; γ]
    vec_saved_θ = Matrix{Float64}(undef, iteration, length(initial_θ))

    vec_saved_θ[1, :] = initial_θ
    for iter in 2:iteration
        @unpack κQ, kQ_infty, σ²FF, ηψ, ψ, ψ0, Σₒ, γ = saved_θ[iter]
        vec_saved_θ[iter, :] = [κQ; kQ_infty; σ²FF; ηψ; vec(ψ); ψ0; Σₒ; γ]
    end

    eff = Vector{Float64}(undef, length(initial_θ))
    kernel = QuadraticSpectralKernel{Andrews}()
    for i in 1:length(initial_θ)
        bw = CovarianceMatrices.optimalbandwidth(kernel, vec_saved_θ[:, i], prewhite=false)
        eff[i] = lrvar(QuadraticSpectralKernel(bw), vec_saved_θ[:, i], scale=iteration / (iteration - 1)) / var(vec_saved_θ[:, i])
    end

    return eff
end

function load_object(saved_θ, object)
    return [saved_θ[i][object] for i in eachindex(saved_θ)]
end
# 
export
    # GDTSM.jl
    Tuning_Hyperparameter,
    sampler,
    AR_res_var,
    generative,
    effective_sample,
    sampling_GDTSM,
    load_object,

    # EB_margianl.jl
    log_marginal,

    # Empirical.jl
    loglik_mea,
    loglik_tran,
    isstationary,

    # Theoretical.jl
    GQ_XX,
    dimQ,
    TP,
    PCs2latents,
    PCA
end

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
