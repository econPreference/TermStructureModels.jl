module GDTSM

using Base: @kwdef
using LinearAlgebra, Statistics, Distributions, SpecialFunctions, Roots, Optim, LineSearches, CovarianceMatrices, ProgressMeter
using BlackBoxOptim: bboptimize, best_candidate, MixedPrecisionRectSearchSpace
import Distributions: TDist
import Base: getindex

##When install RCall##
# 1. Install R form internet
# 2. In R, run " R.home() " and copy the home address
# 3. In R, run " install.packages("GIGrvg") " and " install.packages("glasso") " to install the packages
# 3. In Juila, run  " ENV["R_HOME"]="" "
# 4. In Juila, run  " ENV["PATH"]="...the address in step 2..." "
# 5. In Juila, run " using Pkg " and " Pkg.add("RCall") "
using RCall
######################

"""
* p: the lag of the transition equation
* q: the degree of shrinkages of the intercept and the slope coefficient of the transition equation
    - q[1]: shrinkages for the lagged dependent variable
    - q[2]: shrinkages for cross variables
    - q[3]: power of the lag shrinkage
    - q[4]: shrinkages for the intercept
* ν0(d.f.), Ω0(scale): hyper-parameters of the Inverse-Wishart prior distribution for the error covariance matrix in the transition equation
"""
@kwdef struct HyperParameter
    p::Float64
    q::Vector{Float64}
    ν0::Float64
    Ω0::Vector{Float64}
end
"""
* It contains structs, that are Parameter, LatentSpace, TermPremium, and Scenario.
"""
abstract type PosteriorSample end
function getindex(x::Vector{<:PosteriorSample}, c::Symbol)
    return getproperty.(x, c)
end

"""
* It contains statistical parameters of the model that are sampled from function "posterior_sampler".
"""
@kwdef struct Parameter <: PosteriorSample
    κQ::Float64
    kQ_infty::Float64
    ϕ::Matrix{Float64}
    σ²FF::Vector{Float64}
    ηψ::Float64 = 1.0
    ψ::Matrix{Float64} = ones(1, 1)
    ψ0::Vector{Float64} = ones(1)
    Σₒ::Vector{Float64}
    γ::Vector{Float64}
end
"""
* When the model goes to the JSZ latent factor space, the statistical parameters in struct Parameter are also transformed. This struct contains the transformed parameters.
* Transformation: latent Xₜ = T0P_ + inv(T1X)*PCsₜ
* Transition equation in the latent factor space
    - data = [latent macros]
    - data[t,:] = KₚXF + GₚXFXF*vec(data[t:-1:t-p+1]') + MvNormal(O,ΩXFXF)  
"""
@kwdef struct LatentSpace <: PosteriorSample
    latent::Matrix{Float64}
    κQ::Float64
    kQ_infty::Float64
    KₚXF::Vector{Float64}
    GₚXFXF::Matrix{Float64}
    ΩXFXF::Matrix{Float64}
end
"""
* It contains term premium estimates.
* TP: term premium estimates of a specific maturity bond
    - TP = timevarying_TP + const_TP + jensen
* timevarying_TP: rows:time, cols:factors, values: contributions of factors on TP
* const_TP: constant part in TP
* jensen: the part due to the Jensen's inequality
"""
@kwdef struct TermPremium <: PosteriorSample
    TP::Vector{Float64}
    timevarying_TP::Matrix{Float64}
    const_TP::Float64
    jensen::Float64
end

"""
* It contains conditioning scenarios in the scenario analysis.
* When y is a observed vector, combinations*y = values constitutes the scenario.
* Matrix combination[:,:,t] is the scenario at time t, and Vector values[:,t] is the conditioning value at time t.
    - combinations[:,:,t]*y = values[:,t] is the conditioning scenario at time t.
"""
@kwdef struct Scenario
    combinations::Array{Float64}
    values::Array{Float64}
end

"""
* It contains a result of the scenario analysis, the conditional prediction for yields, factors, and term premiums.
    - Prediction for the expectation hypothesis part = yields - TP
"""
@kwdef struct Prediction <: PosteriorSample
    yields::Matrix{Float64}
    factors::Matrix{Float64}
    TP::Array{Float64}
end

include("Theoretical.jl") # Theoretical results in GDTSM
include("prior.jl") # Contains prior distributions of statistical parameters
include("EB_marginal.jl") # Calculate the marginal likelihood of the transition VAR equation.
include("Empirical.jl") # Other statistical results not related to prior, posteior, and the marginal likelihood
include("Gibbs.jl") # posterior sampler.
include("scenario.jl") # scenario analysis
include("inference.jl") # implementation part

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
    tuning_hyperparameter,
    AR_res_var,
    generative,
    ineff_factor,
    posterior_sampler,
    sparsify_precision,
    # load_object,

    # priors.jl
    prior_κQ,
    dcurvature_dτ,

    # scenario.jl
    scenario_sampler,

    # Theoretical.jl
    GQ_XX,
    dimQ,
    termPremium,
    PCs_2_latents,
    PCA
end
