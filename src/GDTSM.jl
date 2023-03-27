module GDTSM

using Base: @kwdef
using LinearAlgebra, Statistics, Distributions, SpecialFunctions, Roots, CovarianceMatrices, ProgressMeter, BlackBoxOptim, Distributed, PositiveFactorizations
import Distributions: TDist
import Base: getindex
import Statistics: mean, median, std, var, quantile

##When install RCall##
# 1. Install R form internet
# 2. In R, run " R.home() " and copy the home address
# 3. In R, run " install.packages("GIGrvg") " and " install.packages("glasso") " to install the packages
# 4. In Juila, run  " ENV["R_HOME"]="" "
# 5. In Juila, run  " ENV["PATH"]="...the address in step 2..." "
# 6. In Juila, run " using Pkg " and " Pkg.add("RCall") "
using RCall
######################

"""
* @kwdef struct HyperParameter
    - p::Int64
    - q::Vector{Float64}
    - ν0::Float64
    - Ω0::Vector{Float64}
* p: the lag of the transition equation
* q: the degree of shrinkages of the intercept and the slope coefficient of the transition equation
    - q[1]: shrinkages for the lagged dependent variable
    - q[2]: shrinkages for cross variables
    - q[3]: power of the lag shrinkage
    - q[4]: shrinkages for the intercept
* ν0(d.f.), Ω0(scale): hyper-parameters of the Inverse-Wishart prior distribution for the error covariance matrix in the transition equation
"""
@kwdef struct HyperParameter
    p::Int64
    q::Vector{Float64}
    ν0::Float64
    Ω0::Vector{Float64}
    σ²kQ_infty::Float64
end

"""
abstract type PosteriorSample
* It contains structs, that are Parameter, LatentSpace, TermPremium, and Scenario.
"""
abstract type PosteriorSample end

"""
* @kwdef struct Parameter <: PosteriorSample
    - κQ::Float64
    - kQ_infty::Float64
    - ϕ::Matrix{Float64}
    - σ²FF::Vector{Float64}
    - ηψ::Float64
    - ψ::Matrix{Float64}
    - ψ0::Vector{Float64}
    - Σₒ::Vector{Float64}
    - γ::Vector{Float64}
* It contains statistical parameters of the model that are sampled from function "posterior_sampler".
"""
@kwdef struct Parameter <: PosteriorSample
    κQ::Float64
    kQ_infty::Float64
    ϕ::Matrix{Float64}
    σ²FF::Vector{Float64}
    ηψ::Float64
    ψ::Matrix{Float64}
    ψ0::Vector{Float64}
    Σₒ::Vector{Float64}
    γ::Vector{Float64}
end
"""
* struct ReducedForm <: PosteriorSample
    - κQ::Float64
    - kQ_infty::Float64
    - KₚF::Vector{Float64}
    - GₚFF::Matrix{Float64}
    - ΩFF::Matrix{Float64}
    - Σₒ::Vector{Float64}
    - λP::Vector{Float64}
    - ΛPF::Matrix{Float64}
    - mpr::Matrix{Float64}
* It contains statistical parameters in terms of the reduced form VAR(p) in P-dynamics. λP and ΛPF are paramters in the market prices of risks equation, and they only contain non-zero elements. 
"""
@kwdef struct ReducedForm <: PosteriorSample
    κQ::Float64
    kQ_infty::Float64
    KₚF::Vector{Float64}
    GₚFF::Matrix{Float64}
    ΩFF::Matrix{Float64}
    Σₒ::Vector{Float64}
    λP::Vector{Float64}
    ΛPF::Matrix{Float64}
    mpr::Matrix{Float64}
end

"""
* @kwdef struct LatentSpace <: PosteriorSample
    - latents::Matrix{Float64}
    - κQ::Float64
    - kQ_infty::Float64
    - KₚXF::Vector{Float64}
    - GₚXFXF::Matrix{Float64}
    - ΩXFXF::Matrix{Float64}
* When the model goes to the JSZ latent factor space, the statistical parameters in struct Parameter are also transformed. This struct contains the transformed parameters.
* Transformation: latent Xₜ = T0P_ + inv(T1X)*PCsₜ
* Transition equation in the latent factor space
    - data = [latent macros]
    - data[t,:] = KₚXF + GₚXFXF*vec(data[t:-1:t-p+1]') + MvNormal(O,ΩXFXF)  
"""
@kwdef struct LatentSpace <: PosteriorSample
    latents::Matrix{Float64}
    κQ::Float64
    kQ_infty::Float64
    KₚXF::Vector{Float64}
    GₚXFXF::Matrix{Float64}
    ΩXFXF::Matrix{Float64}
end

"""
* @kwdef struct YieldCurve <: PosteriorSample
    - latents::Matrix{Float64}
    - yields::VecOrMat{Float64}
    - intercept::Union{Float64,Vector{Float64}}
    - slope::Matrix{Float64}
* It contains fitted yield curve.
    - yields[t,:] = intercept + slope*latents[t,:]
"""
@kwdef struct YieldCurve <: PosteriorSample
    latents::Matrix{Float64}
    yields::VecOrMat{Float64}
    intercept::Union{Float64,Vector{Float64}}
    slope::Matrix{Float64}
end

"""
* @kwdef struct TermPremium <: PosteriorSample
    - TP::Vector{Float64}
    - timevarying_TP::Matrix{Float64}
    - const_TP::Float64
    - jensen::Float64
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
* @kwdef struct Scenario
    - combinations::Array{Float64}
    - values::Array{Float64}
* It contains conditioning scenarios in the scenario analysis.
* When y is a observed vector, combinations*y = values constitutes the scenario.
    - Here, yₜ = [yields[t,:]; macros[t,:]]
* Matrix combination[:,:,t] is the scenario at time t, and Vector values[:,t] is the conditioned value at time t.
    - combinations[:,:,t]*yₜ = values[:,t] is the conditioned scenario at time t.
"""
@kwdef struct Scenario
    combinations::Array{Float64}
    values::Array{Float64}
end

"""
* @kwdef struct Forecast <: PosteriorSample
    - yields::Matrix{Float64}
    - factors::Matrix{Float64}
    - TP::Array{Float64}
* It contains a result of the scenario analysis, the conditional prediction for yields, factors = [PCs macros], and term premiums.
    - Prediction for the expectation hypothesis part = yields - TP
"""
@kwdef struct Forecast <: PosteriorSample
    yields::Matrix{Float64}
    factors::Matrix{Float64}
    TP::Array{Float64}
end

include("utilities.jl") # utility functions
include("theoretical.jl") # Theoretical results in GDTSM
include("prior.jl") # Contains prior distributions of statistical parameters
include("EB_marginal.jl") # Calculate the marginal likelihood of the transition VAR equation.
include("empirical.jl") # Other statistical results not related to prior, posteior, and the marginal likelihood
include("Gibbs.jl") # posterior sampler.
include("scenario.jl") # scenario analysis
include("inference.jl") # implementation part

export
    # EB_margianl.jl
    log_marginal,

    # empirical.jl
    loglik_mea,
    loglik_tran,
    isstationary,
    stationary_θ,
    LDL,
    ϕ_σ²FF_2_ΩFF,
    reducedform,

    # GDTSM.jl
    HyperParameter,
    PosteriorSample,
    Parameter,
    ReducedForm,
    LatentSpace,
    YieldCurve,
    TermPremium,
    Scenario,
    Forecast,

    # inference.jl
    tuning_hyperparameter,
    AR_res_var,
    generative,
    ineff_factor,
    posterior_sampler,
    sparse_precision,

    # priors.jl
    prior_κQ,
    dcurvature_dτ,

    # scenario.jl
    scenario_sampler,

    # theoretical.jl
    GQ_XX,
    dimQ,
    term_premium,
    latentspace,
    PCA,
    fitted_YieldCurve,
    maximum_SR,

    # utilities.jl
    getindex,
    mean,
    median,
    std,
    var,
    quantile

end
