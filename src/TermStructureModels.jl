module TermStructureModels

using Base: @kwdef
using LinearAlgebra, Statistics, Distributions, SpecialFunctions, CovarianceMatrices, ProgressMeter, Distributed, Random, Roots, BlackBoxOptim
import Base: getindex
import Statistics: mean, median, std, var, quantile

"""
    @kwdef struct Hyperparameter
- `p::Int`
- `q::Matrix`
- `nu0`
- `Omega0::Vector`
- `mean_phi_const::Vector = zeros(length(Omega0))`: It is a prior mean of a constant term in our VAR.
"""
@kwdef struct Hyperparameter
    p::Int
    q::Matrix
    nu0
    Omega0::Vector
    mean_phi_const::Vector = zeros(length(Omega0))
end

"""
    abstract type PosteriorSample
It is a super-set of structs `Parameter`, `ReducedForm`, `LatentSpace`, `YieldCurve`, `TermPremium`, `Forecast`.
"""
abstract type PosteriorSample end

"""
    @kwdef struct Parameter <: PosteriorSample
It contains statistical parameters of the model that are sampled from function `posterior_sampler`.
- `kappaQ::Float64`
- `kQ_infty::Float64`
- `phi::Matrix{Float64}`
- `varFF::Vector{Float64}`
- `SigmaO::Vector{Float64}`
- `gamma::Vector{Float64}`
"""
@kwdef struct Parameter <: PosteriorSample
    kappaQ::Float64
    kQ_infty::Float64
    phi::Matrix{Float64}
    varFF::Vector{Float64}
    SigmaO::Vector{Float64}
    gamma::Vector{Float64}
end

"""
    @kwdef struct ReducedForm <: PosteriorSample
It contains statistical parameters in terms of the reduced form VAR(p) in P-dynamics. `lambdaP` and `LambdaPF` are parameters in the market prices of risks equation, and they only contain the first `dQ` non-zero equations. 
- `kappaQ`
- `kQ_infty`
- `KPF`
- `GPFF`
- `OmegaFF::Matrix`
- `SigmaO::Vector`
- `lambdaP`
- `LambdaPF`
- `mpr::Matrix(`market prices of risks`, T, dP)`
"""
@kwdef struct ReducedForm <: PosteriorSample
    kappaQ
    kQ_infty
    KPF
    GPFF
    OmegaFF::Matrix
    SigmaO::Vector
    lambdaP
    LambdaPF
    mpr::Matrix
end

"""
    @kwdef struct LatentSpace <: PosteriorSample 
When the model goes to the JSZ latent factor space, the statistical parameters in struct Parameter are also transformed. This struct contains the transformed parameters. Specifically, the transformation is `latents[t,:] = T0P_ + inv(T1X)*PCs[t,:]`. 

In the latent factor space, the transition equation is `data[t,:] = KPXF + GPXFXF*vec(data[t-1:-1:t-p,:]') + MvNormal(O,OmegaXFXF)`, where `data = [latent macros]`.
- `latents::Matrix`
- `kappaQ`
- `kQ_infty`
- `KPXF::Vector`
- `GPXFXF::Matrix`
- `OmegaXFXF::Matrix`
"""
@kwdef struct LatentSpace <: PosteriorSample
    latents::Matrix
    kappaQ
    kQ_infty
    KPXF::Vector
    GPXFXF::Matrix
    OmegaXFXF::Matrix
end

"""
    @kwdef struct YieldCurve <: PosteriorSample 
It contains a fitted yield curve. `yields[t,:] = intercept + slope*latents[t,:]` holds.
- `latents::Matrix`: latent pricing factors in LatentSpace
- `yields`
- `intercept`
- `slope`
"""
@kwdef struct YieldCurve <: PosteriorSample
    latents::Matrix
    yields
    intercept
    slope
end

"""
    @kwdef struct TermPremium <: PosteriorSample 
It contains a estimated time series of a term premium for one maturity.
- `TP::Vector`: term premium estimates of a specific maturity bond. `TP = timevarying_TP + const_TP + jensen` holds.
- `timevarying_TP::Matrix`: rows:time, cols:factors, values: contributions of factors on TP
- `const_TP::Float64`: constant part in TP
- `jensen::Float64`: the part due to the Jensen's inequality
"""
@kwdef struct TermPremium <: PosteriorSample
    TP::Vector
    timevarying_TP::Matrix
    const_TP::Float64
    jensen::Float64
end

"""
    @kwdef struct Scenario
It contains scenarios to be conditioned in the scenario analysis. When `y = [yields; macros]` is a observed vector in our measurement equation, `Scenario.combinations*y = Scenario.values` constitutes the scenario at a specific time. `Vector{Scenario}` is used to describe a time-series of scenarios.

`combinations` and `values` should be `Matrix` and `Vector`. If `values` is a scalar, `combinations` would be a matrix with one raw vector and `values` should be one-dimensional vector, for example [values]. 
- `combinations::Matrix`
- `values::Vector`
"""
@kwdef struct Scenario
    combinations::Matrix
    values::Vector
end

"""
    @kwdef struct Forecast <: PosteriorSample
It contains a result of the scenario analysis, the conditional prediction for yields, `factors = [PCs macros]`, and term premiums.
- `yields`
- `factors`
- `TP`: term premium forecasts
"""
@kwdef struct Forecast <: PosteriorSample
    yields
    factors
    TP
end

include("utilities.jl") # utility functions
include("theoreticals.jl") # Theoretical results
include("prior.jl") # Contains prior distributions of statistical parameters
include("EB_marginal.jl") # Calculate the marginal likelihood of the transition VAR equation.
include("empiricals.jl") # Other statistical results not related to prior, posteior, and the marginal likelihood
include("gibbs.jl") # posterior sampler.
include("scenario.jl") # scenario analysis
include("inference.jl") # implementation part

export
    # EB_margianl.jl
    log_marginal,

    # empiricals.jl
    loglik_mea,
    loglik_tran,
    isstationary,
    erase_nonstationary_param,
    LDL,
    phi_varFF_2_OmegaFF,
    reducedform,
    phi_2_phi₀_C,
    calibrate_mean_phi_const,

    # TermStructureModels.jl
    Hyperparameter,
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

    # priors.jl
    prior_kappaQ,
    dcurvature_dτ,

    # scenario.jl
    conditional_forecasts,
    scenario_analysis,

    # theoreticals.jl
    GQ_XX,
    dimQ,
    term_premium,
    latentspace,
    PCA,
    fitted_YieldCurve,

    # utilities.jl
    getindex,
    mean,
    median,
    std,
    var,
    quantile

end
