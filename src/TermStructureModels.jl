module TermStructureModels

using Base: @kwdef
using LinearAlgebra, Statistics, Distributions, SpecialFunctions, ProgressMeter, Distributed, Random, Roots, BlackBoxOptim, Optim, LineSearches, Turing, MCMCChains
import Base: getindex
import Statistics: mean, median, std, var, quantile
import AdvancedHMC, AxisArrays

"""
    @kwdef struct Hyperparameter
- `p::Int`
- `q::Matrix`
- `nu0`
- `Omega0::Vector`
- `mean_phi_const::Vector = zeros(length(Omega0))`: This is the prior mean of the constant term in the VAR.
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
This is a super-set of structs `Parameter`, `ReducedForm`, `LatentSpace`, `YieldCurve`, `TermPremium`, `Forecast`.
"""
abstract type PosteriorSample end

"""
    @kwdef struct Parameter <: PosteriorSample
This struct contains the statistical parameters of the model that are sampled from function `posterior_sampler`.
- `kappaQ`
- `kQ_infty::Float64`
- `phi::Matrix{Float64}`
- `varFF::Vector{Float64}`
- `SigmaO::Vector{Float64}`
- `gamma::Vector{Float64}`
"""
@kwdef struct Parameter <: PosteriorSample
    kappaQ
    kQ_infty::Float64
    phi::Matrix{Float64}
    varFF::Vector{Float64}
    SigmaO::Vector{Float64}
    gamma::Vector{Float64}
end

"""
    @kwdef struct Parameter_NUTS <: PosteriorSample
This struct contains the statistical parameters of the model that are sampled from function `posterior_NUTS`.
- `q`
- `nu0`
- `kappaQ`
- `kQ_infty::Float64`
- `phi::Matrix{Float64}`
- `varFF::Vector{Float64}`
- `SigmaO::Vector{Float64}`
- `gamma::Vector{Float64}`
"""
@kwdef struct Parameter_NUTS <: PosteriorSample
    q
    nu0
    kappaQ
    kQ_infty::Float64
    phi::Matrix{Float64}
    varFF::Vector{Float64}
    SigmaO::Vector{Float64}
    gamma::Vector{Float64}
end

"""
    @kwdef struct ReducedForm <: PosteriorSample
This struct contains the statistical parameters in terms of the reduced form VAR(p) in P-dynamics. `lambdaP` and `LambdaPF` are parameters in the market prices of risks equation, and they only contain the first `dQ` non-zero equations. 
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
This struct contains the fitted yield curve. `yields[t,:] = intercept + slope*latents[t,:]` holds.
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
The yields are decomposed into the term premium(`TP`) and the expectation hypothesis component(`EH`). Each component has constant terms(`const_TP` and `const_EH`) and time-varying components(`timevarying_TP` and `timevarying_EH`). `factorloading_EH` and `factorloading_TP` are coefficients of the pricing factors for the time varying components. Each column of the outputs indicates the results for each maturity.

The time-varying components are not stored in `TermPremium`, and they are the separate outputs in function [`term_premium`](@ref). 
- TP
- EH
- factorloading_TP
- factorloading_EH
- const_TP
- const_EH
"""
@kwdef struct TermPremium <: PosteriorSample
    TP
    EH
    factorloading_TP
    factorloading_EH
    const_TP
    const_EH
end

"""
    @kwdef struct Scenario
This struct contains scenarios to be conditioned in the scenario analysis. When `y = [yields; macros]` is an observed vector in the measurement equation, `Scenario.combinations*y = Scenario.values` constitutes the scenario at a specific time. `Vector{Scenario}` is used to describe a time-series of scenarios.

`combinations` and `values` should be `Matrix` and `Vector`. If `values` is a scalar, `combinations` would be a matrix with one row vector and `values` should be one-dimensional vector, for example [values]. 
- `combinations::Matrix`
- `values::Vector`
"""
@kwdef struct Scenario
    combinations::Matrix
    values::Vector
end

"""
    @kwdef struct Forecast <: PosteriorSample
This struct contains the results of the scenario analysis, the conditional prediction for yields, `factors = [PCs macros]`, and term premiums.
- `yields`
- `factors`
- `TP`: term premium forecasts
- `EH`: estimated expectation hypothesis component
"""
@kwdef struct Forecast <: PosteriorSample
    yields
    factors
    TP
    EH
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
    Parameter_NUTS,
    ReducedForm,
    LatentSpace,
    YieldCurve,
    TermPremium,
    Scenario,
    Forecast,

    # inference.jl
    tuning_hyperparameter,
    tuning_hyperparameter_with_vs,
    AR_res_var,
    generative,
    ineff_factor,
    posterior_sampler,
    posterior_NUTS,

    # priors.jl
    prior_kappaQ,
    dcurvature_dτ,

    # scenario.jl
    conditional_forecast,
    conditional_expectation,

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
    quantile,
    hessian

end
