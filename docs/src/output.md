# How to Utilize the Outputs of Functions

When users execute some functions, the output is [`Vector{<:PosteriorSample}`](https://econpreference.github.io/TermStructureModels.jl/dev/api/#TermStructureModels.PosteriorSample). The examples of `Vector{<:PosteriorSample}` are

- [`Vector{Parameter}`](https://econpreference.github.io/TermStructureModels.jl/dev/api/#TermStructureModels.Parameter): output of [`posterior_sampler`](@ref)
- [`Vector{ReducedForm}`](https://econpreference.github.io/TermStructureModels.jl/dev/api/#TermStructureModels.ReducedForm): output of [`reducedform`](@ref)
- [`Vector{LatentSpace}`](https://econpreference.github.io/TermStructureModels.jl/dev/api/#TermStructureModels.LatentSpace): output of [`latentspace`](@ref)
- [`Vector{YieldCurve}`](https://econpreference.github.io/TermStructureModels.jl/dev/api/#TermStructureModels.YieldCurve): output of [`fitted_YieldCurve`](@ref)
- [`Vector{TermPremium}`](https://econpreference.github.io/TermStructureModels.jl/dev/api/#TermStructureModels.TermPremium): output of [`term_premium`](@ref)
- [`Vector{Forecast}`](https://econpreference.github.io/TermStructureModels.jl/dev/api/#TermStructureModels.Forecast): outputs of [`conditional_forecasts`](@ref) and [`scenario_analysis`](@ref)

Each entry of the above vectors is a posterior sample and has the form of a `struct`(`Parameter`, `ReducedForm`, `LatentSpace`, `YieldCurve`, `TermPremium`, `Forecast`). The above six `struct` have their unique fields. See [the API section](https://econpreference.github.io/TermStructureModels.jl/dev/api/#API-documentation) to see what fields each `struct` contains.

## Extract Posterior Samples of fields

`Vector{<:PosteriorSample}` contains posterior samples of the fields of the corresponding `struct`. You can call posterior samples of a specific field by using [`getindex`](@ref). For example, if we want to get posterior samples of `phi` in `Parameter`, run

```julia
samples_phi = saved_params[:phi]
```

for `saved_params::Vector{Parameter}`. Then, `samples_phi` is a vector, and `samples_phi[i]` is the i-th posterior sample of `phi`. Note that `samples_phi[i]` is a matrix in this case. (Julialang allows Vector to have Array elements.)

## Descriptive Statistics of the Posterior Distributions

We extend [`mean`](@ref), [`var`](@ref), [`std`](@ref), [`median`](@ref), and [`quantile`](@ref) from [Statistics.jl](https://github.com/JuliaStats/Statistics.jl) to [`Vector{<:PosteriorSample}`](https://econpreference.github.io/TermStructureModels.jl/dev/api/#TermStructureModels.PosteriorSample).

Therefore, these five functions can be conveniently used to calculate descriptive statistics of the posterior distribution, such as the posterior mean or posterior variance. For example, the posterior mean of `phi` can be calculated by

```julia
mean_phi = mean(saved_params)[:phi]
```

`mean_phi[i,j]` is the posterior mean of the entry in the i-th row and j-th column of `phi`. Outputs of all functions(`mean`, `var`, `std`, `median`, and `quantile`) have the same shapes as their corresponding parameters. `quantile` needs the second input. For example, in the case of

```julia
q_phi = quantile(saved_params, 0.4)[:phi]
```

40% of posterior samples of `phi[i,j]` are less than `q_phi[i,j]`.

!!! tip

    To get posterior samples or posterior descriptive statistics of a specific parameter, we need to know which `struct` contains the parameter. Page [Notations](https://econpreference.github.io/TermStructureModels.jl/dev/notations/) organize which structs contain the parameter. Also, refer to the documentation of each `struct` in [the API section](https://econpreference.github.io/TermStructureModels.jl/dev/api/#API-documentation).
