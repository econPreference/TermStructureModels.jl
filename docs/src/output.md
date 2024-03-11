# How to Utilize the Outputs of Functions

## Extract Posterior Samples

When users execute some functions, the output is [`Vector{<:PosteriorSample}`](https://econpreference.github.io/TermStructureModels.jl/dev/api/#TermStructureModels.PosteriorSample). That is, some outputs are

- [`Vector{Parameter}`](https://econpreference.github.io/TermStructureModels.jl/dev/api/#TermStructureModels.Parameter)
- [`Vector{ReducedForm}`](https://econpreference.github.io/TermStructureModels.jl/dev/api/#TermStructureModels.ReducedForm)
- [`Vector{LatentSpace}`](https://econpreference.github.io/TermStructureModels.jl/dev/api/#TermStructureModels.LatentSpace)
- [`Vector{YieldCurve}`](https://econpreference.github.io/TermStructureModels.jl/dev/api/#TermStructureModels.YieldCurve)
- [`Vector{TermPremium}`](https://econpreference.github.io/TermStructureModels.jl/dev/api/#TermStructureModels.TermPremium)
- [`Vector{Forecast}`](https://econpreference.github.io/TermStructureModels.jl/dev/api/#TermStructureModels.Forecast)

In this case, you can call posterior samples of a specific parameter by using [`getindex`](@ref). For example, if we want to get posterior samples of `phi`, run

```julia
samples_phi = saved_params[:phi]
```

for `saved_params::Vector{Parameter}`, the output of [`posterior_sampler`](https://econpreference.github.io/TermStructureModels.jl/dev/estimation/#Step-2.-Sampling-the-Posterior-Distribution-of-Parameters). Then, `samples_phi` is a vector, and `samples_phi[i]` is the i-th posterior sample of `phi`. Note that `samples_phi[i]` is a matrix in this case. (Julialang allows Vector to have Array elements.)

## Descriptive Statistics of the Posterior Distributions

We extend [`mean`](@ref), [`var`](@ref), [`std`](@ref), [`median`](@ref), and [`quantile`](@ref) from [Statistics.jl](https://github.com/JuliaStats/Statistics.jl) to [`Vector{<:PosteriorSample}`](https://econpreference.github.io/TermStructureModels.jl/dev/api/#TermStructureModels.PosteriorSample). [`Vector{<:PosteriorSample}`](https://econpreference.github.io/TermStructureModels.jl/dev/api/#TermStructureModels.PosteriorSample) includes

- [`Vector{Parameter}`](https://econpreference.github.io/TermStructureModels.jl/dev/api/#TermStructureModels.Parameter)
- [`Vector{ReducedForm}`](https://econpreference.github.io/TermStructureModels.jl/dev/api/#TermStructureModels.ReducedForm)
- [`Vector{LatentSpace}`](https://econpreference.github.io/TermStructureModels.jl/dev/api/#TermStructureModels.LatentSpace)
- [`Vector{YieldCurve}`](https://econpreference.github.io/TermStructureModels.jl/dev/api/#TermStructureModels.YieldCurve)
- [`Vector{TermPremium}`](https://econpreference.github.io/TermStructureModels.jl/dev/api/#TermStructureModels.TermPremium)
- [`Vector{Forecast}`](https://econpreference.github.io/TermStructureModels.jl/dev/api/#TermStructureModels.Forecast)

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

    To get posterior samples or posterior descriptive statistics of a specific parameter, we need to know which `struct` contains the parameter. Page [Notations](https://econpreference.github.io/TermStructureModels.jl/dev/notations/) organize which structs contain the parameter. Also, refer to the documentation of each `struct`.
