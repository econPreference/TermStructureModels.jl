## Extract Posterior Samples of Specific Parameters

When users execute some functions, the output is `Vector{<:PosteriorSample}`. That is, some outputs are

- Vector{Parameter}
- Vector{ReducedForm}
- Vector{LatentSpace}
- Vector{YieldCurve}
- Vector{TermPremium}
- Vector{Forecast}

In this case, you can call posterior samples of a specific parameter by using [`getindex`](@ref). For example, if we want to get posterior samples of `phi`, do

```julia
samples_phi = saved_params[:phi]
```

for `saved_params::Vector{Parameter}`, the output of `posterior_sampler`. Then, `samples_phi` is a vector, and `samples_phi[i]` is the i-th posterior sample of `phi`. Note that `samples_phi[i]` is a matrix in this case.(Julialang allows Vector to have Array elements.)

!!! tip

    To get posterior samples or posterior descriptive statistics of a specific parameter, we need to know which `struct` contains the parameter. Page [Notations](https://econpreference.github.io/TermStructureModels.jl/dev/notations/) organize which structs contain the parameter. Also, refer to the documentation of each `struct`.

## Descriptive Statistics of the Posterior Distributions

We extend `mean`, `var`, `std`, `median`, and `quantile` from [Statistics.jl](https://github.com/JuliaStats/Statistics.jl) to `Vector{<:PosteriorSample}`. `Vector{<:PosteriorSample}` includes

- Vector{Parameter}
- Vector{ReducedForm}
- Vector{LatentSpace}
- Vector{YieldCurve}
- Vector{TermPremium}
- Vector{Forecast}

For example, the posterior mean of `phi` can be calculated by

```julia
mean_phi = mean(saved_params)[:phi]
```

`mean_phi[i,j]` is the posterior mean of the entry in the i-th row and j-th column of `phi`. Outputs of all functions(`mean`, `var`, `std`, `median`, and `quantile`) have the same shapes as their corresponding parameters. `quantile` needs the second input. For example, in the case of

```julia
q_phi = quantile(saved_params, 0.4)[:phi]
```

40% of posterior samples for phi[i,j] are less than `q_phi[i,j]`.

## Inference for Parameters

You can get posterior samples of term structure model parameters using [`reducedform`](@ref).

```julia
reduced_params = reducedform(saved_params, yields, macros, tau_n; data_scale=1200)
```

`yields` is a `T` by `N` matrix, and `T` is the length of the time period. `N` is the number of maturities in data. `tau_n` is a `N`-Vector that contains maturities in data. For example, if there are two maturities, 3 and 24 months, in a monthly term structure model, `tau_n=[3; 24]`. `macros` is a `T` by `dP-dQ` matrix in which each column is an individual macroeconomic variable.

!!! note "Reason Why we have to run `reducedform` in addition to `posterior_sampler`"

    We estimate the $\mathbb{P}$-VAR by transforming it into a recursive VAR form. Therefore, `Parameter`, the output of `posterior_sampler`, contains parameters from the recursive VAR. In contrast, `ReducedForm`, the output of `reducedform`, contains parameters in the original reduced-form $\mathbb{P}$-VAR.

## Yield Curve Interpolation

We first have to transform the parameter space from the principal component space to the latent factor space. It is done by [`latentspace`](@ref). And then, use [`fitted_YieldCurve`](@ref) to get fitted yields on the yield curve. Specifically,

```julia
saved_latent_params = latentspace(saved_params, yields, tau_n; data_scale=1200)
fitted_yields = fitted_YieldCurve(τ0, saved_latent_params::Vector{LatentSpace}; data_scale=1200)
```

`τ0` is a Vector containing the maturities for which we want to calculate fitted yields through interpolation.

## Term Premiums

[`term_premium`](@ref) calculate the term premium of `τ`-maturity bond. `τ` should be a scalar.

```julia
saved_TP = term_premium(τ, tau_n, saved_params, yields, macros; data_scale=1200)
```
