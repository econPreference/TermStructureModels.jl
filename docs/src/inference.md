# Statistical Inference

## Inference for Parameters

You can get posterior samples of the term structure model parameters using [`reducedform`](@ref).

```julia
reduced_params = reducedform(saved_params, yields, macros, tau_n; data_scale=1200)
```

`yields` is a `T` by `N` matrix, and `T` is the length of the sample period. `N` is the number of bond maturities in data. `tau_n` is a `N`-Vector that contains maturities in data. For example, if there are two maturities, 3 and 24 months, in the monthly term structure model, `tau_n=[3; 24]`. `macros` is a `T` by `dP-dQ` matrix in which each column is an individual macroeconomic variable.

!!! note "Reason Why We have to Run `reducedform` in addition to `posterior_sampler`"

    We estimate the $\mathbb{P}$-VAR by transforming it into a recursive VAR form. Therefore, `Parameter`, the output of `posterior_sampler`, contains parameters in the recursive VAR. In contrast, `ReducedForm`, the output of `reducedform`, contains parameters in the original reduced-form $\mathbb{P}$-VAR.

## Yield Curve Interpolation

We first have to transform the parameter space from the principal component space to the latent factor space. It is done by [`latentspace`](@ref). And then, use [`fitted_YieldCurve`](@ref) to get fitted yields. Specifically,

```julia
saved_latent_params = latentspace(saved_params, yields, tau_n; data_scale=1200)
fitted_yields = fitted_YieldCurve(τ0, saved_latent_params::Vector{LatentSpace}; data_scale=1200)
```

`τ0` is a Vector containing the maturities for which we want to calculate fitted yields through interpolation.

## Term Premiums

[`term_premium`](@ref) calculates the term premium of `τ`-maturity bond. `τ` should be a scalar.

```julia
saved_TP = term_premium(τ, tau_n, saved_params, yields, macros; data_scale=1200)
```
