# Statistical Inference

## Inference for Parameters

You can obtain posterior samples of the term structure model parameters using [`reducedform`](@ref).

```julia
reduced_params = reducedform(saved_params, yields, macros, tau_n; data_scale=1200, pca_loadings=[])
```

`yields` is a `T` by `N` matrix, `T` is the length of the sample period and `N` is the number of bond maturities in the data. `tau_n` is an `N`-Vector that contains maturities in the data. For example, if there are two maturities, 3 and 24 months, in the monthly term structure model, `tau_n=[3; 24]`. `macros` is a `T` by `dP-dQ` matrix in which each column represents an individual macroeconomic variable.

!!! note "Reason Why You Need to Run `reducedform` in Addition to `posterior_sampler`"

    We estimate the $\mathbb{P}$-VAR by transforming it into a recursive VAR form. Therefore, `Parameter`, the output of `posterior_sampler`, contains parameters in the recursive VAR. In contrast, `ReducedForm`, the output of `reducedform`, contains parameters in the original reduced-form $\mathbb{P}$-VAR.

Each entry in `reduced_params::Vector{ReducedForm}` is a joint posterior sample of the parameters.

## Yield Curve Interpolation

First, transform the parameter space from the principal component space to the latent factor space. This is done using [`latentspace`](@ref). Then, use [`fitted_YieldCurve`](@ref) to obtain fitted yields. Specifically,

```julia
saved_latent_params = latentspace(saved_params, yields, tau_n; data_scale=1200, pca_loadings=[])
fitted_yields = fitted_yieldcurve(tau_vec, saved_latent_params::Vector{LatentSpace}; data_scale=1200)
```

`tau_vec` is a vector containing the maturities for which you want to calculate fitted yields through interpolation. `fitted_yields::Vector{YieldCurve}` contains the interpolation results.

## Term Premiums

[`term_premium`](@ref) calculates the term premium of the bonds. `tau_interest` contains the maturities of interest and should be a `Vector` (at least a one-dimensional vector).

```julia
saved_TP, saved_tv_TP, saved_tv_EH = term_premium(tau_interest, tau_n, saved_params, yields, macros; data_scale=1200)
```

`saved_TP::Vector{TermPremium}` contains the results of the term premium calculations. Both the term premiums and expectation hypothesis components are decomposed into time-invariant and time-varying parts. For the maturity `tau_interest[i]`, the time-varying parts are saved in `saved_tv_TP[:, :, i]` and `saved_tv_EH[:, :, i]`. The time-varying parts driven by the `j`-th pricing factor are stored in `saved_tv_TP[:, j, i]` and `saved_tv_EH[:, j, i]`.
