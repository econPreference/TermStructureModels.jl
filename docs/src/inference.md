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

First, transform the parameter space from the principal component space to the latent factor space. This is done using [`latentspace`](@ref). Then, use [`fitted_yieldcurve`](@ref) to obtain fitted yields. Specifically,

```julia
saved_latent_params = latentspace(saved_params, yields, tau_n; data_scale=1200, pca_loadings=[])
fitted_yields = fitted_yieldcurve(tau_vec, saved_latent_params::Vector{LatentSpace}; data_scale=1200)
```

`tau_vec` is a vector containing the maturities for which you want to calculate fitted yields through interpolation. `fitted_yields::Vector{YieldCurve}` contains the interpolation results.

## Term Premiums

[`term_premium`](@ref) calculates the term premium of the bonds. `tau_interest` contains the maturities of interest and should be a `Vector` (at least a one-dimensional vector), in strictly increasing order without duplicates.

```julia
saved_TP, saved_tv_TP, saved_tv_EH = term_premium(tau_interest, tau_n, saved_params, yields, macros; data_scale=1200)
```

`yields` and `macros` are the data used to estimate `saved_params`. If the yield curves to decompose are the same as `yields`, leave both `decomp_yields` and `decomp_macros` as `[]`. To decompose another yield curve dataset while keeping the estimated model fixed, supply `decomp_yields` and, if the model includes macro variables, `decomp_macros`. For example:

```julia
saved_TP, saved_tv_TP, saved_tv_EH = term_premium(tau_interest, tau_n, saved_params, yields, macros;
    data_scale=1200, decomp_yields=new_yields, decomp_macros=new_macros)
```

`decomp_yields` may have a different number of observations and a different observation frequency, but must use the same yield units/scaling and the same maturity columns in the same order as `yields` and `tau_n`. The PCA rotation, ordering, signs, and centering are determined from the estimation sample `yields`; the new yields are projected through that fixed transformation. The estimated model frequency, maturity units, `data_scale`, and P- and Q-dynamics remain unchanged. External decomposition data are supported only for P-dynamics with `p=1`; outputs exclude the first observation of that sample.

When supplying `decomp_yields`, also supply `decomp_macros` if the model includes macro variables; otherwise leave `decomp_macros` empty. Its rows must match `decomp_yields`, and its columns must match the number and order of variables in `macros`. It cannot be supplied without `decomp_yields`.

`saved_TP::Vector{TermPremium}` contains the results of the term premium calculations. Both the term premiums and expectation hypothesis components are decomposed into time-invariant and time-varying parts. For the maturity `tau_interest[i]`, the time-varying parts are saved in `saved_tv_TP[:, :, i]` and `saved_tv_EH[:, :, i]`. The time-varying parts driven by the `j`-th pricing factor are stored in `saved_tv_TP[:, j, i]` and `saved_tv_EH[:, j, i]`.
