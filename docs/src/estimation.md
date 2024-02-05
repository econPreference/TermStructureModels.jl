## Step 1. Tuning Hyperparameters

We have five hyperparameters, `p`, `q`, `nu0`, `Omega0`, and `mean_phi_const`.

- `p::Float64`: lag length of the $\mathbb{P}$-VAR(p)
- `q::Matrix{Float64}( , 4, 2)`: Shrinkage degrees in the Minnesota prior
- `nu0::Float64`(d.f.) and `Omega0::Vector`(scale matrix): Prior distribution of the error covariance matrix in the $\mathbb{P}$-VAR(p)
- `mean_phi_const`: Prior mean of the intercept term in the $\mathbb{P}$-VAR(p)

We recommend [`tuning_hyperparameter`](@ref) for deciding the hyperparameters.

```julia
tuned, results = tuning_hyperparameter(yields, macros, tau_n, rho; populationsize=50, maxiter=10_000, medium_tau=collect(24:3:48), upper_q=[1 1; 1 1; 10 10; 100 100], mean_kQ_infty=0, std_kQ_infty=0.1, upper_nu0=[], mean_phi_const=[], fix_const_PC1=false, upper_p=18, mean_phi_const_PC1=[], data_scale=1200, medium_tau_pr=[], init_nu0=[])
```

Note that the default upper bound of `p` is `upper_p=18`. `tuned::Hyperparameter` is the one we need for the estimation.

If users accept our default values, the function simplifies, that is

```julia
tuned, results = tuning_hyperparameter(yields, macros, tau_n, rho)
```

`yields` is a `T` by `N` matrix, and `T` is the length of the time period. `N` is the number of maturities in data. `tau_n` is a `N`-Vector that contains maturities in data. For example, if there are two maturities, 3 and 24 months, in a monthly term structure model, `tau_n=[3; 24]`. `macros` is a `T` by `dP-dQ` matrix in which each column is an individual macroeconomic variable. `rho` is a `dP-dQ`-Vector. In general, the i-th entry in `rho` is `1` if i-th macro variable is in level, or it is set to 0 if the variable is differenced.

!!! note "Computational length of the optimization"

    Since we adopt the Differential Evolutionary algorithm, it is hard to set the terminal condition. Our strategy was "Run the algorithm with sufficient `maxiter`(our defaults), and verify that it is an global optimum by plotting the objective function". It is appropriate for academic projects.

    However, it is not good for practical projects. Small `populationsize` or `maxiter` may not lead to the best model, but it will find a good model. The prior distribution does not need to be the best form. Set `maxiter` based on your computational resources.

!!! tip "Normalization of Data"

    Our package demeans the principal components, which are risk factors in the bond market. Therefore, we recommend using macro data after demeaning it.

## Step 2. Sampling the Posterior Distribution of Parameters

In Step 1, we got `tuned::Hyperparameter`. [`posterior_sampler`](@ref) uses it for the estimation.

```julia
saved_params, acceptPrMH = posterior_sampler(yields, macros, tau_n, rho, iteration, tuned::Hyperparameter; medium_tau=collect(24:3:48), init_param=[], ψ=[], ψ0=[], gamma_bar=[], medium_tau_pr=[], mean_kQ_infty=0, std_kQ_infty=0.1, fix_const_PC1=false, data_scale=1200)
```

If users changed the default values in Step 1, the corresponding default values in the above function also should be changed. If users use our defaults, the function simplifies to

```julia
saved_params, acceptPrMH = posterior_sampler(yields, macros, tau_n, rho, iteration, tuned::Hyperparameter)
```

`iteration` is the number of posterior samples that users want to get. Our MCMC starts at the prior mean, and you have to erase burn-in samples manually.

Output `saved_params` is a Vector that has a length of `iteration`. Each entry is struct `Parameter`. `acceptPrMH` is dQ-Vector, and i-th entry shows the HM acceptance rate for i-th principal component in the recursive $\mathbb{P}$-VAR.

## Diagnostics for MCMC

We believe in the efficiency of our algorithm, so users do not need to be overly concerned about the convergence of the posterior samples. In our opinion, sampling 6,000 posterior samples and erase the first 1,000 samples as burn-in would be enough.

We provide [a measure](@ref) to gauge the efficiency of the algorithm, that is

```julia
ineff = ineff_factor(saved_params)
```

`saved_params::Vector{Parameter}` is the output of `posterior_sampler`. `ineff` is `Tuple(kappaQ, kQ_infty, gamma, SigmaO, varFF, phi)`. Each object in the tuple has the same shape as its corresponding parameter. If an inefficiency factor is high, it indicates poor sampling efficiency for the parameter located at the same position.

You can calculate the maximum inefficiency factor by

```julia
max_ineff = (ineff[1], ineff[2], ineff[3] |> maximum, ineff[4] |> maximum, ineff[5] |> maximum, ineff[6] |> maximum) |> maximum
```

Dividing the total number of posterior samples by `max_ineff` allows for the calculation of the effective number of posterior samples, taking into account the efficiency of the sampler. For example, let's say `max_ineff = 10`. Then, if 6,000 posterior samples are drawn and the first 1,000 samples are erased as burn-in, the remaining 5,000 posterior samples have the same efficiency as using 500 i.i.d samples, calculated as `(6000-1000)/max_ineff`. For reference, in [our paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4708628), the maximum inefficiency factor was `2.38`.
