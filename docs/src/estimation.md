## Step 1. Tuning Hyperparameters

We have five hyperparameters, `p`, `q`, `nu0`, `Omega0`, and `mean_phi_const`.

- `p::Float64`: lag length of the $\mathbb{P}$-VAR(p)
- `q::Matrix{Float64}( , 4, 2)`: Shrinkage degrees in the Minnesota prior
- `nu0::Float64`(d.f.) and `Omega0::Vector`(scale matrix): Prior distribution of the error covariance matrix in the $\mathbb{P}$-VAR(p)
- `mean_phi_const`: Prior mean of the intercept term in the $\mathbb{P}$-VAR(p)

We recommend [`tuning_hyperparameter`](https://econpreference.github.io/TermStructureModels.jl/dev/api/#TermStructureModels.tuning_hyperparameter-NTuple{4,%20Any}) for deciding the hyperparameters.

```julia
tuned, results = tuning_hyperparameter(yields, macros, tau_n, rho; populationsize=50, maxiter=10_000, medium_tau=collect(24:3:48), upper_q=[1 1; 1 1; 10 10; 100 100], mean_kQ_infty=0, std_kQ_infty=0.1, upper_nu0=[], mean_phi_const=[], fix_const_PC1=false, upper_p=18, mean_phi_const_PC1=[], data_scale=1200, medium_tau_pr=[], init_nu0=[])
```

Note that the default upper bound of `p` is `upper_p=18`. `tuned::Hyperparameter` is the one we need for the estimation.

If users accept our default values, the function simplifies, that is

```juila
tuned, results = tuning_hyperparameter(yields, macros, tau_n, rho)
```

`yields` is a T by N matrix, and T is the length of the time period. N is the number of maturities in data. `tau_n` is a N-Vector that contains maturities in data. For example, if there are two maturities, 3 and 24 months, in a monthly term structure model, `tau_n=[3; 24]`. `macros` is a T by (dP-dQ) matrix in which each column is an individual macroeconomic variable. `rho` is a (dP-dQ)-Vector. In general, the i-th entry in `rho` is `1` if i-th macro variable is in level, or it is set to 0 if the variable is differenced.

!!! note "Computational length of the optimization"

    Since we adopt the Differential Evolutionary algorithm, it is hard to set the terminal condition. Our strategy was "Run the algorithm with sufficient `maxiter`(our defaults), and verify that it is an global optimum by plotting the objective function". It is appropriate for academic projects.

    However, it is not good for practical projects. small `populationsize` or `maxiter` may not lead to the best model, but it will find a good model. The prior distribution does not need to be the best form. Set `maxiter` based on your computational resources.

## Step 2. Sampling the Posterior Distribution of Parameters

In Step 1, we got `tuned::Hyperparameter`. [`posterior_sampler`](https://econpreference.github.io/TermStructureModels.jl/dev/api/#TermStructureModels.posterior_sampler-Tuple{Any,%20Any,%20Any,%20Any,%20Any,%20Hyperparameter}) uses it for the estimation.

```juila
saved_params, acceptPrMH = posterior_sampler(yields, macros, tau_n, rho, iteration, tuned::Hyperparameter; medium_tau=collect(24:3:48), init_param=[], ψ=[], ψ0=[], gamma_bar=[], medium_tau_pr=[], mean_kQ_infty=0, std_kQ_infty=0.1, fix_const_PC1=false, data_scale=1200)
```

If users changed the default values in Step 1, the corresponding default values in the above function also should be changed. If users use our defaults, the function simplifies to

```juila
saved_params, acceptPrMH = posterior_sampler(yields, macros, tau_n, rho, iteration, tuned::Hyperparameter)
```

`iteration` is the number of posterior samples that users want to get. Our MCMC starts at the prior mean, and you have to erase burn-in samples manually.

Output `saved_params` is a Vector that has a length of `iteration`. Each entry is struct `Parameter`. `acceptPrMH` is dQ-Vector, and i-th entry shows the HM acceptance rate for i-th principal component in the recursive $\mathbb{P}$-VAR.
