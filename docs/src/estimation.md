# Estimation

To estimate the model, the following two steps must be undertaken.

## Step 1. Tuning Hyperparameters

We have five hyperparameters, `p`, `q`, `nu0`, `Omega0`, and `mean_phi_const`.

- `p::Float64`: lag length of the $\mathbb{P}$-VAR(p)
- `q::Matrix{Float64}( , 4, 2)`: Shrinkage degrees in the Minnesota prior
- `nu0::Float64`(d.f.) and `Omega0::Vector`(diagonals of the scale matrix): Prior distribution of the error covariance matrix in the $\mathbb{P}$-VAR(p)
- `mean_phi_const`: Prior mean of the intercept term in the $\mathbb{P}$-VAR(p)

We recommend [`tuning_hyperparameter`](@ref) for deciding the hyperparameters.

```julia
tuned, results = tuning_hyperparameter(yields, macros, tau_n, rho;
                                        populationsize=50,
                                        maxiter=10_000,
                                        medium_tau=collect(24:3:48),
                                        upper_q=[1 1; 1 1; 10 10; 100 100],
                                        mean_kQ_infty=0,
                                        std_kQ_infty=0.1,
                                        upper_nu0=[],
                                        mean_phi_const=[],
                                        fix_const_PC1=false,
                                        upper_p=18,
                                        mean_phi_const_PC1=[],
                                        data_scale=1200,
                                        medium_tau_pr=[],
                                        init_nu0=[])
```

Note that the default upper bound of `p` is `upper_p=18`. The output `tuned::Hyperparameter` is the object that needs to be obtained in Step 1. `results` contains the optimization results.

If users accept our default values, the function is simplified, that is

```julia
tuned, results = tuning_hyperparameter(yields, macros, tau_n, rho)
```

`yields` is a `T` by `N` matrix, and `T` is the length of the sample period. `N` is the number of maturities in data. `tau_n` is a `N`-Vector that contains bond maturities in data. For example, if there are two maturities, 3 and 24 months, in the monthly term structure model, `tau_n=[3; 24]`. `macros` is a `T` by `dP-dQ` matrix in which each column is an individual macroeconomic variable. `rho` is a `dP-dQ`-Vector. In general, `rho[i] = 1` if `macros[:, i]` is in level, or it is set to 0 if the macro variable is differenced.

!!! note "Yield-Only Model"

    Users may want to use yield-only models in which `macros` is an empty set. In such instances, set `macros = []` for all functions.

!!! tip "Computational Cost of the Optimization"

    Since we adopt the Differential Evolutionary(DE) algorithm (Specifically, [`BlackBoxOptim.jl`](https://github.com/robertfeldt/BlackBoxOptim.jl)), it is hard to set the terminal condition. Our strategy was to run the algorithm with a sufficient number of iterations (our default settings) and to verify that it reaches a global optimum by plotting the objective function.

    The reason for using `BlackBoxOptim.jl` is that this package was the most suitable for our model. After trying several optimization packages in Python and Julia, `BlackBoxOptim.jl` consistently found the optimum values most reliably. A downside of DE algorithms like `BlackBoxOptim.jl` is that they can have high computational costs. If the computational cost is excessively high to you, you can reduce it by setting `populationsize` or `maxiter` options in `tuning_hyperparameter` to lower values. However, this may lead to a decrease in model performance.

!!! note "Range of Data over which the Marginal Likelihood is Calculated"

    In Bayesian methodology, the standard criterion of the model comparison is the marginal likelihood. When we compare models using the marginal likelihood, the most crucial prerequisite is that the marginal likelihoods of all models must be calculated over the same observations.

    For instance, let's say we have `data` with the number of rows being 100. Model 1 has `p=1`, and Model 2 has `p=2`. In this case, the marginal likelihood should be computed over `data[3:end, :]`. This means that for Model 1, `data[2, :]` is used as the initial value, and for Model 2, `data[1:2, :]` is used as initial values. `tuning_hyperparameter` automatically reflects this fact by calculating the marginal likelihood over `data[upper_p+1:end, :]` for model comparison.

## Step 2. Sampling the Posterior Distribution of Parameters

In Step 1, we got `tuned::Hyperparameter`. [`posterior_sampler`](@ref) uses it for the estimation.

```julia
saved_params, acceptPrMH = posterior_sampler(yields, macros, tau_n, rho, iteration, tuned::Hyperparameter;
                                            medium_tau=collect(24:3:48),
                                            init_param=[],
                                            ψ=[],
                                            ψ0=[],
                                            gamma_bar=[],
                                            medium_tau_pr=[],
                                            mean_kQ_infty=0,
                                            std_kQ_infty=0.1,
                                            fix_const_PC1=false,
                                            data_scale=1200)
```

If users changed the default values in Step 1, the corresponding default values in the above function also should be changed. If users use our default values, the function simplifies to

```julia
saved_params, acceptPrMH = posterior_sampler(yields, macros, tau_n, rho, iteration, tuned::Hyperparameter)
```

`iteration` is the number of posterior samples that users want to get. Our MCMC starts at the prior mean, and you have to erase burn-in samples manually.

`saved_params::Vector{Parameter}` has a length of `iteration` and each entry is a posterior sample. `acceptPrMH` is dQ-Vector, and the i-th entry shows the MH acceptance rate for i-th principal component in the recursive $\mathbb{P}$-VAR.

After users get posterior samples(`saved_params`), they might want to discard some samples as burn-in. If the number of burn-in samples is `burnin`, run

```julia
saved_params = saved_params[burnin+1:end]
```

Also, users might want to erase posterior samples that do not satisfies the stationary condition. It can be done by [`erase_nonstationary_param`](@ref).

```julia
saved_params, Pr_stationary = erase_nonstationary_param(saved_params)
```

All entries in the above `saved_params::Vector{Parameter}` are posterior samples that satisfy the stationary condition.

!!! warning "Reduction in the Number of Posterior Samples"

    The vector length of `saved_params` decreases after the burn-in process and `erase_nonstationary_param`. Note that this leads to a gap between `iteration` and `length(saved_params)`.

## Diagnostics for MCMC

We believe in the efficiency of our algorithm, so users do not need to be overly concerned about the convergence of the posterior samples. In our opinion, sampling 6,000 posterior samples and erase the first 1,000 samples as burn-in would be enough.

We provide [a measure](https://econpreference.github.io/TermStructureModels.jl/dev/api/#TermStructureModels.ineff_factor-Tuple{Any}) to gauge the efficiency of the algorithm, that is

```julia
ineff = ineff_factor(saved_params)
```

`saved_params::Vector{Parameter}` is the output of `posterior_sampler`. `ineff` is `Tuple(kappaQ, kQ_infty, gamma, SigmaO, varFF, phi)`. Each object in the tuple has the same shape as its corresponding parameter. The entries of the `Array` in the `Tuple` represent the inefficiency factors of the corresponding parameters. If an inefficiency factor is high, it indicates poor sampling efficiency of the parameter located at the same position.

You can calculate the maximum inefficiency factor by

```julia
max_ineff = (ineff[1], ineff[2], ineff[3] |> maximum, ineff[4] |> maximum, ineff[5] |> maximum, ineff[6] |> maximum) |> maximum
```

The value obtained by dividing the number of posterior samples by `max_ineff` is the effective number of posterior samples, taking into account the efficiency of the sampler. For example, let's say `max_ineff = 10`. Then, if 6,000 posterior samples are drawn and the first 1,000 samples are erased as burn-in, the remaining 5,000 posterior samples have the same efficiency as using 500 i.i.d samples, calculated as `(6000-1000)/max_ineff`. For reference, in [our paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4708628), the maximum inefficiency factor was `2.38`.
