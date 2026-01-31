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
tuned, results = tuning_hyperparameter(yields, macros, tau_n, rho; populationsize=50, maxiter=10_000, medium_tau=collect(24:3:48), upper_q=[1 1; 1 1; 4 4; 100 100], mean_kQ_infty=0, std_kQ_infty=0.1, upper_nu0=[], mean_phi_const=[], fix_const_PC1=false, upper_p=24, mean_phi_const_PC1=[], data_scale=1200, kappaQ_prior_pr=[], init_nu0=[], is_pure_EH=false, psi_common=[], psi_const=[], pca_loadings=[], prior_mean_diff_kappaQ=[], prior_std_diff_kappaQ=[], optimizer=:LBFGS, ml_tol=1.0, init_x=[])
```

Note that the default upper bound of `p` is `upper_p=24`. The output `tuned::Hyperparameter` is the object that should be obtained in Step 1. `results` contains the optimization results.

If you accept the default values, the function is simplified to

```julia
tuned, results = tuning_hyperparameter(yields, macros, tau_n, rho)
```

`yields` is a `T` by `N` matrix, `T` is the length of the sample period and `N` is the number of maturities in the data. `tau_n` is an `N`-Vector that contains bond maturities in the data. For example, if there are two maturities, 3 and 24 months, in the monthly term structure model, `tau_n=[3; 24]`. `macros` is a `T` by `dP-dQ` matrix in which each column represents an individual macroeconomic variable. `rho` is a `dP-dQ`-Vector. In general, `rho[i] = 1` if `macros[:, i]` is in levels, or it is set to 0 if the macro variable is differenced.

### Several relevant points regarding hyperparameter optimization

#### Optimization Algorithms

We provide two optimization algorithms via the `optimizer` option:

- **`:LBFGS` (default, recommended)**: Uses gradient-based LBFGS optimization from `Optim.jl`. This algorithm alternates between optimizing hyperparameters (with fixed lag) and selecting the best lag (with fixed hyperparameters) until convergence. It is fast and efficient, making it suitable for most applications. However, it does not guarantee finding the global optimum due to its local search nature.

- **`:BBO`**: Uses a Differential Evolutionary (DE) algorithm from [`BlackBoxOptim.jl`](https://github.com/robertfeldt/BlackBoxOptim.jl). This algorithm optimizes hyperparameters and lag simultaneously and is more likely to find the global optimum. The downside is that it has higher computational costs and lacks automatic convergence detection—users must verify convergence by examining the objective function values or setting a sufficient number of iterations via `maxiter`.

We recommend using the default `:LBFGS` optimizer for most cases due to its speed and reliability. If you suspect the optimization is stuck in a local optimum or need to verify global optimality, you can use `:BBO` with the `populationsize` and `maxiter` options adjusted according to your computational budget.

#### Range of Data over which the Marginal Likelihood is Calculated

In Bayesian methodology, the standard criterion for model comparison is the marginal likelihood. When comparing models using the marginal likelihood, the most crucial prerequisite is that the marginal likelihoods of all models must be calculated over the same observations.

For instance, suppose we have `data` with 100 rows. Model 1 has `p=1`, and Model 2 has `p=2`. In this case, the marginal likelihood should be computed over `data[3:end, :]`. This means that for Model 1, `data[2, :]` is used as the initial value, and for Model 2, `data[1:2, :]` are used as initial values. `tuning_hyperparameter` automatically accounts for this by calculating the marginal likelihood over `data[upper_p+1:end, :]` for model comparison.

#### Prior Belief about the Expectation Hypothesis

The algorithm has an inductive bias that the estimates should not deviate too much from the Expectation Hypothesis (EH). Here, the assumed EH means that the term premium is a non-zero constant. If you want to introduce an inductive bias centered around the pure EH, where the term premium is zero, set `is_pure_EH=true`. However, note that using this option may take some additional time to numerically set the prior distribution.

## Step 2. Sampling the Posterior Distribution of Parameters

In Step 1, we obtained `tuned::Hyperparameter`. [`posterior_sampler`](@ref) uses it for the estimation.

```julia
saved_params, acceptPrMH = posterior_sampler(yields, macros, tau_n, rho, iteration, tuned::Hyperparameter; medium_tau=collect(24:3:48), init_param=[], psi=[], psi_const=[], gamma_bar=[], kappaQ_prior_pr=[], mean_kQ_infty=0, std_kQ_infty=0.1, fix_const_PC1=false, data_scale=1200, pca_loadings=[], kappaQ_proposal_mode=[])
```

If you changed the default values in Step 1, the corresponding default values in the above function should also be changed. If you use the default values, the function simplifies to

```julia
saved_params, acceptPrMH = posterior_sampler(yields, macros, tau_n, rho, iteration, tuned::Hyperparameter)
```

`iteration` is the number of posterior samples to generate. The MCMC sampler starts at the prior mean, and you need to discard burn-in samples manually.

`saved_params::Vector{Parameter}` has length `iteration`, and each entry is a posterior sample. `acceptPrMH` is a `dQ+1`-Vector, where the `i(<=dQ)`-th entry shows the MH acceptance rate for the i-th principal component in the recursive $\mathbb{P}$-VAR. The last entry of `acceptPrMH` is the MH acceptance rate for `kappaQ` under the unrestricted JSZ model. It is zero under the AFNS restriction.

## Step 3. Discard Burn-in and Nonstationary Posterior Samples

After obtaining posterior samples (`saved_params`), you may want to discard some samples as burn-in. If the number of burn-in samples is `burnin`, run

```julia
saved_params = saved_params[burnin+1:end]
```

You may also want to discard posterior samples that do not satisfy the stationarity condition. This can be done using [`erase_nonstationary_param`](@ref).

```julia
saved_params, Pr_stationary = erase_nonstationary_param(saved_params; threshold=1)
```

All entries in the output (`saved_params::Vector{Parameter}`) are posterior samples that satisfy the stationarity condition.

!!! warning "Reduction in the Number of Posterior Samples"

    The length of `saved_params` decreases after the burn-in process and applying `erase_nonstationary_param`. Note that this creates a gap between `iteration` and `length(saved_params)`.

!!! note "Handling Non-Stationary Data"

    The optional input `threshold` is designed to discard posterior samples with eigenvalues of the VAR system exceeding the specified threshold. Traditionally, we use a stationary VAR, so the default threshold is set to `1`. However, for non-stationary VAR models, it may be necessary to allow for a slightly higher threshold. In such cases, you can set `threshold` to a value greater than `1`, such as `1.05`.

## Diagnostics for MCMC

We believe in the efficiency of the algorithm, so you do not need to be overly concerned about the convergence of the posterior samples. In our opinion, sampling 6,000 posterior samples and discarding the first 1,000 samples as burn-in should be sufficient.

We provide [a measure](https://econpreference.github.io/TermStructureModels.jl/dev/api/#TermStructureModels.ineff_factor-Tuple{Any}) to gauge the efficiency of the algorithm, that is

```julia
ineff = ineff_factor(saved_params)
```

`saved_params::Vector{Parameter}` is the output of `posterior_sampler`. `ineff` is a `Tuple(kappaQ, kQ_infty, gamma, SigmaO, varFF, phi)`. Each object in the tuple has the same shape as its corresponding parameter. The entries in the `Array` of the `Tuple` represent the inefficiency factors of the corresponding parameters. If an inefficiency factor is high, it indicates poor sampling efficiency for the parameter at that position.

You can calculate the maximum inefficiency factor by

```julia
max_ineff = (ineff[1] |> maximum, ineff[2], ineff[3] |> maximum, ineff[4] |> maximum, ineff[5] |> maximum, ineff[6] |> maximum) |> maximum
```

The value obtained by dividing the number of posterior samples by `max_ineff` is the effective number of posterior samples, accounting for the efficiency of the sampler. For example, suppose `max_ineff = 10`. If 6,000 posterior samples are drawn and the first 1,000 samples are discarded as burn-in, the remaining 5,000 posterior samples have the same efficiency as 500 i.i.d. samples, calculated as `(6000-1000)/max_ineff`.
