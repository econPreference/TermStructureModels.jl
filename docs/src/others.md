# Other Forms of the Model

## Yield-Only Model

Users may want to use yield-only models in which `macros` is an empty set. In such instances, set `macros = []` and `rho = []` for all functions.

## Unrestricted JSZ model

Our model is the three-factor JSZ[(Joslin, Singleton, and Zhu, 2011)](https://academic.oup.com/rfs/article-abstract/24/3/926/1590594) model. Under our default option, the JSZ model is constrained by the AFNS[(Christensen, Diebold, and Rudebusch, 2011)](https://www.sciencedirect.com/science/article/pii/S0304407611000388) restriction. Under this restriction, the eigenvalues of the risk-neutral slope matrix are [1, exp(-kappaQ), exp(-kappaQ)].

Our package also allows users to estimate three distinct eigenvalues through an option. However, in this case, users must introduce prior distributions for the three eigenvalues. These prior distributions should be the form of `Distribution` from the [`Distributions.jl`](https://github.com/JuliaStats/Distributions.jl) package. For example, the following prior distributions can be introduced:

```julia
kappaQ_prior_pr = [truncated(Normal(0.9, 0.05), -1, 1), truncated(Normal(0.9, 0.05), -1, 1), truncated(Normal(0.9, 0.05), -1, 1)]
```

The `kappaQ_prior_pr` containing the prior distributions is a vector of length 3, and each entry is an object from the `Distributions.jl` package. `kappaQ_prior_pr[i]` is the prior distribution of the i-th diagonal element of the slope matrix of the VAR system under the risk-neutral measure.

After constructing `kappaQ_prior_pr`, it should be inputted as a keyword argument in functions that require the `kappaQ_prior_pr` variable (notably [`tuning_hyperparameter`](@ref) and [`posterior_sampler`](@ref)). If `kappaQ_prior_pr` is not provided, the model operates under the AFNS constraint automatically.

A point to note when setting prior distributions is that "the prior expectation of the slope matrix of the first lag of the VAR model under the physical measure" is set to `diagm(mean.(kappaQ_prior_pr))` under the unrestricted JSZ model. Therefore, the prior distributions of the eigenvalues should be set to reflect the prior expectations under the physical measure to some extent.
