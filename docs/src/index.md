# TermStructureModels.jl

[TermStructureModels.jl](https://github.com/econPreference/TermStructureModels.jl) has the below functions.

- [Statistical Inference](https://econpreference.github.io/TermStructureModels.jl/dev/inference/)
  - [Parameters](https://econpreference.github.io/TermStructureModels.jl/dev/inference/#Inference-for-Parameters)
  - [Yield Curve Interpolation](https://econpreference.github.io/TermStructureModels.jl/dev/inference/#Yield-Curve-Interpolation)
  - [Term Premiums](https://econpreference.github.io/TermStructureModels.jl/dev/inference/#Term-Premiums)
- [Forecasting](https://econpreference.github.io/TermStructureModels.jl/dev/scenario)
  - [Conditional Forecasting without scenarios (Baseline Forecasts)](https://econpreference.github.io/TermStructureModels.jl/dev/scenario/#Baseline-Forecasts)
  - [Scenario Analysis (Scenario Forecasts)](https://econpreference.github.io/TermStructureModels.jl/dev/scenario/#Scenario-Forecasts)

To use the above functions, [an estimation of the model](https://econpreference.github.io/TermStructureModels.jl/dev/estimation/) must first be conducted. That is, use [`posterior_sampler`](@ref) to obtain posterior samples of parameters. The posterior samples are used for the above functions (Statistical inference and Forecasting). For details, refer to the corresponding pages.

Some outputs of our package are not simple arrays. They are

- [`Vector{Parameter}`](https://econpreference.github.io/TermStructureModels.jl/dev/api/#TermStructureModels.Parameter): output of [`posterior_sampler`](@ref)
- [`Vector{ReducedForm}`](https://econpreference.github.io/TermStructureModels.jl/dev/api/#TermStructureModels.ReducedForm): output of [`reducedform`](@ref)
- [`Vector{LatentSpace}`](https://econpreference.github.io/TermStructureModels.jl/dev/api/#TermStructureModels.LatentSpace): output of [`latentspace`](@ref)
- [`Vector{YieldCurve}`](https://econpreference.github.io/TermStructureModels.jl/dev/api/#TermStructureModels.YieldCurve): output of [`fitted_YieldCurve`](@ref)
- [`Vector{TermPremium}`](https://econpreference.github.io/TermStructureModels.jl/dev/api/#TermStructureModels.TermPremium): output of [`term_premium`](@ref)
- [`Vector{Forecast}`](https://econpreference.github.io/TermStructureModels.jl/dev/api/#TermStructureModels.Forecast): outputs of [`conditional_forecasts`](@ref) and [`scenario_analysis`](@ref)

The above outputs contain information about the posterior distributions of objects of interest. Users can use the outputs above to [extract posterior samples](https://econpreference.github.io/TermStructureModels.jl/dev/output/#Extract-Posterior-Samples) or [calculate descriptive statistics of the posterior distributions](https://econpreference.github.io/TermStructureModels.jl/dev/output/#Descriptive-Statistics-of-the-Posterior-Distributions).

Our package is based on [our paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4708628). Descriptions of our model and the meanings of each variable can be found in the paper. [The Notation section](https://econpreference.github.io/TermStructureModels.jl/dev/notations/) details how notations in the paper correspond to variables in our package. Additionally, [the example file](https://github.com/econPreference/TermStructureModels.jl/blob/main/examples/LargeVAR_Yields_Macros/LargeVAR_Yields_Macros.ipynb) used in our paper is available in the repository.

**Users are encouraged to read the two text boxes below.**

!!! warning "Unit of Data"

    Theoretical term structure models typically describe bond yields as decimals per one time period. However, yield data is typically presented in percent per annum. Therefore, you have to address the issue by using the option `data_scale`. `data_scale` represents the scale of the data. Specifically,

    ```julia
    `yields_in_data` = `data_scale`*`theoretical_yields_in_the_model`
    ```

    holds. For example, suppose we have monthly yield data in percent per annum. If we use a monthly term structure model, `data_scale=1200`. **The default value of `data_scale` is 1200 for all functions.**

    Functions that have option `data_scale` are as follows:
    - [`tuning_hyperparameter`](@ref)
    - [`posterior_sampler`](@ref)
    - [`term_premium`](@ref)
    - [`conditional_forecasts`](@ref)
    - [`scenario_analysis`](@ref)
    - [`latentspace`](@ref)
    - [`reducedform`](@ref)
    - [`fitted_YieldCurve`](@ref)
    - [`generative`](@ref)
    - [`calibrate_mean_phi_const`](@ref)

!!! tip "Normalization of Data"

    Our package demeans the principal components of bond yields, which are spanned risk factors in the bond market. Therefore, we recommend using macro data after demeaning it. Of course, demeaning the macro variables is recommended but not mandatory.

## Other Forms of the Model

### Yield-Only Model

Users may want to use yield-only models in which `macros` is an empty set. In such instances, set `macros = []` and `rho = []` for all functions.

### Unrestricted JSZ model

Our model is the three-factor JSZ[(Joslin, Singleton, and Zhu, 2011)](https://academic.oup.com/rfs/article-abstract/24/3/926/1590594) model. Under our default option, the JSZ model is constrained by the AFNS[(Christensen, Diebold, and Rudebusch, 2011)](https://www.sciencedirect.com/science/article/pii/S0304407611000388) restriction. Under this restriction, the eigenvalues of the risk-neutral slope matrix are [1, exp(-kappaQ), exp(-kappaQ)].

Our package also allows users to estimate three distinct eigenvalues through an option. However, in this case, users must introduce prior distributions for the three eigenvalues. These prior distributions should be the form of `Distribution` from the [`Distributions.jl`](https://github.com/JuliaStats/Distributions.jl) package. For example, the following prior distributions can be introduced:

```julia
kappaQ_prior_pr = [truncated(Normal(0.9, 0.05), -1, 1), truncated(Normal(0.9, 0.05), -1, 1), truncated(Normal(0.9, 0.05), -1, 1)]
```

The `kappaQ_prior_pr` containing the prior distributions is a vector of length 3, and each entry is an object from the `Distributions.jl` package. `kappaQ_prior_pr[i]` is the prior distribution of the i-th diagonal element of the slope matrix of the VAR system under the risk-neutral measure.

After constructing `kappaQ_prior_pr`, it should be inputted as a keyword argument in functions that require the `kappaQ_prior_pr` variable (notably [`tuning_hyperparameter`](@ref) and [`posterior_sampler`](@ref)). If `kappaQ_prior_pr` is not provided, the model operates under the AFNS constraint automatically.

A point to note when setting prior distributions is that "the prior expectation of the slope matrix of the first lag of the VAR model under the physical measure" is set to `diagm(mean.(kappaQ_prior_pr))` under the unrestricted JSZ model. Therefore, the prior distributions of the eigenvalues should be set to reflect the prior expectations under the physical measure to some extent.
