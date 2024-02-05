# TermStructureModels.jl

[TermStructureModels.jl](https://github.com/econPreference/TermStructureModels.jl) has the below functions,

- [Statistical Inference](https://econpreference.github.io/TermStructureModels.jl/dev/inference/)
  - [Parameters](https://econpreference.github.io/TermStructureModels.jl/dev/inference/#Inference-for-Parameters)
  - [Yield Curve Interpolation](https://econpreference.github.io/TermStructureModels.jl/dev/inference/#Yield-Curve-Interpolation)
  - [Term Premiums](https://econpreference.github.io/TermStructureModels.jl/dev/inference/#Term-Premiums)
- [Forecasting](https://econpreference.github.io/TermStructureModels.jl/dev/scenario)
  - [Conditional Forecasting without scenarios(Baseline Forecast)](https://econpreference.github.io/TermStructureModels.jl/dev/scenario/#Baseline-Forecast)
  - [Scenario Analysis(Scenario Forecast)](https://econpreference.github.io/TermStructureModels.jl/dev/scenario/#Scenario-Forecast)

To use such functions, [an estimation of the model](https://econpreference.github.io/TermStructureModels.jl/dev/estimation/) must first be conducted. That is, use [`posterior_sampler`](@ref) to obtain posterior samples of parameters. The output of the function has a form of `Vector{Parameter}(posterior, iteration)`. This output is used for the above functions(Statistical inference and Forecasting). For details, refer to the corresponding pages.

Our package is based on [our paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4708628). Descriptions of our model and the meanings of each variable can be found in the paper. [The notation section](https://econpreference.github.io/TermStructureModels.jl/dev/notations/) organizes which notation in the paper corresponds to the variable in our package. [The example file](https://github.com/econPreference/TermStructureModels.jl/blob/main/examples/LargeVAR_Yields_Macros.ipynb) used for our paper is in the repo.

Users are encouraged to read the two text boxes below.

!!! warning "Units of Data"

    Theoretical term structure models typically describe bond yields as decimals per one time period. However, yield data is typically presented in percent per annum. One way to address this issue is by transforming the data into yields in decimal form per one time period.

    **Our recommendation** is to use the option `data_scale`. `data_scale` represents the scale of the data. Specifically,

    yields in data = `data_scale`*theoretical yields in the model

    holds. For example, suppose we have monthly yield data in percent per annum. If we use a monthly term structure model, `data_scale=1200`. The default value of `data_scale` is 1200.

    Functions that have option `data_scale` are as follows:
    - `tuning_hyperparameter`
    - `posterior_sampler`
    - `term_premium`
    - `conditional_forecasts`
    - `scenario_analysis`
    - `latentspace`
    - `reducedform`
    - `fitted_YieldCurve`
    - `generative`
    - `calibrate_mean_phi_const`

!!! tip "Normalization of Data"

    Our package demeans the principal components, which are risk factors in the bond market. Therefore, we recommend using macro data after demeaning it. Of course, it's not necessary to use demeaned macro variables.
