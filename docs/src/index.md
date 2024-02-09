# TermStructureModels.jl

[TermStructureModels.jl](https://github.com/econPreference/TermStructureModels.jl) has the below functions.

- [Statistical Inference](https://econpreference.github.io/TermStructureModels.jl/dev/inference/)
  - [Parameters](https://econpreference.github.io/TermStructureModels.jl/dev/inference/#Inference-for-Parameters)
  - [Yield Curve Interpolation](https://econpreference.github.io/TermStructureModels.jl/dev/inference/#Yield-Curve-Interpolation)
  - [Term Premiums](https://econpreference.github.io/TermStructureModels.jl/dev/inference/#Term-Premiums)
- [Forecasting](https://econpreference.github.io/TermStructureModels.jl/dev/scenario)
  - [Conditional Forecasting without scenarios (Baseline Forecasts)](https://econpreference.github.io/TermStructureModels.jl/dev/scenario/#Baseline-Forecast)
  - [Scenario Analysis (Scenario Forecasts)](https://econpreference.github.io/TermStructureModels.jl/dev/scenario/#Scenario-Forecast)

To use the above functions, [an estimation of the model](https://econpreference.github.io/TermStructureModels.jl/dev/estimation/) must first be conducted. That is, use [`posterior_sampler`](@ref) to obtain posterior samples of parameters. The posterior samples are used for the above functions (Statistical inference and Forecasting). For details, refer to the corresponding pages.

Our package is based on [our paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4708628). Descriptions of our model and the meanings of each variable can be found in the paper. [The Notation section](https://econpreference.github.io/TermStructureModels.jl/dev/notations/) details how notations in the paper correspond to variables in our package. Additionally, [the example file](https://github.com/econPreference/TermStructureModels.jl/blob/main/examples/LargeVAR_Yields_Macros.ipynb) used in our paper is available in the repository.

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
