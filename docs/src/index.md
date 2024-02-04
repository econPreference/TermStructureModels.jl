# TermStructureModels.jl

Our Package has the below functions,

- [Statistical Inference](https://econpreference.github.io/TermStructureModels.jl/dev/inference/)
  - [Parameters](https://econpreference.github.io/TermStructureModels.jl/dev/inference/#Inference-for-Parameters)
  - [Yield Curve Interpolation](https://econpreference.github.io/TermStructureModels.jl/dev/inference/#Yield-Curve-Interpolation)
  - [Term Premiums](https://econpreference.github.io/TermStructureModels.jl/dev/inference/#Term-Premiums)
- [Forecasting](https://econpreference.github.io/TermStructureModels.jl/dev/scenario)
  - [Conditional Forecasting without scenarios(Baseline Forecasts)](https://econpreference.github.io/TermStructureModels.jl/dev/scenario/#Baseline-Forecasts)
  - [Scenario Analysis](https://econpreference.github.io/TermStructureModels.jl/dev/scenario/#Scenario-Analysis)

To use such functions, [an estimation of the model](https://econpreference.github.io/TermStructureModels.jl/dev/estimation/) must first be conducted. That is, use [`posterior_sampler`](https://econpreference.github.io/TermStructureModels.jl/dev/api/#TermStructureModels.posterior_sampler-Tuple{Any,%20Any,%20Any,%20Any,%20Any,%20Hyperparameter}) to obtain posterior samples of parameters. The output of the function has a form of `Vector{Parameter}(posterior, iteration)`. This output is used for the above functions(Statistical inference and Forecasting). For details, refer to the corresponding pages.

Our package is based on [our paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4708628). Descriptions of our model and the meanings of each variable can be found in the paper. [The notation section](https://econpreference.github.io/TermStructureModels.jl/dev/notations/) organizes which notation in the paper corresponds to the variable in our package.
