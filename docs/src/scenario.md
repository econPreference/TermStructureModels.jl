# Forecasting

The package provides two kinds of forecasts.

1. [Baseline Forecasts](@ref)
2. [Scenario Forecasts](@ref) (Scenario Analysis)

Both forecasts are conditional forecasts, because they are based on information in the data. The difference is that the scenario forecast assumes additional scenarios that describe future paths of some variables.

Baseline forecasts and scenario forecasts can be represented either as the posterior distribution of predicted objects or as the posterior distribution of conditional expectations of predicted objects. To summarize:

1. Posterior Distribution of Predicted Objects
   - In other words, the distribution of future objects conditional on past observations and the scenario
   - Function: [`conditional_forecast`](@ref)
2. Posterior Distribution of Conditional Expectations of Predicted Objects
   - In other words, the posterior distribution of "E[future object|past obs, scenario, parameters]"
   - Function: [`conditional_expectation`](@ref)

In this summary, for baseline forecasts, the scenario is the empty set.

The first one is the full Bayesian treatment, so it is mathematically strict. However, it can be difficult to derive meaningful implications from the prediction because of its wide prediction intervals. The second one considers only parameter uncertainty, so it underestimates the prediction uncertainty. However, it is appropriate when you make decisions based on the expected path of future variables. **We recommend the second version** (`conditional_expectation`).

The required inputs and the type of the output are the same between `conditional_forecast` and `conditional_expectation`. That is,

```julia
projections = conditional_forecast(S::Vector, tau, horizon, saved_params, yields, macros, tau_n; baseline=[], mean_macros::Vector=[], data_scale=1200, pca_loadings=[])
```

and

```julia
projections = conditional_expectation(S::Vector, tau, horizon, saved_params, yields, macros, tau_n; baseline=[], mean_macros::Vector=[], data_scale=1200, pca_loadings=[])
```

`projections::Vector{Forecast}` contains the results of the forecasting. `tau` is a vector, and the term premium of `tau[i]`-bond is forecasted for each `i`. If `tau` is set to `[]`, the term premium is not forecasted. `horizon` is the forecasting horizon. `horizon` should not be smaller than `length(S)`. `saved_params::Vector{Parameter}` is the output of [`posterior_sampler`](https://econpreference.github.io/TermStructureModels.jl/dev/estimation/#Step-2.-Sampling-the-Posterior-Distribution-of-Parameters).

You can use the same `yields`, `tau_n` and `macros` you employed when executing `posterior_sampler`. If you wish to compute conditional forecasts using observations up to a certain point, you can simply use `yields` and `macros` from the initial period up to that point. However, parameter uncertainty is incorporated independently of `yields` and `macros` through `saved_params`.

If you use demeaned macro data, option `mean_macros` is useful. If the sample mean of macro data is specified as the input value for `mean_macros`, `projections` contains conditional forecasts of non-demeaned macro variables. The sample mean of macro data can be calculated as follows.

```julia
mean_macros = mean(raw_macros_data, dims=1)[1, :]
```

!!! warning "Option `mean_macros`"

    If macro variables are not demeaned, ignore option `mean_macros`.

`S` determines whether we are computing a baseline forecast or a scenario forecast. How `S` is set will be described in the following sections.

## Baseline Forecasts

Do

```julia
S = []
```

It sets a scenario to an empty set, so the package calculates baseline forecasts.

## Scenario Forecasts

`S` should be `Vector{Scenario}`. `S` can be initialized by

```julia
S = Vector{Scenario}(undef, len)
```

`len` is the length of `S`. For example, if the scenario is assumed for the next 5 time periods, `len=5`.

`S[i]` represents the scenario for future variables at time `T+i`, where `T` refers to the time of the last observation in `macros` and `yields`. The type of `S[i]` is `Scenario`, and struct `Scenario` has two fields: `combinations::Matrix` and `values::Vector`. The fields in `S[i]` are implicitly defined by

```julia
S[i].combination*[yields[T+i,:]; macros[T+i, :]] == S[i].values
```

`[yields[T+i,:]; macros[T+i, :]]` is a predicted variable that is not observed. Scenario forecasts are calculated assuming that the above equation holds at time `T+i`, based on `S[i]` set by you. The number of rows in `S[i].combination` and the length of `S[i].values` are the same, and this length represents the number of scenarios assumed at time `T+i`.

Setting the two fields of `S[i]` is straightforward. Suppose that the content of the scenarios at time `T+i` is

```julia
combs*[yields[T+i,:]; macros[T+i, :]] == vals
```

Then, you can assign the content to `S[i]` by executing

```julia
S[i] = Scenario(combinations=combs, values=vals)
```

## Deviation-from-Baseline Scenario Forecasts

By default, scenario forecasts return the predicted path levels. If you want to express both the scenario inputs and the forecast outputs as **deviations from a baseline**, use the `baseline` keyword argument.

`baseline` should be the output of `conditional_forecast` (or `conditional_expectation`) obtained with `S = []`.

When `baseline` is provided, the scenario in `S` must be specified as deviations from `baseline`. For example, if the scenario assumes that the 3-month yield at time `T+1` is 50 basis points above the baseline, `S[1].values` should be set to `0.5` (not the level of the yield itself). The output forecasts are then also returned as deviations from `baseline`.

!!! warning "Consistency of `mean_macros`"

    If `mean_macros` was used when computing `baseline`, it must also be passed when using `baseline` as an input. Conversely, if `mean_macros` was not used for `baseline`, do not include it.
