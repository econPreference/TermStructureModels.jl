# Forecasting

We have two kinds of forecasts.

1. [Baseline Forecast](@ref)
2. Scenario Analysis (or [Scenario Forecast](@ref))

All of two forecasts are conditional forecasts, because they are based on information in data. The difference is that the scenario forecast assumes additional scenarios that describe future paths of some variables.

On the other hand, we have two kinds of estimates of forecasts.

1. Posterior Distribution of Forecasts
   - In other words, Posterior Distribution of "future observation|past observation, scenario"
   - Function: [`conditional_forecasts`](@ref)
2. Posterior Distribution of expected future values of variables
   - In other words, Posterior Distribution of "E[future obs|past obs, scenario, parameters]"
   - Function: [`scenario_analysis`](@ref)

The first one is the full-Bayesian version of forecasts, so it is mathematically strict. However, it can be difficult to derive meaningful implications from the prediction because of its wide prediction intervals. The second one consider only parameter uncertainty, so it underestimates the prediction uncertainty. However, it is appropriate when the policymaker makes decisions based on the expected path of the future economy. **We recommend the second version.**

The required inputs and the type of the output are the same between the two functions.

```julia
projections = conditional_forecasts(S::Vector, τ, horizon, saved_params, yields, macros, tau_n;
                                    mean_macros::Vector=[],
                                    data_scale=1200)
```

or

```julia
projections = scenario_analysis(S::Vector, τ, horizon, saved_params, yields, macros, tau_n;
                                mean_macros::Vector=[],
                                data_scale=1200)
```

`τ` is a vector. The term premium of `τ[i]`-bond is forecasted for each i. If `τ` is set to `[]`, the term premium is not forecasted. `horizon` is the forecasting horizon. `horizon` should not be smaller than `length(S)`. `saved_params::Vector{Parameter}` is the output of [`posterior_sampler`](@ref).

Users can use the same `yields`, `tau_n` and `macros` they employed when executing `posterior_sampler`. If one wishes to compute conditional forecasts using observations up to a certain point, they can simply use the `yields` and `macros` from the initial period up to that point. However, parameter uncertainty is incorporated independently of `yields` and `macros` through `saved_params`.

If you use demeaned macro data, option `mean_macros` is useful. This option allows for the calculation of conditional forecasts for the non-demeaned macro variables. An input for `mean_macros` is calculated by

```julia
    mean_macros = mean(macros, dims=1)[1, :]
```

`S` determines whether we are computing a baseline forecast or a scenario forecast. How `S` is set will be described in the following sections.

## Baseline Forecast

Do

```julia
S = []
```

It sets a scenario to an empty set, so the package calculate baseline forecasts.

## Scenario Forecast

`S` should be `Vector{Scenario}`. `S` can be initialized by

```julia
S = Vector{Scenario}(undef, len)
```

`len` is the length of `S`. For example, if Scenario `S` is assumed for the next 5 time periods, `len=5`.

`S[i]` represents the scenario for future variables at time `T+i`, where `T` refers to the time of the last observation in `macros` and `yields`. The type of `S[i]` is `Scenario`, and struct `Scenario` has two fields: `combinations::Matrix` and `values::Vector`. The fields in `S[i]` say that

```julia
S[i].combination*[yields[T+i,:]; macros[T+i, :]] == S[i].values
```

holds at time `T+i`. The number of rows in `S[i].combination` and the length of `S[i].values` are the same, and this length represents the number of scenarios assumed at time `T+i`.

The two fields can be set using a content of scenarios at time `T+i`. Suppose that the content of the scenarios at time `T+i` is

```julia
combs*[yields[T+i,:]; macros[T+i, :]] == vals
```

Then, users can assign the content to `S[i]` by executing

```julia
S[i] = Scenario(combinations=combs, values=vals)
```
