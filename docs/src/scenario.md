# Forecasting

We have two kinds of forecasts.

1. [Baseline Forecasts](@ref)
2. [Scenario Forecasts](@ref) (Scenario Analysis)

All of two forecasts are conditional forecasts, because they are based on information in data. The difference is that the scenario forecast assumes additional scenarios that describe future paths of some variables.

Baseline forecasts and scenario forecasts can be represented either as the posterior distribution of predicted objects or as the posterior distribution of conditional expectations of predicted objects. To summarize:

1. Posterior Distribution of Predicted Objects
   - In other words, Distribution of future objects conditional on past observations and the scenario
   - Function: [`conditional_forecasts`](@ref)
2. Posterior Distribution of Conditional Expectations of Predicted Objects
   - In other words, Posterior Distribution of "E[future object|past obs, scenario, parameters]"
   - Function: [`scenario_analysis`](@ref)

In this summary, for baseline forecasts, the scenario is the empty set.

The first one is the full Bayesian treatment, so it is mathematically strict. However, it can be difficult to derive meaningful implications from the prediction because of its wide prediction intervals. The second one consider only parameter uncertainty, so it underestimates the prediction uncertainty. However, it is appropriate when users make decisions based on the expected path of future variables. **We recommend the second version (`scenario_analysis`).**

The required inputs and the type of the output are the same between `conditional_forecasts` and `scenario_analysis`. That is,

```julia
projections = conditional_forecasts(S::Vector, τ, horizon, saved_params, yields, macros, tau_n;
                                    mean_macros::Vector=[],
                                    data_scale=1200)
```

and

```julia
projections = scenario_analysis(S::Vector, τ, horizon, saved_params, yields, macros, tau_n;
                                mean_macros::Vector=[],
                                data_scale=1200)
```

`τ` is a vector. The term premium of `τ[i]`-bond is forecasted for each `i`. If `τ` is set to `[]`, the term premium is not forecasted. `horizon` is the forecasting horizon. `horizon` should not be smaller than `length(S)`. `saved_params::Vector{Parameter}` is the output of [`posterior_sampler`](@ref).

Users can use the same `yields`, `tau_n` and `macros` they employed when executing `posterior_sampler`. If one wishes to compute conditional forecasts using observations up to a certain point, they can simply use `yields` and `macros` from the initial period up to that point. However, parameter uncertainty is incorporated independently of `yields` and `macros` through `saved_params`.

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

It sets a scenario to an empty set, so the package calculate baseline forecasts.

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

`[yields[T+i,:]; macros[T+i, :]]` is a predicted value that is not observed. Scenario forecasts are calculated assuming that the above equation holds at time `T+i`, based on `S[i]` set by users. The number of rows in `S[i].combination` and the length of `S[i].values` are the same, and this length represents the number of scenarios assumed at time `T+i`.

Setting the two fields of `S[i]` is straightforward. Suppose that the content of the scenarios at time `T+i` is

```julia
combs*[yields[T+i,:]; macros[T+i, :]] == vals
```

Then, users can assign the content to `S[i]` by executing

```julia
S[i] = Scenario(combinations=combs, values=vals)
```
