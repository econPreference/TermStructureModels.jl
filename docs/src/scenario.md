# Forecasting

We have two kinds of forecasts.

1. Baseline Forecast
2. Scenario Analysis (or Scenario Forecast)

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
projections = conditional_forecasts(S::Vector, τ, horizon, saved_params, yields, macros, tau_n; mean_macros::Vector=[], data_scale=1200)
```

or

```julia
projections = scenario_analysis(S::Vector, τ, horizon, saved_params, yields, macros, tau_n; mean_macros::Vector=[], data_scale=1200)
```

`τ` is a vector. The term premium of `τ[i]`-bond is forecasted for each i. If `τ` is set to `[]`, the term premium is not forecasted. `horizon` is the forecasting horizon. `horizon` should not be smaller than `length(S)`. `saved_params::Vector{Parameter}` is the output of [`posterior_sampler`](@ref)
`S` determines whether we are computing a baseline forecast or a scenario forecast. How `S` is set will be described in the following sections.

## Baseline Forecast

S = []

## Scenario Forecast

comb = zeros(2, size([yields macros], 2), 3)
values = zeros(2, 3)
for t in 1:3 # for simplicity, we just assume the same scenario for time = T+1, T+2, T+3. Users can freely assume different scenarios for each time T+t.
comb[1, 1, t] = 1.0 # one month yield is selected as a conditioned variable in the first combination
comb[2, 20, t] = 0.5
comb[2, 21, t] = 0.5 # the average of 20th and 21st observables is selected as a second conditioned combination
values[1,t] = 3.0 # one month yield at time T+t is 3.0
values[2,t] = 0.0 # the average value is zero.
end
S = Scenario(combinations=comb, values=values)

```

Here, **both "combinations" and "values" should be type Array{Float64}**. Also, "horizon" should not be smaller than size(values, 2).
```
