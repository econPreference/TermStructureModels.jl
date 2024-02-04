# Forecasting

## Baseline Forecasts

```juila
prediction = scenario_sampler(S::Scenario, τ, horizon, saved_params, yields, macros, tau_n)
```

The function generates (un)conditional forecasts using our model. We use the Kalman filter to make conditional filtered forecasts (Bańbura, Giannone, and Lenza, 2015), and then we use Kim and Nelson (1999) to make smoothed posterior samples of the conditional forecasts. "S" is a conditioned scenario, and yields, risk factors, and a term premium of maturity "τ" are forecasted. "horizon" is a forecasting horizon. "tau*n", "yields", and "macros" are the things that were inputs of function "posterior sampler". "saved*θ" is an output of function "posterior sampler". The output is Vector{Forecast}.

Struct Scenario has two elements, "combinations" and "values". Meaning of the struct can be found by help? command. Examples of making struct "Scenario" are as follows.

```juila
# Case 1. Unconditional Forecasts
S = []

# Case 2. Scenario with one conditioned variable and time length 2
comb = zeros(1, size([yields macros], 2))
comb[1, 1] = 1.0 # one month yield is selected as a conditioned variable
values = [3.0] # Scenario: one month yield at time T+1 is 3.0
S = Scenario(combinations=comb, values=values)

# Case 3. Scenario with two conditioned combinations and time length 3
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

## Scenario Analysis
