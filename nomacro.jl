## Setting
using Pkg, Revise
Pkg.activate(@__DIR__)
Pkg.instantiate()
Pkg.precompile()
using TermStructureModels, ProgressMeter, Distributions, LinearAlgebra, Distributions
using CSV, DataFrames, XLSX, StatsBase, Dates, JLD2

## Data setting
upper_p = 18
date_start = Date("1987-01-01", "yyyy-mm-dd") |> x -> x - Month(upper_p + 2)
date_end = Date("2022-12-01", "yyyy-mm-dd")
tau_n = [1; 3; 6; 9; collect(12:6:60); collect(72:12:120)]
medium_tau = collect(36:42)

function data_loading(; date_start, date_end, tau_n)

    ## Yield data
    raw_yield = XLSX.readdata("LW_monthly.xlsx", "Sheet1", "A293:DQ748") |> x -> [Date.(string.(x[:, 1]), DateFormat("yyyymm")) convert(Matrix{Float64}, x[:, tau_n.+1])] |> x -> DataFrame(x, ["date"; ["Y$i" for i in tau_n]])
    yields = raw_yield[findall(x -> x == yearmonth(date_start), yearmonth.(raw_yield[:, 1]))[1]:findall(x -> x == yearmonth(date_end), yearmonth.(raw_yield[:, 1]))[1], :]
    yields = yields[3:end, :]

    yields = [Date.(string.(yields[:, 1]), DateFormat("yyyy-mm-dd")) Float64.(yields[:, 2:end])]
    rename!(yields, Dict(:x1 => "date"))

    return yields

end
yields = data_loading(; date_start, date_end, tau_n)

## Setting
# optimization
upper_q =
    [1 1
        1 1
        10 10
        100 100] .+ 0.0
std_kQ_infty = 0.2
kappaQ_prior_pr = [truncated(Normal(0.9, 0.05), -1, 1), truncated(Normal(0.9, 0.05), -1, 1), truncated(Normal(0.9, 0.05), -1, 1)]

# estimation
iteration = 5_000
burnin = 1_000
iter_sub = 2

## Do

tuned, opt = tuning_hyperparameter(Array(yields[:, 2:end]), [], tau_n, []; upper_p=1, upper_q, std_kQ_infty, medium_tau, kappaQ_prior_pr)
JLD2.save("tuned.jld2", "tuned", tuned, "opt", opt)
p = tuned.p

saved_params, acceptPrMH = posterior_sampler(Array(yields[upper_p-p+1:end, 2:end]), [], tau_n, [], iteration, tuned; medium_tau, std_kQ_infty, kappaQ_prior_pr)
saved_params = saved_params[burnin+1:end]
iteration = length(saved_params)
saved_params, Pr_stationary = erase_nonstationary_param(saved_params)
iteration = length(saved_params)
ineff = ineff_factor(saved_params)

saved_TP = term_premium(120, tau_n, saved_params[1:iter_sub:end], Array(yields[upper_p-p+1:end, 2:end]), [])
saved_paramsX = latentspace(saved_params, yields, tau_n)
reduced_params = reducedform(saved_params, yields, macros, tau_n)
fits = fitted_YieldCurve(tau_n, saved_paramsX)

idx = 18
plot(mean(fits)[:yields][:, idx])
plot!(yields[:, idx])

data = [yields macros]
factors = [PCA(yields, p)[1] macros]
latentms = [latents macros]
# Compare observations and fitted variables of the reducedform transition equation
fitted_samples = Vector{Matrix}(undef, length(reduced_params))
@showprogress for iter in eachindex(reduced_params)
    KPF = reduced_params[iter].KPF
    GPFF = reduced_params[iter].GPFF
    OmegaFF = reduced_params[iter].OmegaFF

    fitted = zeros(size(factors, 1), dP)
    fitted[1:p, :] = factors[1:p, :]
    for t in p+1:size(factors, 1)
        Xs = factors[t-1:-1:t-p, :]' |> vec
        fitted[t, :] = KPF + GPFF * Xs + rand(MvNormal(zeros(dP), OmegaFF))
    end
    fitted_samples[iter] = fitted
end
idx = 1
plot(factors[:, idx])
plot!(mean(fitted_samples)[:, idx])
aux = Vector{Float64}(undef, size(factors, 1))
for t in 1:size(factors, 1)
    aux[t] = quantile([fitted_samples[i][t, idx] for i in eachindex(fitted_samples)], 0.025)
end
plot!(aux)
aux = Vector{Float64}(undef, size(factors, 1))
for t in 1:size(factors, 1)
    aux[t] = quantile([fitted_samples[i][t, idx] for i in eachindex(fitted_samples)], 0.975)
end
plot!(aux)

# Compare observations and fitted variables of the latent space transition equation
fitted_samples = Vector{Matrix}(undef, length(saved_paramsX))
@showprogress for iter in eachindex(saved_paramsX)
    KPXF = saved_paramsX[iter].KPXF
    GPXFXF = saved_paramsX[iter].GPXFXF
    OmegaXFXF = saved_paramsX[iter].OmegaXFXF + eps() * I(dP) |> x -> 0.5(x + x')

    fitted = zeros(size(latentms, 1), dP)
    fitted[1:p, :] = latentms[1:p, :]
    for t in p+1:size(latentms, 1)
        Xs = latentms[t-1:-1:t-p, :]' |> vec
        fitted[t, :] = KPXF + GPXFXF * Xs + rand(MvNormal(zeros(dP), OmegaXFXF))
    end
    fitted_samples[iter] = fitted
end
idx = 4
plot(latentms[:, idx])
plot!(mean(fitted_samples)[:, idx])
aux = Vector{Float64}(undef, size(latentms, 1))
for t in 1:size(latentms, 1)
    aux[t] = quantile([fitted_samples[i][t, idx] for i in eachindex(fitted_samples)], 0.025)
end
plot!(aux)
aux = Vector{Float64}(undef, size(latentms, 1))
for t in 1:size(latentms, 1)
    aux[t] = quantile([fitted_samples[i][t, idx] for i in eachindex(fitted_samples)], 0.975)
end
plot!(aux)

## Diagnostics by the scenario analysis
predicted_yields = zeros(size(yields, 1), length(tau_n))
predicted_factors = zeros(size(yields, 1), dP)
for t = (size(yields, 1)-10):(size(yields, 1)-1)
    prediction = conditional_forecasts([], 120, 1, saved_params, yields[1:t, :], macros[1:t, :], tau_n)
    predicted_yields[t+1, :] = mean(prediction)[:yields][end, :]
    predicted_factors[t+1, :] = mean(prediction)[:factors][end, :]
end
idx = 4
plot(factors[(size(yields, 1)-9):(size(yields, 1)), idx])
plot!(predicted_factors[(size(yields, 1)-9):(size(yields, 1)), idx])
idx = 18
plot(yields[(size(yields, 1)-9):(size(yields, 1)), idx])
plot!(predicted_yields[(size(yields, 1)-9):(size(yields, 1)), idx])

predicted_yields = zeros(size(yields, 1), length(tau_n))
predicted_factors = zeros(size(yields, 1), dP)
for t = (size(yields, 1)-10):(size(yields, 1)-1)

    S = zeros(1, dP - dimQ() + length(tau_n))
    S[1, 19] = 1.0
    s = [data[t+1, 19]]
    scene = Scenario(combinations=deepcopy(S), values=deepcopy(s))

    prediction = conditional_forecasts([scene], 120, 1, saved_params, yields[1:t, :], macros[1:t, :], tau_n)
    predicted_yields[t+1, :] = mean(prediction)[:yields][end, :]
    predicted_factors[t+1, :] = mean(prediction)[:factors][end, :]
end
idx = 4
plot(factors[(size(yields, 1)-9):(size(yields, 1)), idx])
plot!(predicted_factors[(size(yields, 1)-9):(size(yields, 1)), idx])
idx = 18
plot(yields[(size(yields, 1)-9):(size(yields, 1)), idx])
plot!(predicted_yields[(size(yields, 1)-9):(size(yields, 1)), idx])

scene = Vector{Scenario}(undef, 4)
S = zeros(2, dP - dimQ() + length(tau_n))
s = zeros(2)
for h = 1:4
    S[1, 1] = 1.0
    s[1] = data[size(data, 1)-4+h, 1]
    S[2, 19] = 1.0
    s[2] = data[size(data, 1)-4+h, 19]
    scene[h] = Scenario(combinations=deepcopy(S), values=deepcopy(s))
end
prediction = conditional_forecasts(scene, 120, 8, saved_params, yields[1:size(data, 1)-4, :], macros[1:size(data, 1)-4, :], tau_n)
plot(mean(prediction)[:TP])
idx = 4
plot(factors[(end-3):end, idx])
plot!(mean(prediction)[:factors][:, idx])
idx = 18
plot(yields[(end-3):end, idx])
plot!(mean(prediction)[:yields][:, idx])