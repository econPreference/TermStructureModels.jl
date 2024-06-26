## Setting
using Pkg, Revise
Pkg.activate(@__DIR__)
Pkg.instantiate()
Pkg.precompile()
using TermStructureModels, ProgressMeter, Distributions, LinearAlgebra, Distributions, Plots

## Simulating sample data
T = 1000
dQ = 4
dP = 5
tau_n = [1; 3; 6; 9; collect(12:6:60); collect(72:12:120)]
p = 2

kappaQ = 0.0597
kQ_infty = 0
KPXF = zeros(dP)
OmegaXFXF = 0.01I(dP)

diag_G = rand(dP) # for diag_G to be global
GPXFXF = fill(100, dP, dP * p)
while ~isstationary(GPXFXF)
    aux = 0.1randn(dP, dP)
    aux -= diagm(diag(aux))
    global diag_G = 0.9rand(dP) .+ 0
    global GPXFXF = [diagm(diag_G) + aux 0.1randn(dP, (p - 1) * dP)]
    global GPXFXF[1:dimQ(), 1:dimQ()] = diagm([0.98, 0.93, 0.88])
end

# Generating samples
yields, latents, macros = generative(T, dQ, dP, tau_n, p, 0.0001; kappaQ=[0.99, 0.94, 0.9], kQ_infty, KPXF, GPXFXF, OmegaXFXF)

## Turing hyper-parameters
diag_G = diag_G[dimQ()+1:end]
rho = zeros(dP - dimQ())
rho[diag_G.>0.5] .= 1.0
medium_tau = collect(24:3:60)
kappaQ_prior_pr = [truncated(Normal(0.9, 0.05), -1, 1), truncated(Normal(0.9, 0.05), -1, 1), truncated(Normal(0.9, 0.05), -1, 1)]
std_kQ_infty = 0.2
tuned, opt = tuning_hyperparameter(yields, macros, tau_n, rho; std_kQ_infty, medium_tau, kappaQ_prior_pr)
p = tuned.p

## Estimating
iteration = 10_000
saved_θ, acceptPrMH = posterior_sampler(yields, macros, tau_n, rho, iteration, tuned; medium_tau, std_kQ_infty, kappaQ_prior_pr)
saved_θ = saved_θ[round(Int, 0.1iteration):end]
saved_θ, accept_rate = erase_nonstationary_param(saved_θ)
ineff = ineff_factor(saved_θ)
saved_Xθ = latentspace(saved_θ, yields, tau_n)
saved_TP = term_premium(120, tau_n, saved_θ[1:50:end], yields, macros)
reduced_θ = reducedform(saved_θ, yields, macros, tau_n)
fits = fitted_YieldCurve(tau_n, saved_Xθ)

idx = 18
plot(mean(fits)[:yields][:, idx])
plot!(yields[:, idx])

data = [yields macros]
factors = [PCA(yields, p)[1] macros]
latentms = [latents macros]
# Compare observations and fitted variables of the reducedform transition equation
fitted_samples = Vector{Matrix}(undef, length(reduced_θ))
@showprogress for iter in eachindex(reduced_θ)
    KPF = reduced_θ[iter].KPF
    GPFF = reduced_θ[iter].GPFF
    OmegaFF = reduced_θ[iter].OmegaFF

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
fitted_samples = Vector{Matrix}(undef, length(saved_Xθ))
@showprogress for iter in eachindex(saved_Xθ)
    KPXF = saved_Xθ[iter].KPXF
    GPXFXF = saved_Xθ[iter].GPXFXF
    OmegaXFXF = saved_Xθ[iter].OmegaXFXF + eps() * I(dP) |> x -> 0.5(x + x')

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
    prediction = conditional_forecasts([], 120, 1, saved_θ, yields[1:t, :], macros[1:t, :], tau_n)
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

    prediction = conditional_forecasts([scene], 120, 1, saved_θ, yields[1:t, :], macros[1:t, :], tau_n)
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
prediction = conditional_forecasts(scene, 120, 8, saved_θ, yields[1:size(data, 1)-4, :], macros[1:size(data, 1)-4, :], tau_n)
plot(mean(prediction)[:TP])
idx = 4
plot(factors[(end-3):end, idx])
plot!(mean(prediction)[:factors][:, idx])
idx = 18
plot(yields[(end-3):end, idx])
plot!(mean(prediction)[:yields][:, idx])