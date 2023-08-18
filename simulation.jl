## Setting
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
Pkg.precompile()
using GDTSM, ProgressMeter, Distributions, LinearAlgebra, Plots

## Simulating sample data
T = 1000
dP = 20
τₙ = [1; 3; 6; 9; collect(12:6:60); collect(72:12:120)]
p = 2

κQ = 0.0597
kQ_infty = 0
KₚXF = zeros(dP)
ΩXFXF = 0.01I(dP)

diag_G = rand(dP) # for diag_G to be global
GₚXFXF = fill(100, dP, dP * p)
while ~isstationary(GₚXFXF)
    aux = 0.1randn(dP, dP)
    aux -= diagm(diag(aux))
    global diag_G = 0.9rand(dP) .+ 0
    global GₚXFXF = [diagm(diag_G) + aux 0.1randn(dP, (p - 1) * dP)]
    global GₚXFXF[1:dimQ(), 1:dimQ()] = GQ_XX(; κQ)
end

# Generating samples
yields, latents, macros = generative(T, dP, τₙ, p, 0.01; κQ, kQ_infty, KₚXF, GₚXFXF, ΩXFXF)

## Turing hyper-parameters
diag_G = diag_G[dimQ()+1:end]
ρ = zeros(dP - dimQ())
ρ[diag_G.>0.5] .= 1.0
init_ν0 = dP + 10
tuned, opt = tuning_hyperparameter(yields, macros, τₙ, ρ; maxiter=1000, upper_p=4, init_ν0)

## Estimating
iteration = 10_000
saved_θ, acceptPrMH = posterior_sampler(yields, macros, τₙ, ρ, iteration, tuned)
saved_θ = saved_θ[round(Int, 0.1iteration):end]
saved_θ, accept_rate = stationary_θ(saved_θ)
ineff = ineff_factor(saved_θ)
saved_Xθ = latentspace(saved_θ, yields, τₙ)
saved_TP = term_premium(120, τₙ, saved_θ[1:50:end], yields, macros)
reduced_θ = reducedform(saved_θ, yields, macros, τₙ)
fitted = fitted_YieldCurve(τₙ, saved_Xθ)
idx = 18
plot(mean(fitted)[:yields][:, idx])
plot!(yields[:, idx])

data = [yields macros]
factors = [PCA(yields, p)[1] macros]
predicted_yields = zeros(size(yields, 1), length(τₙ))
predicted_factors = zeros(size(yields, 1), dP)
for t = (size(yields, 1)-10):(size(yields, 1)-1)
    prediction = scenario_sampler([], 120, 1, saved_θ, yields[1:t, :], macros[1:t, :], τₙ)
    predicted_yields[t+1, :] = mean(prediction)[:yields][end, :]
    predicted_factors[t+1, :] = mean(prediction)[:factors][end, :]
end
idx = 20
plot(factors[(size(yields, 1)-9):(size(yields, 1)), idx])
plot!(predicted_factors[(size(yields, 1)-9):(size(yields, 1)), idx])
idx = 18
plot(yields[(size(yields, 1)-9):(size(yields, 1)), idx])
plot!(predicted_yields[(size(yields, 1)-9):(size(yields, 1)), idx])

predicted_yields = zeros(size(yields, 1), length(τₙ))
predicted_factors = zeros(size(yields, 1), dP)
for t = (size(yields, 1)-10):(size(yields, 1)-1)

    S = zeros(1, dP - dimQ() + length(τₙ))
    S[1, 20] = 1.0
    s = [data[t+1, 20]]
    scene = Scenario(combinations=deepcopy(S), values=deepcopy(s))

    prediction = scenario_sampler([scene], 120, 1, saved_θ, yields[1:t, :], macros[1:t, :], τₙ)
    predicted_yields[t+1, :] = mean(prediction)[:yields][end, :]
    predicted_factors[t+1, :] = mean(prediction)[:factors][end, :]
end
idx = 20
plot(factors[(size(yields, 1)-9):(size(yields, 1)), idx])
plot!(predicted_factors[(size(yields, 1)-9):(size(yields, 1)), idx])
idx = 18
plot(yields[(size(yields, 1)-9):(size(yields, 1)), idx])
plot!(predicted_yields[(size(yields, 1)-9):(size(yields, 1)), idx])

scene = Vector{Scenario}(undef, 4)
S = zeros(2, dP - dimQ() + length(τₙ))
s = zeros(2)
for h = 1:4
    S[1, 1] = 1.0
    s[1] = data[size(data, 1)-4+h, 1]
    S[2, 20] = 1.0
    s[2] = data[size(data, 1)-4+h, 20]
    scene[h] = Scenario(combinations=deepcopy(S), values=deepcopy(s))
end
prediction = scenario_sampler(scene, 120, 8, saved_θ, yields[1:size(data, 1)-4, :], macros[1:size(data, 1)-4, :], τₙ)
plot(mean(prediction)[:TP])
idx = 20
plot(factors[(end-3):end, idx])
plot!(mean(prediction)[:factors][:, idx])
idx = 18
plot(yields[(end-3):end, idx])
plot!(mean(prediction)[:yields][:, idx])
