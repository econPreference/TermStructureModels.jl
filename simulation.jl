using Distributions, LinearAlgebra
using GDTSM

## Simulating sample data
T = 200
dP = 20
τₙ = [1, 3, 6, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120]
p = 2

κQ = 0.0597
kQ_infty = 0
KₚXF = zeros(dP)
ΩXFXF = 0.01I(dP)

diag_G = rand(dP)
GₚXFXF = fill(100, dP, dP * p)
while ~isstationary(GₚXFXF)
    aux = 0.1randn(dP, dP)
    aux -= diagm(diag(aux))
    diag_G = 1rand(dP) .+ 0
    GₚXFXF = [diagm(diag_G) + aux 0.1randn(dP, (p - 1) * dP)]
    GₚXFXF[1:dimQ(), 1:dimQ()] = GQ_XX(; κQ)
end

# Generating samples
yields, latents, macros = generative(T, dP, τₙ, p; κQ, kQ_infty, KₚXF, GₚXFXF, ΩXFXF)

## Turing hyper-parameters
diag_G = diag_G[dimQ()+1:end]
ρ = zeros(dP - dimQ())
ρ[diag_G.>0.5] .= 0.9
tuned = tuning_hyperparameter(yields, macros, ρ)

## Estimating
iteration = 10_000
saved_θ, acceptPr_C_σ²FF, acceptPr_ηψ = posterior_sampler(yields, macros, τₙ, ρ, iteration, tuned; sparsity=true)
saved_θ = saved_θ[round(Int, 0.1iteration):end]
saved_θ, accept_rate = stationary_θ(saved_θ)
sparse_θ, trace_sparsity = sparse_precision(saved_θ, size(macros, 1) - tuned.p)
saved_Xθ = latentspace(saved_θ, yields, τₙ)
saved_TP = term_premium(120, τₙ, saved_θ, yields, macros)
reduced_θ = reducedform(saved_θ)
fitted = fitted_YieldCurve(τₙ, saved_Xθ)
idx = 13
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
idx = 11
plot(factors[(size(yields, 1)-9):(size(yields, 1)), idx])
plot!(predicted_factors[(size(yields, 1)-9):(size(yields, 1)), idx])

predicted_yields = zeros(size(yields, 1), length(τₙ))
predicted_factors = zeros(size(yields, 1), dP)
for t = (size(yields, 1)-10):(size(yields, 1)-1)

    S = zeros(1, dP - dimQ() + length(τₙ))
    S[1, 20] = 1.0
    s = [data[t+1, 20]]
    scene = Scenario(combinations=S, values=s)

    prediction = scenario_sampler(scene, 120, 1, saved_θ, yields[1:t, :], macros[1:t, :], τₙ)
    predicted_yields[t+1, :] = mean(prediction)[:yields][end, :]
    predicted_factors[t+1, :] = mean(prediction)[:factors][end, :]
end
idx = 10
plot(factors[(size(yields, 1)-9):(size(yields, 1)), idx])
plot!(predicted_factors[(size(yields, 1)-9):(size(yields, 1)), idx])

S = zeros(2, dP - dimQ() + length(τₙ), 4)
s = Matrix{Float64}(undef, 2, 4)
for h = 1:4
    S[1, 1, h] = 1.0
    s[1, h] = data[size(data, 1)-4+h, 1]
    S[2, 20, h] = 1.0
    s[2, h] = data[size(data, 1)-4+h, 20]
end
scene = Scenario(combinations=S, values=s)
prediction = scenario_sampler(scene, 120, 8, saved_θ, yields[1:size(data, 1)-4, :], macros[1:size(data, 1)-4, :], τₙ)
plot(mean(prediction)[:TP])
