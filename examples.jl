using Random, Distributions, LinearAlgebra
using GDTSM
Random.seed!(111)

## Simulating sample data
T = 200
dP = 20
τₙ = [1, 3, 6, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120]
p = 2

κQ = 0.0597
kQ_infty = 0
KₚXF = zeros(dP)
ΩXFXF = 0.01I(dP)

aux = 0.01randn(dP, dP)
aux -= diagm(diag(aux))
GₚXFXF = [diagm(0.5rand(dP) .+ 0.4) + aux 0.05 * randn(dP, (p - 1) * dP)]
GₚXFXF[1:3, 1:3] = GQ_XX(; κQ)
while ~isstationary(GₚXFXF)
    local aux = 0.01randn(dP, dP)
    local aux -= diagm(diag(aux))
    global GₚXFXF = [diagm(0.5rand(dP) .+ 0.4) + aux 0.05 * randn(dP, (p - 1) * dP)]
    global GₚXFXF[1:3, 1:3] = GQ_XX(; κQ)
end
# Generating samples
yields, latents, macros = generative(T, dP, τₙ, p; κQ, kQ_infty, KₚXF, GₚXFXF, ΩXFXF)

## Turing hyper-parameters
ρ = 0.9ones(dP - dimQ())
tuned = tuning_hyperparameter(yields, macros, ρ)

p = 2
q = [0.06043749644018473, 0.0033752554401493304,
    0.21413614787951749, 0.03591711110516522]
ν0 = 42.00017952629186
Ω0 = [6.349207374193957, 1.7587741155165533,
    0.3123599833935631, 0.36972726334003414,
    0.37097564991453486, 0.3412247322702357,
    0.42068553869353353, 0.31497092514479114,
    0.3653730469344294, 0.30128527979285447,
    0.42374975219081756, 0.38336999424288015,
    0.36024771442074255, 0.42632632882095545,
    0.37195027755690696, 0.4129211810390276,
    0.43295669089475974, 0.34432776918391284,
    0.3749625127396211, 0.3648218490360464]

## Estimating
iteration = 10_000
saved_θ = posterior_sampler(yields, macros, τₙ, ρ, iteration, tuned)
saved_θ = saved_θ[round(Int, 0.1iteration):end]
saved_θ, accept_rate = stationary_θ(saved_θ)
sparse_θ, trace_λ, trace_sparsity = sparsify_precision(saved_θ, yields, macros, τₙ)
saved_Xθ = PCs_2_latents(saved_θ, yields, τₙ)
saved_TP = TP(120, τₙ, saved_θ, yields, macros)

predicted = zeros(size(yields, 1), dP)
for t = (p+4):(size(yields, 1)-1)
    local prediction = scenario_sampler([], 120, 1, saved_θ, yields[1:t, :], macros[1:t, :], τₙ)
    predicted[t+1, :] = mean(load_object(prediction, "predicted_factors"))[end, :]
end
factors = [PCA(yields, p)[1] macros]
plot(factors[p+5:end, 10])
plot!(predicted[p+5:end, 10])

data = [yields macros]
factors = [PCA(yields, p)[1] macros]
predicted_yields = zeros(size(yields, 1), length(τₙ))
predicted_factors = zeros(size(yields, 1), dP)
for t = (p+4):(size(yields, 1)-1)
    scene = []
    S = zeros(1, dP - dimQ() + length(τₙ) + 1)
    S[1, 20] = 1
    S[1, end] = data[t+1, 20]
    # S[2, 20] = 1
    # S[2, end] = data[t+1, 20]
    push!(scene, S)

    prediction = scenario_sampler(scene, 120, 1, saved_θ, yields[1:t, :], macros[1:t, :], τₙ)
    predicted_yields[t+1, :] = mean(load_object(prediction, "predicted_yields"))[end, :]
    predicted_factors[t+1, :] = mean(load_object(prediction, "predicted_factors"))[end, :]
end
plot(yields[p+10:end, end])
plot!(predicted_yields[p+10:end, end])

data = [yields macros]
scene = []
for h = 1:4
    S = zeros(1, dP - dimQ() + length(τₙ) + 1)
    S[1, 20] = 1
    S[1, end] = data[size(data, 1)-4+h, 20]
    # S[2, 20] = 1
    # S[2, end] = data[t+1, 20]
    push!(scene, S)
end
prediction = scenario_sampler(scene, 120, 8, saved_θ, yields[1:size(data, 1)-4, :], macros[1:size(data, 1)-4, :], τₙ)
aux = mean(load_object(prediction, "predicted_TP"))
plot(aux[1:4])
aux = mean(load_object(saved_TP, "TP"))
plot!(aux[end-3:end])