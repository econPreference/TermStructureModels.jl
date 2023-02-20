using Random, Distributions, LinearAlgebra
using GDTSM
Random.seed!(111)

## Simulating sample data
T = 500
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
#p, q, ν0, Ω0 = Tuning_Hyperparameter(yields, macros, ρ)
p = 2
q = [0.05248138313452114, 0.003540122464956814, 0.14205717997555042, 0.042771927731284146]
ν0 = 95.70620935078027
Ω0 = [43.034108428186734,
    5.322912137790291,
    2.2820088994392527,
    0.9918603760286976,
    1.0119737576512369,
    0.9088662494440232,
    1.0732609254891574,
    0.92413779913074,
    0.9572280047847496,
    0.938077096580834,
    1.0920959670316854,
    1.0357985890347834,
    0.9741068695062206,
    1.1164122345672007,
    0.9626589645970145,
    1.1167461560039285,
    1.0999319262657223,
    0.9676818663282197,
    1.0593774331335413,
    0.9316851593997177]

## Estimating
iteration = 10000
saved_θ = posterior_sampler(yields, macros, τₙ, ρ, iteration; p, q, ν0, Ω0)
saved_θ = saved_θ[round(Int, 0.1iteration):end]

stat, accept_rate = stationary_saved_θ(saved_θ)
TPs = TP(120, τₙ, stat, yields, macros)
TPm = load_object(TPs, "TP")
