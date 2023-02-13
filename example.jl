using Random, Distributions, LinearAlgebra, Pkg
using Plots
Pkg.build()
using GDTSM
Random.seed!(111)

## Simulating sample data
T = 500
dP = 20
τₙ = [1, 3, 6, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120]
p = 2

κQ = 0.0609
kQ_infty = 0
KₚXF = zeros(dP)
ΩXFXF = rand(InverseWishart(dP + 2, Matrix(0.1I(dP))))

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
yields, latent, macros = generative(T, dP, τₙ, p; κQ, kQ_infty, KₚXF, GₚXFXF, ΩXFXF)

## Turing hyper-parameters
ρ = 0.9ones(dP - dimQ())
# p, q, ν0, Ω0 = Tuning_Hyperparameter(yields, macros, ρ)

p = 2
q = [2.0, 0.40760009069570213, 0.10812139087863065, 2.5764361584205435]
ν0 = 90.6343749997971
Ω0 = [66.28982477713093, 1.7365544902786016,
    3.3875037635652863, 0.059353631542339434,
    0.0993516982116516, 0.08527389231399178,
    0.05796496507324872, 0.0955757129295007,
    0.09132632490953446, 0.13404620260739042,
    0.12935042767792854, 0.12432032479052266,
    0.05171529715076053, 0.1914572752840065,
    0.05996036736993146, 0.11873488719949804,
    0.10975787538377273, 0.15509394037161034,
    0.6319034301062262, 0.14768363573379523]

## Estimating
iteration = 100
saved_θ = sampling_GDTSM(yields, macros, τₙ, ρ, iteration; p, q, ν0, Ω0)
saved_θX = PCs2latents(saved_θ, yields, τₙ)

mean(load_object(saved_θ, "ψ"))