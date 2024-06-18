## Setting
using Pkg, Revise
Pkg.activate(@__DIR__)
Pkg.instantiate()
Pkg.precompile()
using TermStructureModels, ProgressMeter, Distributions, LinearAlgebra, Distributions

## Simulating sample data
T = 1000
dP = 4
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
    global GPXFXF[1:dimQ(), 1:dimQ()] = GQ_XX(; kappaQ)
end

# Generating samples
yields, latents, macros = generative(T, dP, tau_n, p, 0.0001; kappaQ, kQ_infty, KPXF, GPXFXF, OmegaXFXF)

## Turing hyper-parameters
diag_G = diag_G[dimQ()+1:end]
rho = zeros(dP - dimQ())
rho[diag_G.>0.5] .= 1.0
medium_tau = collect(36:42)
medium_tau_pr = [truncated(Normal(1, 0.1), -1, 1), truncated(Normal(1, 0.1), -1, 1), truncated(Normal(1, 0.1), -1, 1)]
std_kQ_infty = 0.2
tuned, opt = tuning_hyperparameter(yields, macros, tau_n, rho; std_kQ_infty, medium_tau, medium_tau_pr)

## Estimating
iteration = 10_000
saved_θ, acceptPrMH = posterior_sampler(yields, macros, tau_n, rho, iteration, tuned; medium_tau, std_kQ_infty, medium_tau_pr)
saved_θ = saved_θ[round(Int, 0.1iteration):end]
saved_θ, accept_rate = erase_nonstationary_param(saved_θ)
ineff = ineff_factor(saved_θ)
saved_Xθ = latentspace(saved_θ, yields, τₙ)
saved_TP = term_premium(120, τₙ, saved_θ[1:50:end], yields, macros)
reduced_θ = reducedform(saved_θ, yields, macros, τₙ)
fits = fitted_YieldCurve(τₙ, saved_Xθ)
idx = 18
plot(mean(fits)[:yields][:, idx])
plot!(yields[:, idx])