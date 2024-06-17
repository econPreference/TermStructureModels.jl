## Setting
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
Pkg.precompile()
using TermStructureModels, ProgressMeter, Distributions, LinearAlgebra

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

est, var, opt, mle_est = MLE(yields, macros, tau_n, p; init_kappaQ=0.0609, iterations=20000)
param = latentspace([mle_est], yields, tau_n; data_scale=1200)[1]