## Setting
using Pkg, Revise
Pkg.activate(@__DIR__)
Pkg.instantiate()
Pkg.precompile()
using TermStructureModels, ProgressMeter, Distributions, LinearAlgebra, Distributions, Plots

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
yields, latents, macros = generative(T, dP, tau_n, p, 0.01; kappaQ, kQ_infty, KPXF, GPXFXF, OmegaXFXF)

## Turing hyper-parameters
diag_G = diag_G[dimQ()+1:end]
rho = zeros(dP - dimQ())
rho[diag_G.>0.5] .= 1.0
medium_tau = collect(24:3:60)
medium_tau_pr = [truncated(Normal(1, 0.1), -1, 1), truncated(Normal(1, 0.1), -1, 1), truncated(Normal(1, 0.1), -1, 1)]
std_kQ_infty = 0.2
tuned, opt = tuning_hyperparameter(yields, macros, tau_n, rho; std_kQ_infty, medium_tau, medium_tau_pr)
p = tuned.p

## Estimating
iteration = 10_000
saved_θ, acceptPrMH = posterior_sampler(yields, macros, tau_n, rho, iteration, tuned; medium_tau, std_kQ_infty, medium_tau_pr)
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
# Compare observations and fitted variables of the reducedform transition equation
fitted_samples = Vector{Matrix}(undef, length(reduced_θ))
@showprogress for iter in eachindex(reduced_θ)
    KPF = reduced_θ[iter].KPF
    GPFF = reduced_θ[iter].GPFF
    OmegaFF = reduced_θ[iter].OmegaFF

    fitted = zeros(size(PCs, 1), dP)
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

saved_params = saved_θ
iter = 3
kappaQ = saved_params[:kappaQ][iter]
kQ_infty = saved_params[:kQ_infty][iter]
phi = saved_params[:phi][iter]
varFF = saved_params[:varFF][iter]

phi0, C = phi_2_phi₀_C(; phi)
phi0 = C \ phi0
KPF = phi0[:, 1]
GPFF = phi0[:, 2:end]
OmegaFF = (C \ diagm(varFF)) / C'

dP = size(OmegaFF, 1)
dQ = dimQ() + size(yields, 2) - length(tau_n)
dM = dP - dQ # of macro variables
p = Int(size(GPFF, 2) / dP)
PCs = PCA(yields, p)[1]
Wₚ = PCA(yields, p)[3]
mean_PCs = PCA(yields, p)[5]


# statistical Parameters
bτ_ = bτ(tau_n[end]; kappaQ, dQ)
Bₓ_ = Bₓ(bτ_, tau_n)
T1X_ = T1X(Bₓ_, Wₚ)
T1P_ = inv(T1X_)
data_scale = 1200
aτ_ = aτ(tau_n[end], bτ_, tau_n, Wₚ; kQ_infty, ΩPP=OmegaFF[1:dQ, 1:dQ], data_scale)
Aₓ_ = Aₓ(aτ_, tau_n)
T0P_ = T0P(T1X_, Aₓ_, Wₚ, mean_PCs)

OmegaXFXF = similar(OmegaFF)
OmegaXFXF[1:dQ, 1:dQ] = (T1P_ * OmegaFF[1:dQ, 1:dQ]) * T1P_'
OmegaXFXF[(dQ+1):end, 1:dQ] = OmegaFF[(dQ+1):end, 1:dQ] * T1P_'
OmegaXFXF[1:dQ, (dQ+1):end] = OmegaXFXF[(dQ+1):end, 1:dQ]'
OmegaXFXF[(dQ+1):end, (dQ+1):end] = OmegaFF[(dQ+1):end, (dQ+1):end]

GPXFXF = deepcopy(GPFF)
GₚXX_sum = zeros(dQ, dQ)
GₚMX_sum = zeros(dM, dQ)
for l in 1:p
    GₚXX_l = T1P_ * GPFF[1:dQ, (dP*(l-1)+1):(dP*(l-1)+dQ)] * T1X_
    GPXFXF[1:dQ, (dP*(l-1)+1):(dP*(l-1)+dQ)] = deepcopy(GₚXX_l)
    GₚXX_sum += GₚXX_l

    GₚMX_l = GPFF[(dQ+1):end, (dP*(l-1)+1):(dP*(l-1)+dQ)] * T1X_
    GPXFXF[(dQ+1):end, (dP*(l-1)+1):(dP*(l-1)+dQ)] = deepcopy(GₚMX_l)
    GₚMX_sum += GₚMX_l

    GPXFXF[1:dQ, (dP*(l-1)+dQ+1):(dP*l)] = T1P_ * GPFF[1:dQ, (dP*(l-1)+dQ+1):(dP*l)]
end

KPXF = similar(KPF)
KPXF[1:dQ] = T1P_ * KPF[1:dQ] + (I(dQ) - GₚXX_sum) * T0P_
KPXF[(dQ+1):end] = KPF[(dQ+1):end] - GₚMX_sum * T0P_

# Latent factors
latent = (T0P_ .+ T1P_ * PCs')'


"""
    GQ_XX(; kappaQ)
`kappaQ` governs a conditional mean of the Q-dynamics of `X`, and its slope matrix has a restricted form. This function shows that restricted form.
# Output
- slope matrix of the Q-conditional mean of `X`
"""
function GQ_XX(; kappaQ)
    if length(kappaQ) == 1
        X = [1 0 0
            0 exp(-kappaQ) 1
            0 0 exp(-kappaQ)]
    else
        X = diagm(kappaQ)
    end
    return X
end

"""
    dimQ()
It returns the dimension of Q-dynamics under the standard ATSM.
"""
function dimQ()
    return 3
end

"""
    bτ(N; kappaQ, dQ)
It solves the difference equation for `bτ`.
# Output
- for maturity `i`, `bτ[:, i]` is a vector of factor loadings.
"""
function bτ(N; kappaQ, dQ)
    GQ_XX_ = GQ_XX(; kappaQ)
    ι = ones(dQ)

    b = ones(dQ, N) # factor loadings
    for i in 2:N
        b[:, i] = ι + GQ_XX_' * b[:, i-1]
    end

    return b
end

"""
    Bₓ(bτ_, tau_n)
# Input
- `bτ_` is an output of function `bτ`.
# Output
- `Bₓ`
"""
function Bₓ(bτ_, tau_n)
    return (bτ_[:, tau_n] ./ tau_n')'
end

"""
    T1X(Bₓ_, Wₚ)
# Input
- `Bₓ_` if an output of function `Bₓ`.
# Output
- `T1X`
"""
function T1X(Bₓ_, Wₚ)
    return Wₚ * Bₓ_
end

"""
    aτ(N, bτ_, tau_n, Wₚ; kQ_infty, ΩPP, data_scale)
    aτ(N, bτ_; kQ_infty, ΩXX, data_scale)
The function has two methods(multiple dispatch). 
# Input
- When `Wₚ` ∈ arguments: It calculates `aτ` using `ΩPP`. 
- Otherwise: It calculates `aτ` using `ΩXX = OmegaXFXF[1:dQ, 1:dQ]`, so parameters are in the latent factor space. So, we do not need `Wₚ`.
- `bτ_` is an output of function `bτ`.
- `data_scale::scalar`: In typical affine term structure model, theoretical yields are in decimal and not annualized. But, for convenience(public data usually contains annualized percentage yields) and numerical stability, we sometimes want to scale up yields, so want to use (`data_scale`*theoretical yields) as variable `yields`. In this case, you can use `data_scale` option. For example, we can set `data_scale = 1200` and use annualized percentage monthly yields as `yields`.
# Output
- `Vector(Float64)(aτ,N)`
- For `i`'th maturity, `Output[i]` is the corresponding `aτ`.
"""
function aτ(N, bτ_, tau_n, Wₚ; kQ_infty, ΩPP, data_scale)

    dQ = size(ΩPP, 1)

    a = zeros(N)
    T1X_ = T1X(Bₓ(bτ_, tau_n), Wₚ)
    for i in 2:N
        a[i] = a[i-1] - jensens_inequality(i, bτ_, T1X_; ΩPP, data_scale) + bτ_[:, i-1]' * [kQ_infty; zeros(dQ - 1)]
    end

    return a
end
function aτ(N, bτ_; kQ_infty, ΩXX, data_scale)
    dQ = size(ΩXX, 1)

    a = zeros(N)
    for i in 2:N
        J = 0.5 * ΩXX
        J = bτ_[:, i-1]' * J * bτ_[:, i-1]
        J /= data_scale

        a[i] = a[i-1] - J + bτ_[:, i-1]' * [kQ_infty; zeros(dQ - 1)]
    end

    return a
end

"""
    jensens_inequality(τ, bτ_, T1X_; ΩPP, data_scale)
This function evaluate the Jensen's Ineqaulity term. All term is invariant with respect to the `data_scale`, except for this Jensen's inequality term. So, we need to scale down the term by `data_scale`.
# Output
- Jensen's Ineqaulity term for `aτ` of maturity `τ`.
"""
function jensens_inequality(τ, bτ_, T1X_; ΩPP, data_scale)
    J = 0.5 * ΩPP
    J = (T1X_ \ J) / (T1X_')
    J = bτ_[:, τ-1]' * J * bτ_[:, τ-1]
    J /= data_scale

    return J
end

"""
    Aₓ(aτ_, tau_n)
# Input
- `aτ_` is an output of function `aτ`.
# Output
- `Aₓ`
"""
function Aₓ(aτ_, tau_n)
    return aτ_[tau_n] ./ tau_n
end

"""
    T0P(T1X_, Aₓ_, Wₚ, c)
# Input
- `T1X_` and `Aₓ_` are outputs of function `T1X` and `Aₓ`, respectively. `c` is a sample mean of `PCs`.
# Output
- `T0P`
"""
function T0P(T1X_, Aₓ_, Wₚ, c)
    return -T1X_ \ (Wₚ * Aₓ_ - c)
end

"""
    Aₚ(Aₓ_, Bₓ_, T0P_, Wₒ)
# Input
- `Aₓ_`, `Bₓ_`, and `T0P_` are outputs of function `Aₓ`, `Bₓ`, and `T0P`, respectively.
# Output
- `Aₚ`
"""
function Aₚ(Aₓ_, Bₓ_, T0P_, Wₒ)
    return Wₒ * (Aₓ_ + Bₓ_ * T0P_)
end

"""
    Bₚ(Bₓ_, T1X_, Wₒ)
# Input
- `Bₓ_` and `T1X_` are outputs of function `Bₓ` and `T1X`, respectively.
# Output
- `Bₚ`
"""
function Bₚ(Bₓ_, T1X_, Wₒ)
    return (Wₒ * Bₓ_) / T1X_
end
