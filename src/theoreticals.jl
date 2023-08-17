#This module includes analytical results of our no-arbitrage Gaussian dynamic term structure model. For a specific theoretical settings, see our paper.

"""
    GQ_XX(; κQ)
`κQ` governs a conditional mean of the Q-dynamics of `X`, and its slope matrix has a restricted form. This function shows that restricted form.
# Output
- slope matrix of the Q-conditional mean of `X`
"""
function GQ_XX(; κQ)
    X = [1 0 0
        0 exp(-κQ) 1
        0 0 exp(-κQ)]
    return X
end

"""
    dimQ()
It returns the dimension of Q-dynamics.
"""
function dimQ()
    return 3
end

"""
    bτ(N; κQ)
It solves the difference equation for `bτ`.
# Output
- for maturity `i`, `bτ[:, i]` is a vector of factor loadings.
"""
function bτ(N; κQ)
    dQ = dimQ()
    GQ_XX_ = GQ_XX(; κQ)
    ι = ones(dQ)

    b = ones(dQ, N) # factor loadings
    for i in 2:N
        b[:, i] = ι + GQ_XX_' * b[:, i-1]
    end

    return b
end

"""
    Bₓ(bτ_, τₙ)
# Input
- `bτ_` is an output of function `bτ`.
# Output
- `Bₓ`
"""
function Bₓ(bτ_, τₙ)
    return (bτ_[:, τₙ] ./ τₙ')'
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
    aτ(N, bτ_, τₙ, Wₚ; kQ_infty, ΩPP, data_scale)
    aτ(N, bτ_; kQ_infty, ΩXX, data_scale)
The function has two methods(multiple dispatch). 
# Input
- When `Wₚ` ∈ arguments: It calculates `aτ` using `ΩPP`. 
- Otherwise: It calculates `aτ` using `ΩXX = ΩXFXF[1:dQ, 1:dQ]`, so parameters are in the latent factor space. So, we do not need `Wₚ`.
- `bτ_` is an output of function `bτ`.
- `data_scale::scalar`: In typical affine term structure model, theoretical yields are in decimal and not annualized. But, for convenience(public data usually contains annualized percentage yields) and numerical stability, we sometimes want to scale up yields, so want to use (`data_scale`*theoretical yields) as variable `yields`. In this case, you can use `data_scale` option. For example, we can set `data_scale = 1200` and use annualized percentage monthly yields as `yields`.
# Output
- `Vector(Float64)(aτ,N)`
- For `i`'th maturity, `Output[i]` is the corresponding `aτ`.
"""
function aτ(N, bτ_, τₙ, Wₚ; kQ_infty, ΩPP, data_scale)

    a = zeros(N)
    T1X_ = T1X(Bₓ(bτ_, τₙ), Wₚ)
    for i in 2:N
        a[i] = a[i-1] - jensens_inequality(i, bτ_, T1X_; ΩPP, data_scale) + (i - 1) * kQ_infty
    end

    return a
end
function aτ(N, bτ_; kQ_infty, ΩXX, data_scale)

    a = zeros(N)
    for i in 2:N
        J = 0.5 * ΩXX
        J = bτ_[:, i-1]' * J * bτ_[:, i-1]
        J /= data_scale

        a[i] = a[i-1] - J + (i - 1) * kQ_infty
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
    Aₓ(aτ_, τₙ)
# Input
- `aτ_` is an output of function `aτ`.
# Output
- `Aₓ`
"""
function Aₓ(aτ_, τₙ)
    return aτ_[τₙ] ./ τₙ
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

"""
    _termPremium(τ, PCs, macros, bτ_, T0P_, T1X_; κQ, kQ_infty, KₚF, GₚFF, ΩPP, data_scale)
This function calculates a term premium for maturity `τ`. 
# Input
- `data_scale::scalar` = In typical affine term structure model, theoretical yields are in decimal and not annualized. But, for convenience(public data usually contains annualized percentage yields) and numerical stability, we sometimes want to scale up yields, so want to use (`data_scale`*theoretical yields) as variable `yields`. In this case, you can use `data_scale` option. For example, we can set `data_scale = 1200` and use annualized percentage monthly yields as `yields`.
# Output(4)
`TP`, `timevarying_TP`, `const_TP`, `jensen`
- `TP`: term premium of maturity `τ`
- `timevarying_TP`: contributions of each `[PCs macros]` on `TP` at each time ``t`` (row: time, col: variable)
- `const_TP`: Constant part of `TP`
- `jensen`: Jensen's Ineqaulity part in `TP`
- Output excludes the time period for the initial observations.  
"""
function _termPremium(τ, PCs, macros, bτ_, T0P_, T1X_; κQ, kQ_infty, KₚF, GₚFF, ΩPP, data_scale)

    T1P_ = inv(T1X_)
    # Jensen's Ineqaulity term
    jensen = 0
    for i = 1:(τ-1)
        jensen += jensens_inequality(i + 1, bτ_, T1X_; ΩPP, data_scale)
    end
    jensen /= -τ

    # Constant term
    dQ = dimQ()
    KQ_X = zeros(dQ)
    KQ_X[1] = kQ_infty
    KQ_P = T1X_ * (KQ_X + (GQ_XX(; κQ) - I(dQ)) * T0P_)
    λₚ = KₚF[1:dQ] - KQ_P
    const_TP = sum(bτ_[:, 1:(τ-1)], dims=2)' * (T1P_ * λₚ)
    const_TP = -const_TP[1] / τ

    # Time-varying part
    dP = size(GₚFF, 1)
    p = Int(size(GₚFF, 2) / dP)
    T = size(PCs, 1) # time series length including intial conditions
    timevarying_TP = zeros(T - p, dP) # time-varying part is seperated to see the individual contribution of each priced factor. So, the column length is dP.

    GQ_PP = T1X_ * GQ_XX(; κQ) * T1P_
    Λ_PF = GₚFF[1:dQ, :]
    Λ_PF[1:dQ, 1:dQ] -= GQ_PP
    T1P_Λ_PF = T1P_ * Λ_PF

    if isempty(macros)
        factors = PCs
    else
        factors = [PCs macros]
    end
    for t in (p+1):T # ranges for the dependent variables. The whole range includes initial observations.
        # prediction part
        predicted_X = factors[t:-1:1, :]
        for horizon = 1:(τ-2)
            regressors = vec(predicted_X[1:p, :]')
            predicted = KₚF + GₚFF * regressors
            predicted_X = vcat(predicted', predicted_X)
        end
        reverse!(predicted_X, dims=1)

        # Calculate TP
        for i = 1:(τ-1)
            weight = bτ_[:, τ-i]' * (T1P_Λ_PF)
            for l = 1:p, j = 1:dP
                timevarying_TP[t-p, j] += weight[(l-1)*dP+j] * predicted_X[t+i-l, j] # first row in predicted_X = (time = t-p+1)
            end
        end
    end
    timevarying_TP /= -τ

    TP = sum(timevarying_TP, dims=2) .+ jensen .+ const_TP

    return TP, timevarying_TP, const_TP, jensen
end

"""
    term_premium(τ, τₙ, saved_θ, yields, macros; data_scale=1200)
This function generates posterior samples of the term premiums.
# Input 
- maturity of interest `τ` for Calculating `TP`
- `saved_θ` from function `posterior_sampler`
# Output
- `Vector{TermPremium}(, iteration)`
- Outputs exclude initial observations.
"""
function term_premium(τ, τₙ, saved_θ, yields, macros; data_scale=1200)

    iteration = length(saved_θ)
    saved_TP = Vector{TermPremium}(undef, iteration)

    dQ = dimQ()
    dP = size(saved_θ[:ϕ][1], 1)
    p = Int((size(saved_θ[:ϕ][1], 2) - 1) / dP - 1)
    PCs, ~, Wₚ, ~, mean_PCs = PCA(yields, p)

    prog = Progress(iteration; dt=5, desc="Calculating TPs...")
    Threads.@threads for iter in 1:iteration

        κQ = saved_θ[:κQ][iter]
        kQ_infty = saved_θ[:kQ_infty][iter]
        ϕ = saved_θ[:ϕ][iter]
        σ²FF = saved_θ[:σ²FF][iter]

        ϕ0, C = ϕ_2_ϕ₀_C(; ϕ)
        ϕ0 = C \ ϕ0
        KₚF = ϕ0[:, 1]
        GₚFF = ϕ0[:, 2:end]
        ΩFF = (C \ diagm(σ²FF)) / C'

        bτ_ = bτ(τₙ[end]; κQ)
        Bₓ_ = Bₓ(bτ_, τₙ)
        T1X_ = T1X(Bₓ_, Wₚ)

        aτ_ = aτ(τₙ[end], bτ_, τₙ, Wₚ; kQ_infty, ΩPP=ΩFF[1:dQ, 1:dQ], data_scale)
        Aₓ_ = Aₓ(aτ_, τₙ)
        T0P_ = T0P(T1X_, Aₓ_, Wₚ, mean_PCs)
        TP, timevarying_TP, const_TP, jensen = _termPremium(τ, PCs, macros, bτ_, T0P_, T1X_; κQ, kQ_infty, KₚF, GₚFF, ΩPP=ΩFF[1:dQ, 1:dQ], data_scale)

        saved_TP[iter] = TermPremium(TP=TP[:, 1], timevarying_TP=timevarying_TP, const_TP=const_TP, jensen=jensen)

        next!(prog)
    end
    finish!(prog)

    return saved_TP

end

"""
    latentspace(saved_θ, yields, τₙ; data_scale=1200)
This function translates the principal components state space into the latent factor state space. 
# Input
- `data_scale::scalar`: In typical affine term structure model, theoretical yields are in decimal and not annualized. But, for convenience(public data usually contains annualized percentage yields) and numerical stability, we sometimes want to scale up yields, so want to use (`data_scale`*theoretical yields) as variable `yields`. In this case, you can use `data_scale` option. For example, we can set `data_scale = 1200` and use annualized percentage monthly yields as `yields`.
# Output
- `Vector{LatentSpace}(, iteration)`
- latent factors contain initial observations.
"""
function latentspace(saved_θ, yields, τₙ; data_scale=1200)

    iteration = length(saved_θ)
    saved_θ_latent = Vector{LatentSpace}(undef, iteration)
    prog = Progress(iteration; dt=5, desc="Moving to the latent space...")
    Threads.@threads for iter in 1:iteration

        κQ = saved_θ[:κQ][iter]
        kQ_infty = saved_θ[:kQ_infty][iter]
        ϕ = saved_θ[:ϕ][iter]
        σ²FF = saved_θ[:σ²FF][iter]

        ϕ0, C = ϕ_2_ϕ₀_C(; ϕ)
        ϕ0 = C \ ϕ0
        KₚF = ϕ0[:, 1]
        GₚFF = ϕ0[:, 2:end]
        ΩFF = (C \ diagm(σ²FF)) / C'

        latent, κQ, kQ_infty, KₚXF, GₚXFXF, ΩXFXF = PCs_2_latents(yields, τₙ; κQ, kQ_infty, KₚF, GₚFF, ΩFF, data_scale)
        saved_θ_latent[iter] = LatentSpace(latents=latent, κQ=κQ, kQ_infty=kQ_infty, KₚXF=KₚXF, GₚXFXF=GₚXFXF, ΩXFXF=ΩXFXF)

        next!(prog)
    end
    finish!(prog)

    return saved_θ_latent
end

"""
    PCs_2_latents(yields, τₙ; κQ, kQ_infty, KₚF, GₚFF, ΩFF, data_scale)
Notation `XF` is for the latent factor space and notation `F` is for the PC state space.
# Input
- `data_scale::scalar`: In typical affine term structure model, theoretical yields are in decimal and not annualized. But, for convenience(public data usually contains annualized percentage yields) and numerical stability, we sometimes want to scale up yields, so want to use (`data_scale`*theoretical yields) as variable `yields`. In this case, you can use `data_scale` option. For example, we can set `data_scale = 1200` and use annualized percentage monthly yields as `yields`.
# Output(6)
`latent`, `κQ`, `kQ_infty`, `KₚXF`, `GₚXFXF`, `ΩXFXF`
- latent factors contain initial observations.
"""
function PCs_2_latents(yields, τₙ; κQ, kQ_infty, KₚF, GₚFF, ΩFF, data_scale)

    dP = size(ΩFF, 1)
    dQ = dimQ()
    dM = dP - dQ # of macro variables
    p = Int(size(GₚFF, 2) / dP)
    PCs, ~, Wₚ, ~, mean_PCs = PCA(yields, p)

    # statistical Parameters
    bτ_ = bτ(τₙ[end]; κQ)
    Bₓ_ = Bₓ(bτ_, τₙ)
    T1X_ = T1X(Bₓ_, Wₚ)
    T1P_ = inv(T1X_)

    aτ_ = aτ(τₙ[end], bτ_, τₙ, Wₚ; kQ_infty, ΩPP=ΩFF[1:dQ, 1:dQ], data_scale)
    Aₓ_ = Aₓ(aτ_, τₙ)
    T0P_ = T0P(T1X_, Aₓ_, Wₚ, mean_PCs)

    ΩXFXF = similar(ΩFF)
    ΩXFXF[1:dQ, 1:dQ] = (T1P_ * ΩFF[1:dQ, 1:dQ]) * T1P_'
    ΩXFXF[(dQ+1):end, 1:dQ] = ΩFF[(dQ+1):end, 1:dQ] * T1P_'
    ΩXFXF[1:dQ, (dQ+1):end] = ΩXFXF[(dQ+1):end, 1:dQ]'
    ΩXFXF[(dQ+1):end, (dQ+1):end] = ΩFF[(dQ+1):end, (dQ+1):end]

    GₚXFXF = deepcopy(GₚFF)
    GₚXX_sum = zeros(dQ, dQ)
    GₚMX_sum = zeros(dM, dQ)
    for l in 1:p
        GₚXX_l = T1P_ * GₚFF[1:dQ, (dP*(l-1)+1):(dP*(l-1)+dQ)] * T1X_
        GₚXFXF[1:dQ, (dP*(l-1)+1):(dP*(l-1)+dQ)] = GₚXX_l
        GₚXX_sum += GₚXX_l

        GₚMX_l = GₚFF[(dQ+1):end, (dP*(l-1)+1):(dP*(l-1)+dQ)] * T1X_
        GₚXFXF[(dQ+1):end, (dP*(l-1)+1):(dP*(l-1)+dQ)] = GₚMX_l
        GₚMX_sum += GₚMX_l

        GₚXFXF[1:dQ, (dP*(l-1)+dQ+1):(dP*l)] = T1P_ * GₚFF[1:dQ, (dP*(l-1)+dQ+1):(dP*l)]
    end

    KₚXF = similar(KₚF)
    KₚXF[1:dQ] = T1P_ * KₚF[1:dQ] + (I(dQ) - GₚXX_sum) * T0P_
    KₚXF[(dQ+1):end] = KₚF[(dQ+1):end] - GₚMX_sum * T0P_

    # Latent factors
    latent = (T0P_ .+ T1P_ * PCs')'

    return latent, κQ, kQ_infty, KₚXF, GₚXFXF, ΩXFXF

end

"""
    fitted_YieldCurve(τ0, saved_Xθ::Vector{LatentSpace}; data_scale=1200)
It generates a fitted yield curve.
# Input
- `τ0` is a set of maturities of interest. `τ0` does not need to be the same as the one used for the estimation.
- `saved_Xθ` is a transformed posterior sample using function `latentspace`.
# Output
- `Vector{YieldCurve}(,`# of iteration`)`
- `yields` and `latents` contain initial observations.
"""
function fitted_YieldCurve(τ0, saved_Xθ::Vector{LatentSpace}; data_scale=1200)

    dQ = dimQ()
    iteration = length(saved_Xθ)
    YieldCurve_ = Vector{YieldCurve}(undef, iteration)
    prog = Progress(iteration; dt=5, desc="Fitting yield curve...")
    Threads.@threads for iter in 1:iteration

        latents = saved_Xθ[:latents][iter]
        κQ = saved_Xθ[:κQ][iter]
        kQ_infty = saved_Xθ[:kQ_infty][iter]
        ΩXFXF = saved_Xθ[:ΩXFXF][iter]

        # statistical Parameters
        bτ_ = bτ(τ0[end]; κQ)
        Bₓ_ = Bₓ(bτ_, τ0)
        aτ_ = aτ(τ0[end], bτ_; kQ_infty, ΩXX=ΩXFXF[1:dQ, 1:dQ], data_scale)
        Aₓ_ = Aₓ(aτ_, τ0)

        YieldCurve_[iter] = YieldCurve(
            latents=latents,
            yields=(Aₓ_ .+ Bₓ_ * latents')' |> Matrix,
            intercept=Aₓ_,
            slope=Bₓ_
        )

        next!(prog)
    end
    finish!(prog)

    return YieldCurve_
end

"""
    PCA(yields, p, proxies=[]; rescaling=false)
It derives the principal components from `yields`.
# Input
- `yields[p+1:end, :]` is used to construct the affine transformation, and then all `yields[:,:]` are transformed into the principal components.
- Since signs of `PCs` is not identified, we use proxies to identify the signs. We flip `PCs` to make `cor(proxies[:, i]. PCs[:,i]) > 0`. If `proxies` is not given, we use the following proxies as a default: `[yields[:, end] yields[:, end] - yields[:, 1] 2yields[:, Int(floor(size(yields, 2) / 3))] - yields[:, 1] - yields[:, end]]`.
- `size(proxies) = (size(yields[p+1:end, :], 1), dQ)`
- If `rescaling == true`, all `PCs` and `OCs` are normalized to have an average std of yields.
# Output(4)
`PCs`, `OCs`, `Wₚ`, `Wₒ`, `mean_PCs`
- `PCs`, `OCs`: first `dQ` and the remaining principal components
- `Wₚ`, `Wₒ`: the rotation matrix for `PCs` and `OCs`, respectively
- `mean_PCs`: the mean of `PCs` before demeaned.
- `PCs` are demeaned.
"""
function PCA(yields, p, proxies=[]; rescaling=false)

    dQ = dimQ()
    ## z-score case
    # std_yields = yields[p+1:end, :] .- mean(yields[p+1:end, :], dims=1)
    # std_yields ./= std(yields[p+1:end, :], dims=1)
    # V = reverse(eigen(cov(std_yields)).vectors, dims=2)

    V = reverse(eigen(cov(yields[(p+1):end, :])).vectors, dims=2)
    Wₚ = V[:, 1:dQ]'
    Wₒ = V[:, (dQ+1):end]'

    PCs = (Wₚ * yields')' # Main dQ PCs
    OCs = (Wₒ * yields')' # remaining PCs

    #observables
    if isempty(proxies)
        proxies = [yields[:, end] yields[:, end] - yields[:, 1] 2yields[:, Int(floor(size(yields, 2) / 3))] - yields[:, 1] - yields[:, end]]
    end
    for i in axes(PCs, 2)
        sign_ = cor(proxies[:, i], PCs[:, i]) |> sign
        PCs[:, i] *= sign_
        Wₚ[i, :] *= sign_
    end

    if rescaling == false
        mean_PCs = mean(PCs[p+1:end, :], dims=1)
        PCs .-= mean_PCs

        return Matrix(PCs), Matrix(OCs), Wₚ, Wₒ, mean_PCs[1, :]
    else
        ## rescaling
        mean_std = mean(std(yields[(p+1):end, :], dims=1))
        scale_PCs = mean_std ./ std(PCs, dims=1)'
        scale_OCs = mean_std ./ std(OCs, dims=1)'

        PCs = Matrix((scale_PCs .* PCs')')
        mean_PCs = mean(PCs[p+1:end, :], dims=1)
        PCs .-= mean_PCs
        return PCs, Matrix((scale_OCs .* OCs')'), scale_PCs .* Wₚ, scale_OCs .* Wₒ, mean_PCs[1, :]
    end
end