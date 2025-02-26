#This module includes analytical results of our no-arbitrage Gaussian dynamic term structure model. For a specific theoretical settings, see our paper.

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
        idx = findall(x -> -x < 0.001, diff(kappaQ))
        if !isempty(idx)
            for i in eachindex(idx)
                X[idx[i], idx[i]+1] = 1
            end
        end
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

"""
    term_premium(tau_interest, tau_n, saved_params, yields, macros; data_scale=1200)
This function generates posterior samples of the term premiums.
# Input 
- maturity of interest `tau_interest` for Calculating `TP`
- `saved_params` from function `posterior_sampler`
# Output
- `Vector{TermPremium}(, iteration)`
- Outputs exclude initial observations.
"""
function term_premium(tau_interest, tau_n, saved_params, yields, macros; data_scale=1200)

    iteration = length(saved_params)
    saved_TP = Vector{TermPremium}(undef, iteration)

    dQ = dimQ() + size(yields, 2) - length(tau_n)
    dP = size(saved_params[:phi][1], 1)
    p = Int((size(saved_params[:phi][1], 2) - 1) / dP - 1)
    PCs, ~, Wₚ, ~, mean_PCs = PCA(yields, p)
    T = size(PCs, 1)

    if isempty(macros)
        indfactors = copy(PCs)
    else
        indfactors = [PCs macros]
    end

    factors = Matrix{Float64}(undef, T - p, dP * p + dP)
    factors[:, end-dP+1:end] .= 1
    for i in 0:p-1
        factors[:, dP*i+1:dP*(i+1)] = indfactors[p+1-i:end-i, :]
    end

    prog = Progress(iteration; dt=5, desc="term_premium...")
    Threads.@threads for iter in 1:iteration

        kappaQ = saved_params[:kappaQ][iter]
        kQ_infty = saved_params[:kQ_infty][iter]
        phi = saved_params[:phi][iter]
        varFF = saved_params[:varFF][iter]

        phi0, C = phi_2_phi₀_C(; phi)
        phi0 = C \ phi0
        KPF = phi0[:, 1]
        GPFF = phi0[:, 2:end]
        OmegaFF = (C \ diagm(varFF)) / C'
        if p == 1
            GP = [GPFF diagm(KPF)
                zeros(dP, dP) I(dP)]
        else
            GP = [GPFF diagm(KPF)
                I(dP * p - dP) zeros(dP * p - dP, 2dP)
                zeros(dP, dP * p) I(dP)]
        end
        Gpower = Array{Float64}(undef, size(GP, 1), size(GP, 2), length(tau_interest))
        Gpower_ind = I(size(GP, 1))
        Gpower_sum = I(size(GP, 1))
        for i in 1:tau_interest[end]
            if i ∈ tau_interest
                idx = findall(x -> x == i, tau_interest)
                Gpower[:, :, idx] = Gpower_sum ./ i
            end
            Gpower_ind *= GP
            Gpower_sum += Gpower_ind
        end

        bτ_ = bτ(tau_n[end]; kappaQ, dQ)
        Bₓ_ = Bₓ(bτ_, tau_n)
        T1X_ = T1X(Bₓ_, Wₚ)

        aτ_ = aτ(tau_n[end], bτ_, tau_n, Wₚ; kQ_infty, ΩPP=OmegaFF[1:dQ, 1:dQ], data_scale)
        Aₓ_ = Aₓ(aτ_, tau_n)
        T0P_ = T0P(T1X_, Aₓ_, Wₚ, mean_PCs)

        const_EH = Matrix{Float64}(undef, T - p, length(tau_interest))
        timevarying_EH = Array{Float64}(undef, T - p, p * dP + dP, length(tau_interest))
        fl_EH = Matrix{Float64}(undef, p * dP + dP, length(tau_interest))

        const_EH .= aτ_[1]
        for i in axes(fl_EH, 2)
            fl_EH[:, i] = bτ_[:, 1]' * [I(dQ) zeros(dQ, p * dP + dP - dQ)] * Gpower[:, :, i]
            timevarying_EH[:, :, i] = factors .* fl_EH[:, i]'
            const_EH[:, i] += sum(timevarying_EH[:, end-dP+1:end, i][:, :], dims=2)[:, 1]
        end
        timevarying_EH = timevarying_EH[:, 1:dP*p, :]
        fl_EH = fl_EH[1:dP*p, :]
        EH = const_EH + sum(timevarying_EH, dims=2)[:, 1]

        const_TP = -1 .* const_EH
        fl_TP = -1 .* fl_EH
        timevarying_TP = -1 .* timevarying_EH
        for i in axes(const_TP, 2)
            const_TP[:, i] .+= aτ_[Int(tau_interest[i])] + bτ_[:, Int(tau_interest[i])]' * T0P_ |> x -> x / tau_interest[i]
        end
        fl_TP_sub = bτ_[:, Int.(tau_interest)]' / T1X_ |> x -> x ./ tau_interest
        fl_TP[1:dQ, :] += fl_TP_sub'
        for i in axes(timevarying_TP, 3)
            timevarying_TP[:, 1:dQ, i] .+= PCs[p+1:end, :] .* fl_TP_sub[i, :]'
        end
        TP = const_TP + timevarying_TP

        saved_TP[iter] = TermPremium(TP=copy(TP),
            EH=copy(EH),
            factorloading_TP=copy(fl_TP),
            factorloading_EH=copy(fl_EH),
            timevarying_TP=copy(timevarying_TP),
            timevarying_EH=copy(timevarying_EH),
            const_TP=copy(const_TP),
            const_EH=copy(const_EH)
        )

        next!(prog)
    end
    finish!(prog)

    return saved_TP

end

"""
    _termPremium(τ, PCs, macros, bτ_, T0P_, T1X_; kappaQ, kQ_infty, KPF, GPFF, ΩPP, data_scale)
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
function _termPremium(τ, PCs, macros, bτ_, T0P_, T1X_; kappaQ, kQ_infty, KPF, GPFF, ΩPP, data_scale)

    T1P_ = inv(T1X_)
    # Jensen's Ineqaulity term
    jensen = 0
    for i = 1:(τ-1)
        jensen += jensens_inequality(i + 1, bτ_, T1X_; ΩPP, data_scale)
    end
    jensen /= -τ

    # Constant term
    dQ = size(ΩPP, 1)
    KQ_X = zeros(dQ)
    KQ_X[1] = copy(kQ_infty)
    KQ_P = T1X_ * (KQ_X + (GQ_XX(; kappaQ) - I(dQ)) * T0P_)
    λₚ = KPF[1:dQ] - KQ_P
    const_TP = sum(bτ_[:, 1:(τ-1)], dims=2)' * (T1P_ * λₚ)
    const_TP = -const_TP[1] / τ

    # Time-varying part
    dP = size(GPFF, 1)
    p = Int(size(GPFF, 2) / dP)
    T = size(PCs, 1) # time series length including intial conditions
    timevarying_TP = zeros(T - p, dP) # time-varying part is seperated to see the individual contribution of each priced factor. So, the column length is dP.

    GQ_PP = T1X_ * GQ_XX(; kappaQ) * T1P_
    Λ_PF = GPFF[1:dQ, :]
    Λ_PF[1:dQ, 1:dQ] -= GQ_PP
    T1P_Λ_PF = T1P_ * Λ_PF

    if isempty(macros)
        factors = copy(PCs)
    else
        factors = [PCs macros]
    end
    for t in (p+1):T # ranges for the dependent variables. The whole range includes initial observations.
        # prediction part
        predicted_X = factors[t:-1:1, :]
        for horizon = 1:(τ-2)
            regressors = vec(predicted_X[1:p, :]')
            predicted = KPF + GPFF * regressors
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
    latentspace(saved_params, yields, tau_n; data_scale=1200)
This function translates the principal components state space into the latent factor state space. 
# Input
- `data_scale::scalar`: In typical affine term structure model, theoretical yields are in decimal and not annualized. But, for convenience(public data usually contains annualized percentage yields) and numerical stability, we sometimes want to scale up yields, so want to use (`data_scale`*theoretical yields) as variable `yields`. In this case, you can use `data_scale` option. For example, we can set `data_scale = 1200` and use annualized percentage monthly yields as `yields`.
# Output
- `Vector{LatentSpace}(, iteration)`
- latent factors contain initial observations.
"""
function latentspace(saved_params, yields, tau_n; data_scale=1200)

    iteration = length(saved_params)
    saved_params_latent = Vector{LatentSpace}(undef, iteration)
    prog = Progress(iteration; dt=5, desc="latentspace...")
    Threads.@threads for iter in 1:iteration

        kappaQ = saved_params[:kappaQ][iter]
        kQ_infty = saved_params[:kQ_infty][iter]
        phi = saved_params[:phi][iter]
        varFF = saved_params[:varFF][iter]

        phi0, C = phi_2_phi₀_C(; phi)
        phi0 = C \ phi0
        KPF = phi0[:, 1]
        GPFF = phi0[:, 2:end]
        OmegaFF = (C \ diagm(varFF)) / C'

        latent, kappaQ, kQ_infty, KPXF, GPXFXF, OmegaXFXF = PCs_2_latents(yields, tau_n; kappaQ, kQ_infty, KPF, GPFF, OmegaFF, data_scale)
        saved_params_latent[iter] = LatentSpace(latents=copy(latent), kappaQ=copy(kappaQ), kQ_infty=copy(kQ_infty), KPXF=copy(KPXF), GPXFXF=copy(GPXFXF), OmegaXFXF=copy(OmegaXFXF))

        next!(prog)
    end
    finish!(prog)

    return saved_params_latent
end

"""
    PCs_2_latents(yields, tau_n; kappaQ, kQ_infty, KPF, GPFF, OmegaFF, data_scale)
Notation `XF` is for the latent factor space and notation `F` is for the PC state space.
# Input
- `data_scale::scalar`: In typical affine term structure model, theoretical yields are in decimal and not annualized. But, for convenience(public data usually contains annualized percentage yields) and numerical stability, we sometimes want to scale up yields, so want to use (`data_scale`*theoretical yields) as variable `yields`. In this case, you can use `data_scale` option. For example, we can set `data_scale = 1200` and use annualized percentage monthly yields as `yields`.
# Output(6)
`latent`, `kappaQ`, `kQ_infty`, `KPXF`, `GPXFXF`, `OmegaXFXF`
- latent factors contain initial observations.
"""
function PCs_2_latents(yields, tau_n; kappaQ, kQ_infty, KPF, GPFF, OmegaFF, data_scale)

    dP = size(OmegaFF, 1)
    dQ = dimQ() + size(yields, 2) - length(tau_n)
    dM = dP - dQ # of macro variables
    p = Int(size(GPFF, 2) / dP)
    PCs, ~, Wₚ, ~, mean_PCs = PCA(yields, p)

    # statistical Parameters
    bτ_ = bτ(tau_n[end]; kappaQ, dQ)
    Bₓ_ = Bₓ(bτ_, tau_n)
    T1X_ = T1X(Bₓ_, Wₚ)
    T1P_ = inv(T1X_)

    aτ_ = aτ(tau_n[end], bτ_, tau_n, Wₚ; kQ_infty, ΩPP=OmegaFF[1:dQ, 1:dQ], data_scale)
    Aₓ_ = Aₓ(aτ_, tau_n)
    T0P_ = T0P(T1X_, Aₓ_, Wₚ, mean_PCs)

    OmegaXFXF = similar(OmegaFF)
    OmegaXFXF[1:dQ, 1:dQ] = (T1P_ * OmegaFF[1:dQ, 1:dQ]) * T1P_'
    OmegaXFXF[(dQ+1):end, 1:dQ] = OmegaFF[(dQ+1):end, 1:dQ] * T1P_'
    OmegaXFXF[1:dQ, (dQ+1):end] = OmegaXFXF[(dQ+1):end, 1:dQ]'
    OmegaXFXF[(dQ+1):end, (dQ+1):end] = OmegaFF[(dQ+1):end, (dQ+1):end]

    GPXFXF = copy(GPFF)
    GₚXX_sum = zeros(dQ, dQ)
    GₚMX_sum = zeros(dM, dQ)
    for l in 1:p
        GₚXX_l = T1P_ * GPFF[1:dQ, (dP*(l-1)+1):(dP*(l-1)+dQ)] * T1X_
        GPXFXF[1:dQ, (dP*(l-1)+1):(dP*(l-1)+dQ)] = copy(GₚXX_l)
        GₚXX_sum += GₚXX_l

        GₚMX_l = GPFF[(dQ+1):end, (dP*(l-1)+1):(dP*(l-1)+dQ)] * T1X_
        GPXFXF[(dQ+1):end, (dP*(l-1)+1):(dP*(l-1)+dQ)] = copy(GₚMX_l)
        GₚMX_sum += GₚMX_l

        GPXFXF[1:dQ, (dP*(l-1)+dQ+1):(dP*l)] = T1P_ * GPFF[1:dQ, (dP*(l-1)+dQ+1):(dP*l)]
    end

    KPXF = similar(KPF)
    KPXF[1:dQ] = T1P_ * KPF[1:dQ] + (I(dQ) - GₚXX_sum) * T0P_
    KPXF[(dQ+1):end] = KPF[(dQ+1):end] - GₚMX_sum * T0P_

    # Latent factors
    latent = (T0P_ .+ T1P_ * PCs')'

    return latent, kappaQ, kQ_infty, KPXF, GPXFXF, OmegaXFXF

end

"""
    fitted_YieldCurve(τ0, saved_latent_params::Vector{LatentSpace}; data_scale=1200)
It generates a fitted yield curve.
# Input
- `τ0` is a set of maturities of interest. `τ0` does not need to be the same as the one used for the estimation.
- `saved_latent_params` is a transformed posterior sample using function `latentspace`.
# Output
- `Vector{YieldCurve}(,`# of iteration`)`
- `yields` and `latents` contain initial observations.
"""
function fitted_YieldCurve(τ0, saved_latent_params::Vector{LatentSpace}; data_scale=1200)

    dQ = saved_latent_params[:latents][1] |> x -> size(x, 2)
    iteration = length(saved_latent_params)
    YieldCurve_ = Vector{YieldCurve}(undef, iteration)
    prog = Progress(iteration; dt=5, desc="fitted_YieldCurve...")
    Threads.@threads for iter in 1:iteration

        latents = saved_latent_params[:latents][iter]
        kappaQ = saved_latent_params[:kappaQ][iter]
        kQ_infty = saved_latent_params[:kQ_infty][iter]
        OmegaXFXF = saved_latent_params[:OmegaXFXF][iter]

        # statistical Parameters
        bτ_ = bτ(τ0[end]; kappaQ, dQ)
        Bₓ_ = Bₓ(bτ_, τ0)
        aτ_ = aτ(τ0[end], bτ_; kQ_infty, ΩXX=OmegaXFXF[1:dQ, 1:dQ], data_scale)
        Aₓ_ = Aₓ(aτ_, τ0)

        YieldCurve_[iter] = YieldCurve(
            latents=copy(latents),
            yields=copy((Aₓ_ .+ Bₓ_ * latents')' |> Matrix),
            intercept=copy(Aₓ_),
            slope=copy(Bₓ_)
        )

        next!(prog)
    end
    finish!(prog)

    return YieldCurve_
end

"""
    PCA(yields, p, proxies=[]; rescaling=false, dQ=[])
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
function PCA(yields, p, proxies=[]; rescaling=false, dQ=[])

    if isempty(dQ)
        dQ = dimQ()
    end
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