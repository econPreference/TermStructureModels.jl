#This module includes analytical results of our no-arbitrage Gaussian dynamic term structure model. For a specific theoretical settings, see our paper.

"""
GQ_XX(; κQ)
* κQ governs a conditional mean of the Q-dynamics of X, and its slope matrix has a restricted form. This function shows that restricted form.
* Input: the DNS decay parameter
* Output: slope matrix of the Q-conditional mean of X
"""
function GQ_XX(; κQ)
    X = [1 0 0
        0 exp(-κQ) 1
        0 0 exp(-κQ)]
    return X
end

"""
dimQ()
* It returns the dimension of Q-dynamics.
"""
function dimQ()
    return 3
end

"""
bτ(N; κQ)
* It solves the difference equation for bτ, where log bond price ln[P_{τ,t}] = - aτ - bτ'Xₜ. In other words, yield R_{τ,t} = aτ/τ + (bτ'/τ)Xₜ.
* Input: κQ is the DNS decay parameter. For N, the DE is solved for one to N maturities.
* Output: Solved factor loadings are saved with dQ by N matrix. Therefore, i'th column vector in the output is a factor loading for maturity i.
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
* Suppose that Rₜ a yield vector where the corresponding maturities ∈ τₙ. Then, for a bond market factor Xₜ, Rₜ = Aₓ + BₓXₜ + errorₜ. 
* Input: Output of function bτ, and observed maturity τₙ.
* Output: Bₓ
"""
function Bₓ(bτ_, τₙ)
    return (bτ_[:, τₙ] ./ τₙ')'
end


"""
T1X(Bₓ_, Wₚ)
* The affine transformation from latent X to obseved PCs is PCsₜ = T0X + T1X*Xₜ.
* Input: Output of function Bₓ and rotation matrix Wₚ is a dQ by N matrix where each row is the first dQ eigenvectors. 
* Output: T1X
"""
function T1X(Bₓ_, Wₚ)
    return Wₚ * Bₓ_
end

"""
aτ(N, bτ_, τₙ, Wₚ; kQ_infty, ΩPP)
aτ(N, bτ_; kQ_infty, ΩXX, annual=1)
* It solves the difference equation for aτ, where log bond price ln[P_{τ,t}] = - aτ - bτ'Xₜ. In other words, yield R_{τ,t} = aτ/τ + (bτ'/τ)Xₜ.
* Input: The function has two methods(multiple dispatch). 
    - When Wₚ ∈ arg: It calculates aτ using ΩPP. Here, Wₚ is a dQ by N matrix where each row is the first dQ eigenvectors.
    - Otherwise: It calculates aτ using ΩXX, so parameters are in the latent factor space. So, we do not need Wₚ.
    - For N, the DE is solved for one to N maturities. bτ_ is an output of function bτ.
* Output: Vector(Float64)(aτ,N)
"""
function aτ(N, bτ_, τₙ, Wₚ; kQ_infty, ΩPP)

    a = zeros(N)
    T1X_ = T1X(Bₓ(bτ_, τₙ), Wₚ)
    for i in 2:N
        a[i] = a[i-1] - jensens_inequality(i, bτ_, T1X_; ΩPP) + (i - 1) * kQ_infty
    end

    return a
end
function aτ(N, bτ_; kQ_infty, ΩXX, annual=1)

    a = zeros(N)
    for i in 2:N
        J = 0.5 * ΩXX
        J = bτ_[:, i-1]' * J * bτ_[:, i-1]
        if annual == 1
            J /= 1200
        end

        a[i] = a[i-1] - J + (i - 1) * kQ_infty
    end

    return a
end

"""
jensens_inequality(τ, bτ_, T1X_; ΩPP, annual=1)
* This function evaluate the Jensen's Ineqaulity term. 
"""
function jensens_inequality(τ, bτ_, T1X_; ΩPP, annual=1)
    J = 0.5 * ΩPP
    J = (T1X_ \ J) / (T1X_')
    J = bτ_[:, τ-1]' * J * bτ_[:, τ-1]
    if annual == 1
        J /= 1200
    end
    return J
end

"""
Aₓ(aτ_, τₙ)
* Suppose that Rₜ a yield vector where the corresponding maturities ∈ τₙ. Then, for a bond market factor Xₜ, Rₜ = Aₓ + BₓXₜ + errorₜ. 
* Input: Output of function aτ, and observed maturity τₙ.
* Output: Aₓ
"""
function Aₓ(aτ_, τₙ)
    return aτ_[τₙ] ./ τₙ
end

"""
T0P(T1X_, Aₓ_, Wₚ)
* The affine transformation from obseved PCs to the latent Xₜ is Xₜ = T0P + T1P*PCsₜ.
* Input: Wₚ is a dQ by N matrix where each row is the first dQ eigenvectors. 
* Output: T0P
"""
function T0P(T1X_, Aₓ_, Wₚ)
    return -(T1X_ \ Wₚ) * Aₓ_
end

"""
Aₚ(Aₓ_, Bₓ_, T0P_, Wₒ)
* Suppose that Oₜ is principal components that are not PCs. Then, Oₜ = Aₚ + BₚPCsₜ + errorₜ. 
* Input: Wₒ is a (N-dQ) by N matrix where each row is the remaining eigenvectors. 
* Output: Aₚ
"""
function Aₚ(Aₓ_, Bₓ_, T0P_, Wₒ)
    return Wₒ * (Aₓ_ + Bₓ_ * T0P_)
end

"""
Bₚ(Bₓ_, T1X_, Wₒ)
* Suppose that Oₜ is principal components that are not PCs. Then, Oₜ = Aₚ + BₚPCsₜ + errorₜ. 
* Input: Wₒ is a (N-dQ) by N matrix where each row is the remaining eigenvectors. 
* Output: Bₚ
"""
function Bₚ(Bₓ_, T1X_, Wₒ)
    return (Wₒ * Bₓ_) / T1X_
end

"""
_termPremium(τ, PCs, macros, bτ_, T1X_; κQ, kQ_infty, KₚP, GₚFF, ΩPP)
* This function calculates a term premium for maturity τ. 
* Input: τ is a target maturity. bτ and T1X is calculated from the corresponding functions. PCs and macros are the data. GₚFF is dP by p*dP matrix that contains slope coefficients of the reduced form transition equation. 
    - Remember that data should contains initial observations, that is t = 0,1,...,p-1. 
* Output(4): TP, timevarying_TP, const_TP, jensen
    - TP: term premium of maturity τ
    - timevarying_TP: contributions of each dependent variable on TP at each time t (row: time, col: variable)
    - const_TP: Constant part of TP
    - jensen: Jensen's Ineqaulity part in TP
    - Although input has initial observations, output excludes the time period for the initial observations.  
"""
function _termPremium(τ, PCs, macros, bτ_, T0P_, T1X_; κQ, kQ_infty, KₚP, GₚFF, ΩPP)

    T1P_ = inv(T1X_)
    # Jensen's Ineqaulity term
    jensen = 0
    for i = 1:(τ-1)
        jensen += jensens_inequality(i + 1, bτ_, T1X_; ΩPP)
    end
    jensen /= -τ

    # Constant term
    dQ = dimQ()
    KₓQ = zeros(dQ)
    KₓQ[1] = kQ_infty
    KₚQ = T1X_ * (KₓQ + (GQ_XX(; κQ) - I(dQ)) * T0P_)
    λₚ = KₚP - KₚQ
    const_TP = 0
    for i = 1:(τ-1)
        const_TP += bτ_[:, i]' * (T1P_ * λₚ)
    end
    const_TP /= -τ

    # Time-varying part
    dP = size(GₚFF, 1)
    p = Int(size(GₚFF, 2) / dP)
    T = size(PCs, 1) # time series length including intial conditions
    timevarying_TP = zeros(T - p, dP) # time-varying part is seperated to see the individual contribution of each priced factor. So, the column length is dP.

    GQ_PP = T1X_ * GQ_XX(; κQ) * T1P_
    Λ_PF = GₚFF[1:dQ, :]
    Λ_PF[1:dQ, 1:dQ] -= GQ_PP
    T1P_Λ_PF = T1P_ * Λ_PF

    datas = [PCs macros]
    for t in (p+1):T # ranges for the dependent variables. The whole range includes initial observations.
        # prediction part
        predicted_X = datas[t:-1:1, :]
        for horizon = 1:(τ-2)
            regressors = vec(predicted_X[1:p, :]')
            predicted = GₚFF * regressors
            predicted_X = vcat(predicted', predicted_X)
        end
        reverse!(predicted_X, dims=1)

        # Calculate TP
        for i = 1:(τ-1), l = 1:p
            weight = bτ_[:, τ-i]' * (T1P_Λ_PF)
            for j = 1:dP
                timevarying_TP[t-p, j] += weight[(l-1)*dP+j] * predicted_X[t+i-l, j] # first row in predicted_X = (time = t-p+1)
            end
        end
    end
    timevarying_TP /= -τ

    TP = sum(timevarying_TP, dims=2) .+ jensen .+ const_TP

    return TP, timevarying_TP, const_TP, jensen
end

"""
term_premium(τ, τₙ, saved_θ, yields, macros)
* This function generates posterior samples of the term premiums.
* Input: targeted maturity τ, all observed maturities τₙ = [1;3;6;...], the Gibbs sampling samples "saved_θ", and the data that contains initial observations.
* Output: Vector{Dict}(posterior samples, length(saved_θ)). 
    - "TP", "timevarying\\_TP", "const\\_TP", "jensen" ∈ Output[i]
    - Outputs exclude initial observations.
"""
function term_premium(τ, τₙ, saved_θ, yields, macros)

    iteration = length(saved_θ)
    saved_TP = Vector{TermPremium}(undef, iteration)

    dQ = dimQ()
    dP = size(saved_θ[:ϕ][1], 1)
    p = Int((size(saved_θ[:ϕ][1], 2) - 1) / dP - 1)
    PCs, ~, Wₚ = PCA(yields, p)

    @showprogress 1 "Calculating TPs..." for iter in 1:iteration

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

        aτ_ = aτ(τₙ[end], bτ_, τₙ, Wₚ; kQ_infty, ΩPP=ΩFF[1:dQ, 1:dQ])
        Aₓ_ = Aₓ(aτ_, τₙ)
        T0P_ = T0P(T1X_, Aₓ_, Wₚ)
        TP, timevarying_TP, const_TP, jensen = _termPremium(τ, PCs, macros, bτ_, T0P_, T1X_; κQ, kQ_infty, KₚP=KₚF[1:dQ], GₚFF, ΩPP=ΩFF[1:dQ, 1:dQ])

        saved_TP[iter] = TermPremium(TP=TP[:, 1], timevarying_TP=timevarying_TP, const_TP=const_TP, jensen=jensen)
    end

    return saved_TP

end

"""
latentspace(saved_θ, yields, τₙ)
* This function translates the principal components state space into the latent factor state space. 
* Input: the Gibb sampling result "saved_θ", and the data should include initial observations.
* Output: Vector{Dict}(posterior samples, length(saved_θ)). 
    - "latents", "κQ", "kQ_infty", "KₚXF", "GₚXFXF", "ΩXFXF", "ηψ", "ψ", "ψ0", "Σₒ", "γ" ∈ Output[i]
    - latent factors contain initial observations.
"""
function latentspace(saved_θ, yields, τₙ)

    iteration = length(saved_θ)
    saved_θ_latent = Vector{LatentSpace}(undef, iteration)
    @showprogress 1 "Moving to the latent space..." for iter in 1:iteration

        κQ = saved_θ[:κQ][iter]
        kQ_infty = saved_θ[:kQ_infty][iter]
        ϕ = saved_θ[:ϕ][iter]
        σ²FF = saved_θ[:σ²FF][iter]

        ϕ0, C = ϕ_2_ϕ₀_C(; ϕ)
        ϕ0 = C \ ϕ0
        KₚF = ϕ0[:, 1]
        GₚFF = ϕ0[:, 2:end]
        ΩFF = (C \ diagm(σ²FF)) / C'

        latent, κQ, kQ_infty, KₚXF, GₚXFXF, ΩXFXF = PCs_2_latents(yields, τₙ; κQ, kQ_infty, KₚF, GₚFF, ΩFF)
        saved_θ_latent[iter] = LatentSpace(latents=latent, κQ=κQ, kQ_infty=kQ_infty, KₚXF=KₚXF, GₚXFXF=GₚXFXF, ΩXFXF=ΩXFXF)

    end

    return saved_θ_latent
end

"""
PCs\\_2\\_latents(yields, τₙ; κQ, kQ_infty, KₚF, GₚFF, ΩFF)
* XF are in the latent factor space and F are in the PC state space.
* Input: yields should include initial observations
* Output(6): latent, κQ, kQ_infty, KₚXF, GₚXFXF, ΩXFXF
    - latent factors contain initial observations.
"""
function PCs_2_latents(yields, τₙ; κQ, kQ_infty, KₚF, GₚFF, ΩFF)

    dP = size(ΩFF, 1)
    dQ = dimQ()
    dM = dP - dQ # of macro variables
    p = Int(size(GₚFF, 2) / dP)
    PCs, ~, Wₚ = PCA(yields, p)

    # statistical Parameters
    bτ_ = bτ(τₙ[end]; κQ)
    Bₓ_ = Bₓ(bτ_, τₙ)
    T1X_ = T1X(Bₓ_, Wₚ)
    T1P_ = inv(T1X_)

    aτ_ = aτ(τₙ[end], bτ_, τₙ, Wₚ; kQ_infty, ΩPP=ΩFF[1:dQ, 1:dQ])
    Aₓ_ = Aₓ(aτ_, τₙ)
    T0P_ = T0P(T1X_, Aₓ_, Wₚ)

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
fitted_YieldCurve(τ0, saved_Xθ::Vector{LatentSpace})
* It generates a fitted yield curve
* Input: τ0 is a set of maturities of interest. saved_Xθ is a transformed posterior sample using function latentspace.
    - τ0 does not need to be the same as the one used for the estimation.
* Output: Vector{YieldCurve}(,# of iteration)
    - yields and factors contain initial observations.
"""
function fitted_YieldCurve(τ0, saved_Xθ::Vector{LatentSpace})

    dQ = dimQ()
    iteration = length(saved_Xθ)
    YieldCurve_ = Vector{YieldCurve}(undef, iteration)
    @showprogress 1 "Generating fitted yield curve..." for iter in 1:iteration

        latents = saved_Xθ[:latents][iter]
        κQ = saved_Xθ[:κQ][iter]
        kQ_infty = saved_Xθ[:kQ_infty][iter]
        ΩXFXF = saved_Xθ[:ΩXFXF][iter]

        # statistical Parameters
        bτ_ = bτ(τ0[end]; κQ)
        Bₓ_ = Bₓ(bτ_, τ0)
        aτ_ = aτ(τ0[end], bτ_; kQ_infty, ΩXX=ΩXFXF[1:dQ, 1:dQ])
        Aₓ_ = Aₓ(aτ_, τ0)

        YieldCurve_[iter] = YieldCurve(
            latents=latents,
            yields=(Aₓ_ .+ Bₓ_ * latents')' |> Matrix,
            intercept=Aₓ_,
            slope=Bₓ_
        )
    end

    return YieldCurve_
end

"""
PCA(yields, p; rescaling = false)
* It derives the principal components from yields.
* Input: yields[p+1:end, :] is used to construct the affine transformation, and then all yields[:,:] are transformed into the principal components.
    - If rescaling == true, all PCs and OCs are normalized to have an average std of yields 
* Output(4): PCs, OCs, Wₚ, Wₒ
    - PCs, OCs: first dQ and the remaining principal components
    - Wₚ, Wₒ: the rotation matrix for PCs and OCs, respectively
"""
function PCA(yields, p; rescaling=false)

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

    if rescaling == false
        return Matrix(PCs), Matrix(OCs), Wₚ, Wₒ
    else
        ## rescaling
        # mean_std = mean(std(yields[(p+1):end, :], dims=1))
        mean_std = 1.0
        scale_PCs = mean_std ./ std(PCs, dims=1)'
        scale_OCs = mean_std ./ std(OCs, dims=1)'

        return Matrix((scale_PCs .* PCs')'), Matrix((scale_OCs .* OCs')'), scale_PCs .* Wₚ, scale_OCs .* Wₒ
    end
end
"""
maximum_SR(yields, macros, HyperParameter_::HyperParameter, τₙ, ρ; medium_τ=12 * [1.5, 2, 2.5, 3, 3.5], iteration=1_000)
* It calculate a prior distribution of realized maximum Sharpe ratio. It is unobservable because we do not know true parameters.
* Input: Data should contains initial conditions
* Output: Matrix{Float64}(maximum SR, time length, simulation)
"""
function maximum_SR(yields, macros, HyperParameter_::HyperParameter, τₙ, ρ; medium_τ=12 * [1.5, 2, 2.5, 3, 3.5], iteration=300)

    (; p, q, ν0, Ω0) = HyperParameter_
    PCs, ~, Wₚ = PCA(yields, p)
    factors = [PCs macros]
    dP = length(Ω0)
    dQ = dimQ()

    prior_σ²FF_ = prior_σ²FF(; ν0, Ω0)
    prior_C_ = prior_C(; Ω0)
    prior_κQ_ = prior_κQ(medium_τ)
    prior_ϕ0_ = prior_ϕ0(ρ, prior_κQ_, τₙ, Wₚ; ψ0=ones(dP), ψ=ones(dP, dP * p), q, ν0, Ω0)
    σ²kQ_infty = q[4] * 2scale(prior_σ²FF(; ν0, Ω0)[1]) / (2shape(prior_σ²FF(; ν0, Ω0)[1]) - 2) # prior variance of kQ_infty
    kQ_infty_dist = Normal(0, sqrt(σ²kQ_infty))

    mSR = Vector{Float64}(undef, iteration)
    @showprogress 1 "Calculating maximum SR..." for iter in 1:iteration

        σ²FF = rand.(prior_σ²FF_)
        C = rand.(prior_C_)
        for i in 2:dP, j in 1:(i-1)
            C[i, j] = Normal(0, sqrt(σ²FF[i] * var(prior_C_[i, j]))) |> x -> rand(x)
        end
        ΩFF = (C \ diagm(σ²FF)) / C' |> Symmetric
        ϕ0 = rand.([Normal(mean(prior_ϕ0_[i, j]), sqrt(σ²FF[i] * var(prior_ϕ0_[i, j]))) for i in 1:dP, j in 1:(dP*p+1)])

        ϕ0 = C \ ϕ0
        KₚF = ϕ0[:, 1]
        GₚFF = ϕ0[:, 2:end]

        κQ = rand(prior_κQ_)
        bτ_ = bτ(τₙ[end]; κQ)
        Bₓ_ = Bₓ(bτ_, τₙ)
        T1X_ = T1X(Bₓ_, Wₚ)

        kQ_infty = rand(kQ_infty_dist)
        aτ_ = aτ(τₙ[end], bτ_, τₙ, Wₚ; kQ_infty, ΩPP=ΩFF[1:dQ, 1:dQ])
        Aₓ_ = Aₓ(aτ_, τₙ)
        T0P_ = T0P(T1X_, Aₓ_, Wₚ)

        KₓQ = zeros(dQ)
        KₓQ[1] = kQ_infty
        KₚQ = T1X_ * (KₓQ + (GQ_XX(; κQ) - I(dQ)) * T0P_)
        GQPF = similar(GₚFF[1:dQ, :]) |> (x -> x .= 0)
        GQPF[:, 1:dQ] = T1X_ * GQ_XX(; κQ) / T1X_
        λP = KₚF[1:dQ] - KₚQ
        ΛPF = GₚFF[1:dQ, :] - GQPF

        # # Transition equation: F(t) = μT + G*F(t-1) + N(0,Ω), where F(t): dP*p vector
        # μT = [KₚF
        #     zeros(dP * (p - 1))]
        # G = [GₚFF
        #     I(dP * (p - 1)) zeros(dP * (p - 1), dP)]
        # Ω = [ΩFF zeros(dP, dP * (p - 1))
        #     zeros(dP * (p - 1), dP * p)]
        # mean_Ft = (I(length(μT)) - G) \ μT
        # var_Ft = (I(length(μT)^2) - kron(G, G)) \ vec(Ω) |> x -> reshape(x, length(μT), length(μT)) |> Symmetric
        # Ft = rand(MvNormal(mean_Ft, var_Ft))

        Ft = rand(p+1:size(factors, 1)) |> x -> factors'[:, x:-1:x-p+1] |> vec
        mSR[iter] = cholesky(ΩFF).L \ [λP + ΛPF * Ft; zeros(dP - dQ)] |> x -> sqrt(x'x)
    end

    return mSR
end