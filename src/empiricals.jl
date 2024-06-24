"""
    loglik_mea(yields, tau_n; kappaQ, kQ_infty, phi, varFF, SigmaO, data_scale)
This function generate a log likelihood of the measurement equation.
# Output
- the measurement equation part of the log likelihood
"""
function loglik_mea(yields, tau_n; kappaQ, kQ_infty, phi, varFF, SigmaO, data_scale)

    dP = length(varFF)
    p = Int(((size(phi, 2) - 1) / dP) - 1)
    yields = yields[p+1:end, :] #excludes initial observations
    dQ = dimQ() + size(yields, 2) - length(tau_n)

    PCs, OCs, Wₚ, Wₒ, mean_PCs = PCA(yields, 0)
    bτ_ = bτ(tau_n[end]; kappaQ, dQ)
    Bₓ_ = Bₓ(bτ_, tau_n)
    T1X_ = T1X(Bₓ_, Wₚ)
    Bₚ_ = Bₚ(Bₓ_, T1X_, Wₒ)

    ΩPP = phi_varFF_2_ΩPP(; phi, varFF, dQ)
    aτ_ = aτ(tau_n[end], bτ_, tau_n, Wₚ; kQ_infty, ΩPP, data_scale)
    Aₓ_ = Aₓ(aτ_, tau_n)
    T0P_ = T0P(T1X_, Aₓ_, Wₚ, mean_PCs)
    Aₚ_ = Aₚ(Aₓ_, Bₓ_, T0P_, Wₒ)

    T = size(OCs, 1)
    dist_mea = MvNormal(diagm(SigmaO))
    residuals = (OCs' - (Aₚ_ .+ Bₚ_ * PCs'))'

    logpdf_ = 0
    for t = 1:T
        logpdf_ += logpdf(dist_mea, residuals[t, :])
    end

    return logpdf_
end

"""
    loglik_mea2(yields, tau_n; kappaQ, kQ_infty, phi, varFF, SigmaO, data_scale)
"""
function loglik_mea2(yields, tau_n, p; kappaQ, kQ_infty, ΩPP, SigmaO, data_scale)

    yields = yields[p+1:end, :] #excludes initial observations
    dQ = dimQ() + size(yields, 2) - length(tau_n)

    PCs, OCs, Wₚ, Wₒ, mean_PCs = PCA(yields, 0)
    bτ_ = bτ(tau_n[end]; kappaQ, dQ)
    Bₓ_ = Bₓ(bτ_, tau_n)
    T1X_ = T1X(Bₓ_, Wₚ)
    Bₚ_ = Bₚ(Bₓ_, T1X_, Wₒ)

    aτ_ = aτ(tau_n[end], bτ_, tau_n, Wₚ; kQ_infty, ΩPP, data_scale)
    Aₓ_ = Aₓ(aτ_, tau_n)
    T0P_ = T0P(T1X_, Aₓ_, Wₚ, mean_PCs)
    Aₚ_ = Aₚ(Aₓ_, Bₓ_, T0P_, Wₒ)

    T = size(OCs, 1)
    dist_mea = MvNormal(diagm(SigmaO))
    residuals = (OCs' - (Aₚ_ .+ Bₚ_ * PCs'))'

    logpdf_ = 0
    for t = 1:T
        logpdf_ += logpdf(dist_mea, residuals[t, :])
    end

    return logpdf_
end

"""
    loglik_tran(PCs, macros; phi, varFF)
It calculate log likelihood of the transition equation. 
# Output 
- log likelihood of the transition equation.
"""
function loglik_tran(PCs, macros; phi, varFF)

    dP = length(varFF)
    p = Int(((size(phi, 2) - 1) / dP) - 1)

    yphi, Xphi = yphi_Xphi(PCs, macros, p)

    T = size(yphi, 1)
    logpdf_ = Vector{Float64}(undef, dP)
    for i in 1:dP
        logpdf_[i] = logpdf(MvNormal(Xphi * (phi[i, :]), varFF[i] * I(T)), yphi[:, i])
    end

    return sum(logpdf_)
end

"""
    yphi_Xphi(PCs, macros, p)
This function generate the dependent variable and the corresponding regressors in the orthogonalized transition equation.
# Output(4)
`yphi`, `Xphi = [ones(T - p) Xphi_lag Xphi_contemporaneous]`, `[ones(T - p) Xphi_lag]`, `Xphi_contemporaneous`
- `yphi` and `Xphi` is a full matrix. For i'th equation, the dependent variable is `yphi[:,i]` and the regressor is `Xphi`. 
- `Xphi` is same to all orthogonalized transtion equations. The orthogonalized equations are different in terms of contemporaneous regressors. Therefore, the corresponding regressors in `Xphi` should be excluded. The form of parameter `phi` do that task by setting the coefficients of the excluded regressors to zeros. In particular, for last `dP` by `dP` block in `phi`, the diagonals and the upper diagonal elements should be zero. 
"""
function yphi_Xphi(PCs, macros, p)

    if isempty(macros)
        data = copy(PCs)
    else
        data = [PCs macros]
    end
    T = size(data, 1) # length including initial observations
    dP = size(data, 2)

    yphi = data[(p+1):T, :]

    Xphi_lag = Matrix{Float64}(undef, T - p, dP * p)
    Xphi_contem = Matrix{Float64}(undef, T - p, dP)
    for t in (p+1):T
        Xphi_lag[t-p, :] = vec(data[(t-1):-1:(t-p), :]')'
        Xphi_contem[t-p, :] = -data[t, :]
    end
    Xphi = [ones(T - p) Xphi_lag Xphi_contem]

    return yphi, Xphi, [ones(T - p) Xphi_lag], Xphi_contem
end

"""
    LDL(X)
This function generate a matrix decomposition, called LDLt. `X = L*D*L'`, where `L` is a lower triangular matrix and `D` is a diagonal. How to conduct it can be found at [Wikipedia](https://en.wikipedia.org/wiki/Cholesky_decomposition#LDL_decomposition).
# Input
- Decomposed Object, `X`
# Output(2)
`L`, `D`
- Decomposed result is `X = L*D*L'`
"""
function LDL(X)
    C = cholesky(X).L
    S = Diagonal(diagm(diag(C)))
    L = C / S
    D = S^2

    return L, D
end

"""
    phi_varFF_2_ΩPP(; phi, varFF, dQ=[])
It construct `ΩPP` from statistical parameters.
# Output
- `ΩPP`
"""
function phi_varFF_2_ΩPP(; phi, varFF, dQ=[])

    if isempty(dQ)
        dQ = dimQ()
    end
    ~, C = phi_2_phi₀_C(; phi)

    CQQ = C[1:dQ, 1:dQ]
    return (CQQ \ diagm(varFF[1:dQ])) / CQQ'

end

"""
    phi_2_phi₀_C(; phi)
It divide phi into the lagged regressor part and the contemporaneous regerssor part.
# Output(3)
`phi0`, `C = C0 + I`, `C0`
- `phi0`: coefficients for the lagged regressors
- `C`: coefficients for the dependent variables when all contemporaneous variables are in the LHS of the orthogonalized equations. Therefore, the diagonals of `C` is ones. Note that since the contemporaneous variables get negative signs when they are at the RHS, the signs of `C` do not change whether they are at the RHS or LHS. 
"""
function phi_2_phi₀_C(; phi)

    dP = size(phi, 1)
    phi0 = phi[:, 1:(end-dP)]
    C0 = phi[:, (end-dP+1):end]
    C = C0 + I(dP)

    return phi0, C, C0
end

"""
    phi_varFF_2_OmegaFF(; phi, varFF)
It construct `OmegaFF` from statistical parameters.
# Output
- `OmegaFF`
"""
function phi_varFF_2_OmegaFF(; phi, varFF)

    C = phi_2_phi₀_C(; phi)[2]
    return (C \ diagm(varFF)) / C'
end

"""
    isstationary(GPFF)
It checks whether a reduced VAR matrix has unit roots. If there is at least one unit root, return is false.
# Input
- `GPFF` should not include intercepts. Also, `GPFF` is `dP` by `dP*p` matrix that the coefficient at lag 1 comes first, and the lag `p` slope matrix comes last. 
# Output
- `boolean`
"""
function isstationary(GPFF)
    dP = size(GPFF, 1)
    p = Int(size(GPFF, 2) / dP)

    G = [GPFF
        I(dP * (p - 1)) zeros(dP * (p - 1), dP)]

    return maximum(abs.(eigen(G).values)) < 1 || maximum(abs.(eigen(G).values)) ≈ 1
end

""" 
    erase_nonstationary_param(saved_params)
It filters out posterior samples that implies an unit root VAR system. Only stationary posterior samples remain.
# Input
- `saved_params` is the first output of function `posterior_sampler`.
# Output(2): 
stationary samples, acceptance rate(%)
- The second output indicates how many posterior samples remain.
"""
function erase_nonstationary_param(saved_params)

    iteration = length(saved_params)
    stationary_saved_params = Vector{Parameter}(undef, 0)
    prog = Progress(iteration; dt=5, desc="erase_nonstationary_param...")
    #Threads.@threads 
    for iter in 1:iteration

        kappaQ = saved_params[:kappaQ][iter]
        kQ_infty = saved_params[:kQ_infty][iter]
        phi = saved_params[:phi][iter]
        varFF = saved_params[:varFF][iter]
        SigmaO = saved_params[:SigmaO][iter]
        gamma = saved_params[:gamma][iter]

        phi0, C = phi_2_phi₀_C(; phi)
        phi0 = C \ phi0
        GPFF = phi0[:, 2:end]

        if isstationary(GPFF)
            push!(stationary_saved_params, Parameter(kappaQ=copy(kappaQ), kQ_infty=copy(kQ_infty), phi=copy(phi), varFF=copy(varFF), SigmaO=copy(SigmaO), gamma=copy(gamma)))
        end
        next!(prog)
    end
    finish!(prog)

    return stationary_saved_params, 100length(stationary_saved_params) / iteration
end

"""
    reducedform(saved_params, yields, macros, tau_n; data_scale=1200)
It converts posterior samples in terms of the reduced form VAR parameters.
# Input
- `saved_params` is the first output of function `posterior_sampler`.
# Output
- Posterior samples in terms of struct `ReducedForm`
"""
function reducedform(saved_params, yields, macros, tau_n; data_scale=1200)

    dQ = dimQ() + size(yields, 2) - length(tau_n)
    dP = size(saved_params[:phi][1], 1)
    p = Int((size(saved_params[:phi][1], 2) - 1) / dP - 1)
    PCs, ~, Wₚ, ~, mean_PCs = PCA(yields, p)
    if isempty(macros)
        factors = copy(PCs)
    else
        factors = [PCs macros]
    end

    iteration = length(saved_params)
    reduced_params = Vector{ReducedForm}(undef, iteration)
    prog = Progress(iteration; dt=5, desc="reducedform...")
    Threads.@threads for iter in 1:iteration

        kappaQ = saved_params[:kappaQ][iter]
        kQ_infty = saved_params[:kQ_infty][iter]
        phi = saved_params[:phi][iter]
        varFF = saved_params[:varFF][iter]
        SigmaO = saved_params[:SigmaO][iter]

        phi0, C = phi_2_phi₀_C(; phi)
        phi0 = C \ phi0
        KPF = phi0[:, 1]
        GPFF = phi0[:, 2:end]
        OmegaFF = (C \ diagm(varFF)) / C' |> Symmetric

        bτ_ = bτ(tau_n[end]; kappaQ, dQ)
        Bₓ_ = Bₓ(bτ_, tau_n)
        T1X_ = T1X(Bₓ_, Wₚ)
        aτ_ = aτ(tau_n[end], bτ_, tau_n, Wₚ; kQ_infty, ΩPP=OmegaFF[1:dQ, 1:dQ], data_scale)
        Aₓ_ = Aₓ(aτ_, tau_n)
        T0P_ = T0P(T1X_, Aₓ_, Wₚ, mean_PCs)

        KQ_X = zeros(dQ)
        KQ_X[1] = copy(kQ_infty)
        KQ_P = T1X_ * (KQ_X + (GQ_XX(; kappaQ) - I(dQ)) * T0P_)
        GQPF = similar(GPFF[1:dQ, :]) |> (x -> x .= 0)
        GQPF[:, 1:dQ] = T1X_ * GQ_XX(; kappaQ) / T1X_
        lambdaP = KPF[1:dQ] - KQ_P
        LambdaPF = GPFF[1:dQ, :] - GQPF

        mpr = Matrix{Float64}(undef, size(factors, 1) - p, dP)
        for t in p+1:size(factors, 1)
            Ft = factors'[:, t:-1:t-p+1] |> vec
            mpr[t-p, :] = cholesky(OmegaFF).L \ [lambdaP + LambdaPF * Ft; zeros(dP - dQ)]
        end
        reduced_params[iter] = ReducedForm(kappaQ=copy(kappaQ), kQ_infty=copy(kQ_infty), KPF=copy(KPF), GPFF=copy(GPFF), OmegaFF=copy(OmegaFF), SigmaO=copy(SigmaO), lambdaP=copy(lambdaP), LambdaPF=copy(LambdaPF), mpr=copy(mpr))

        next!(prog)
    end
    finish!(prog)

    return reduced_params
end

"""
    calibrate_mean_phi_const(mean_kQ_infty, std_kQ_infty, nu0, yields, macros, tau_n, p; mean_phi_const_PCs=[], medium_tau=collect(24:3:48), iteration=1000, data_scale=1200, kappaQ_prior_pr=[], τ=[])
The purpose of the function is to calibrate a prior mean of the first `dQ` constant terms in our VAR. Adjust your prior setting based on the prior samples in outputs.
# Input 
- `mean_phi_const_PCs` is your prior mean of the first `dQ` constants. Our default option set it as a zero vector.
- `iteration` is the number of prior samples.
- `τ::scalar` is a maturity for calculating the constant part in the term premium.
    - If τ is empty, the function does not sampling the prior distribution of the constant part in the term premium.
# Output(2)
`prior_λₚ`, `prior_TP`
- samples from the prior distribution of `λₚ` 
- prior samples of constant part in the τ-month term premium
"""
function calibrate_mean_phi_const(mean_kQ_infty, std_kQ_infty, nu0, yields, macros, tau_n, p; mean_phi_const_PCs=[], medium_tau=collect(24:3:48), iteration=1000, data_scale=1200, kappaQ_prior_pr=[], τ=[])

    dQ = dimQ() + size(yields, 2) - length(tau_n)
    PCs, ~, Wₚ, ~, mean_PCs = PCA(yields, p)

    if isempty(macros)
        factors = copy(PCs)
    else
        factors = [PCs macros]
    end
    dP = size(factors, 2)
    OmegaFF_mean = Vector{Float64}(undef, dP)
    for i in eachindex(OmegaFF_mean)
        OmegaFF_mean[i] = AR_res_var(factors[:, i], p)[1]
    end

    if isempty(mean_phi_const_PCs)
        mean_phi_const_PCs = zeros(dQ)
    end
    if isempty(kappaQ_prior_pr)
        kappaQ_prior_pr = length(medium_tau) |> x -> ones(x) / x
    end

    prior_TP = Vector{Float64}(undef, iteration)
    prior_λₚ = Matrix{Float64}(undef, iteration, dQ)
    for iter in 1:iteration
        if isempty(nu0)
            ΩPP = diagm(OmegaFF_mean[1:dQ])
        else
            ΩPP = (nu0 - dP - 1) * diagm(OmegaFF_mean) |> x -> InverseWishart(nu0, x) |> rand |> x -> x[1:dQ, 1:dQ]
        end
        kappaQ = prior_kappaQ(medium_tau, kappaQ_prior_pr) |> rand
        kQ_infty = Normal(mean_kQ_infty, std_kQ_infty) |> rand

        bτ_ = bτ(tau_n[end]; kappaQ, dQ)
        Bₓ_ = Bₓ(bτ_, tau_n)
        T1X_ = T1X(Bₓ_, Wₚ)

        aτ_ = aτ(tau_n[end], bτ_, tau_n, Wₚ; kQ_infty, ΩPP, data_scale)
        Aₓ_ = Aₓ(aτ_, tau_n)
        T0P_ = T0P(T1X_, Aₓ_, Wₚ, mean_PCs)

        # Constant term
        KQ_X = zeros(dQ)
        KQ_X[1] = copy(kQ_infty)
        KQ_P = T1X_ * (KQ_X + (GQ_XX(; kappaQ) - I(dQ)) * T0P_)
        λₚ = mean_phi_const_PCs - KQ_P
        prior_λₚ[iter, :] = copy(λₚ)

        if !isempty(τ)
            # Jensen's Ineqaulity term
            jensen = 0
            for i = 1:(τ-1)
                jensen += jensens_inequality(i + 1, bτ_, T1X_; ΩPP, data_scale)
            end
            jensen /= -τ

            prior_TP[iter] = sum(bτ_[:, 1:(τ-1)], dims=2)' * (T1X_ \ λₚ) |> x -> (-x[1] / τ) + jensen
        end
    end

    return prior_λₚ, prior_TP

end
