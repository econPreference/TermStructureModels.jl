"""
scenario\\_sampler(S, horizon, saved\\_θ, yields, macros, τₙ)
* Input: scenarios, a result of the posterior sampler, and data 
    - Data includes initial conditions
    - S = Vector{Matrix}(scenario[t], period length of the scenario) 
    - S[t] = Matrix{Float64}([S s][row,col], # of scenarios, N + dP - dQ), where S is a linear combination coefficient matrix and s is a vector of conditional values.
    - If we need an unconditional prediction, S = [].
    - horizon: maximum length of the predicted path
* Output(2): spanned\\_yield, spanned\\_F
    - "predicted\\_yields", "predicted\\_factors" ∈ Output
    - Matrix{Float64}(scenario,horizon,dP or N)
    - function "load\\_object" can be applied
"""
function scenario_sampler(S, horizon, saved_θ, yields, macros, τₙ)
    iteration = length(saved_θ)
    scenarios = []
    @showprogress 1 "Predicting scenarios..." for iter in 1:iteration

        κQ = saved_θ[iter]["κQ"]
        kQ_infty = saved_θ[iter]["kQ_infty"]
        ϕ = saved_θ[iter]["ϕ"]
        σ²FF = saved_θ[iter]["σ²FF"]
        Σₒ = saved_θ[iter]["Σₒ"]

        spanned_yield, spanned_F = _scenario_sampler(S, horizon, yields, macros, τₙ; κQ, kQ_infty, ϕ, σ²FF, Σₒ)

        push!(scenarios,
            Dict(
                "predicted_yields" => spanned_yield,
                "predicted_factors" => spanned_F
            ))
    end

    return scenarios
end


"""
_scenario_sampler(S, horizon, yields, macros, τₙ; κQ, kQ_infty, ϕ, σ²FF, Σₒ)
* Input: Data includes initial conditions
* Output(2): spanned_yield, spanned_F
    - Matrix{Float64}(scenario,horizon,dP or N)
"""
function _scenario_sampler(S, horizon, yields, macros, τₙ; κQ, kQ_infty, ϕ, σ²FF, Σₒ)

    ## Construct GDTSM parameters
    ϕ0, C = ϕ_2_ϕ₀_C(; ϕ)
    ϕ0 = C \ ϕ0 # reduced form parameters
    KₚF = ϕ0[:, 1]
    GₚFF = ϕ0[:, 2:end]
    ΩFF = (C \ diagm(σ²FF)) / C' |> Symmetric

    N = length(τₙ)
    dQ = dimQ()
    dP = size(ΩFF, 1)
    k = size(GₚFF, 2) # of factors in the companion from
    p = Int(k / dP)
    dh = length(S) # a time series length of the scenario, dh = 0 for an unconditional prediction
    PCs, ~, Wₚ, Wₒ = PCA(yields, p)
    data = [PCs macros] # no initial conditions
    T = size(data, 1)

    bτ_ = bτ(τₙ[end]; κQ)
    Bₓ_ = Bₓ(bτ_, τₙ)
    aτ_ = aτ(τₙ[end], bτ_, τₙ, Wₚ; kQ_infty, ΩPP=ΩFF[1:dQ, 1:dQ])
    Aₓ_ = Aₓ(aτ_, τₙ)
    T1X_ = T1X(Bₓ_, Wₚ)
    T1P_ = inv(T1X_)
    T0P_ = T0P(T1X_, Aₓ_, Wₚ)
    Σᵣ = [Wₚ; Wₒ] \ [zeros(dQ, N); zeros(N - dQ, dQ) diagm(Σₒ)] / [Wₚ' Wₒ'] |> Symmetric

    if dh > 0
        ## Construct the Kalman filter parameters
        # Transition equation: F(t) = μT + G*F(t-1) + N(0,Ω), where F(t): dP*p vector
        μT = [KₚF
            zeros(dP * (p - 1))]
        G = [GₚFF
            I(dP * (p - 1)) zeros(dP * (p - 1), dP)]
        Ω = [ΩFF zeros(dP, dP * (p - 1))
            zeros(dP * (p - 1), dP * p)]
        # Measurement equation: Y(t) = μM + H*F(t) + N(0,Σ), where Y(t): N + (dP-dQ) vector
        μM = [Aₓ_ + Bₓ_ * T0P_
            zeros(dP - dQ, 1)]
        H = [Bₓ_*T1P_ zeros(N, dP - dQ + dP * (p - 1))
            zeros(dP - dQ, dQ) I(dP - dQ) zeros(dP - dQ, dP * (p - 1))]

        Σ = [Σᵣ zeros(N, dP - dQ)
            zeros(dP - dQ, N + dP - dQ)]

        ## Kalman filtering step
        f_ttm = zeros(k, 1, dh)
        P_ttm = zeros(k, k, dh)
        f_ll = data[end:-1:(end-p+1), :] |> (X -> vec(X'))
        P_ll = zeros(k, k)
        for t = 1:dh

            f_tl = μT + G * f_ll
            P_tl = G * P_ll * G' + Ω

            St = S[t][:, 1:(end-1)]
            st = S[t][:, end]
            # st = St*μM + St*H*F(t) + N(0,St*Σ*St')
            var_tl = (St * H) * P_tl * (St * H)' + St * Σ * St' |> Symmetric
            e_tl = st - St * μM - St * H * f_tl
            Kalgain = P_tl * (St * H)' / var_tl
            f_tt = f_tl + Kalgain * e_tl
            P_tt = P_tl - Kalgain * St * H * P_tl

            f_ttm[:, :, t] = f_tt
            P_ttm[:, :, t] = P_tt

            f_ll = deepcopy(f_tt)
            P_ll = deepcopy(P_tt)

        end

        ## Backward recursion
        predicted_F = zeros(dh, dP)  # T by k
        predicted_yield = zeros(dh, N)

        # beta(T|T) sampling
        P_tt = P_ttm[:, :, dh] |> Symmetric # k by k
        f_tt = f_ttm[:, 1, dh] # k by 1

        ft = deepcopy(f_tt)
        idx = diag(P_tt) .> eps()
        ft[idx] = rand(MvNormal(f_tt[idx], P_tt[idx, idx]))
        predicted_F[dh, :] = ft[1:dP]

        mea_error = [Wₚ; Wₒ] \ [zeros(dQ); rand(MvNormal(zeros(N - dQ), Matrix(diagm(Σₒ))))]
        predicted_yield[dh, :] = (Aₓ_ + Bₓ_ * T0P_) + Bₓ_ * T1P_ * ft[1:dQ] + mea_error

        for t in (dh-1):-1:1

            f_tt = f_ttm[:, 1, t]
            P_tt = P_ttm[:, :, t]

            GPG_Q = G[1:dP, :] * P_tt * G[1:dP, :]' + Ω[1:dP, 1:dP] |> Symmetric # P[t+1|t], k by k

            e_tl = predicted_F[t+1, :] - μT[1:dP] - G[1:dP, :] * f_tt

            PGG = P_tt * G[1:dP, :]' / GPG_Q
            f_tt1 = f_tt + PGG * e_tl

            PGP = PGG * G[1:dP, :] * P_tt
            P_tt1 = P_tt - PGP |> Symmetric

            # beta(t|t+1) sampling
            ft = deepcopy(f_tt1)
            idx = diag(P_tt1) .> eps()
            ft[idx] = rand(MvNormal(f_tt1[idx], P_tt1[idx, idx]))

            predicted_F[t, :] = ft[1:dP]
            mea_error = [Wₚ; Wₒ] \ [zeros(dQ); rand(MvNormal(zeros(N - dQ), Matrix(diagm(Σₒ))))]
            predicted_yield[t, :] = (Aₓ_ + Bₓ_ * T0P_) + Bₓ_ * T1P_ * ft[1:dQ] + mea_error
        end
    end

    spanned_F = Matrix{Float64}(undef, T + horizon, dP)
    spanned_yield = Matrix{Float64}(undef, T + horizon, N)
    spanned_F[1:T, :] = data
    spanned_yield[1:T, :] = yields
    if dh > 0
        spanned_F[(T+1):(T+dh), :] = predicted_F
        spanned_yield[(T+1):(T+dh), :] = predicted_yield
    end
    for t in (T+dh+1):(T+horizon) # predicted period
        X = spanned_F[t-1:-1:t-p, :] |> (X -> vec(X'))
        spanned_F[t, :] = KₚF + GₚFF * X + rand(MvNormal(zeros(dP), ΩFF))
        mea_error = [Wₚ; Wₒ] \ [zeros(dQ); rand(MvNormal(zeros(N - dQ), Matrix(diagm(Σₒ))))]
        spanned_yield[t, :] = (Aₓ_ + Bₓ_ * T0P_) + Bₓ_ * T1P_ * spanned_F[t, 1:dQ] + mea_error
    end

    return spanned_yield[(end-horizon+1):end, :], spanned_F[(end-horizon+1):end, :]
end