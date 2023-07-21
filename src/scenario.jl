"""
scenario\\_sampler(S::Scenario, τ, horizon, saved\\_θ, yields, macros, τₙ)
* Input: scenarios, a result of the posterior sampler, and data 
    - Data includes initial observations
    - S = Vector{Matrix}(scenario[t], period length of the scenario) 
    - S[t] = Matrix{Float64}([S s][row,col], # of scenarios, N + dP - dQ), where S is a linear combination coefficient matrix and s is a vector of conditional values.
    - If we need an unconditional prediction, S = [].
    - τ is a maturity that a term premium of interest has.
    - horizon: maximum length of the predicted path
* Output: Vector{Dict}(scenario, iteration)
    - "predicted\\_yields", "predicted\\_factors", "predicted_TP" ∈ Output
    - element = Matrix{Float64}(scenario,horizon,dP or N or 1)
    - function "load\\_object" can be applied
"""
function scenario_sampler(S, τ, horizon, saved_θ, yields, macros, τₙ; mean_macros=[])
    iteration = length(saved_θ)
    scenarios = Vector{Forecast}(undef, iteration)
    prog = Progress(iteration; dt=5, desc="Predicting scenarios...")
    Threads.@threads for iter in 1:iteration

        κQ = saved_θ[:κQ][iter]
        kQ_infty = saved_θ[:kQ_infty][iter]
        ϕ = saved_θ[:ϕ][iter]
        σ²FF = saved_θ[:σ²FF][iter]
        Σₒ = saved_θ[:Σₒ][iter]

        spanned_yield, spanned_F, predicted_TP = _scenario_sampler(S, τ, horizon, yields, macros, τₙ; κQ, kQ_infty, ϕ, σ²FF, Σₒ, mean_macros)

        scenarios[iter] = Forecast(yields=spanned_yield, factors=spanned_F, TP=predicted_TP)

        next!(prog)
    end
    finish!(prog)

    return scenarios
end


"""
_scenario_sampler(S::Scenario, τ, horizon, yields, macros, τₙ; κQ, kQ_infty, ϕ, σ²FF, Σₒ)
* Input: Data includes initial observations, τ is a maturity that a term premium of interest has.
* Output(3): spanned_yield, spanned_F, predicted_TP
    - Matrix{Float64}(scenario,horizon,dP or N or 1)
"""
function _scenario_sampler(S, τ, horizon, yields, macros, τₙ; κQ, kQ_infty, ϕ, σ²FF, Σₒ, mean_macros)

    ## Construct GDTSM parameters
    ϕ0, C = ϕ_2_ϕ₀_C(; ϕ)
    ϕ0 = C \ ϕ0 # reduced form parameters
    KₚF = ϕ0[:, 1]
    GₚFF = ϕ0[:, 2:end]
    ΩFF = (C \ diagm(σ²FF)) / C' |> Symmetric

    N = length(τₙ)
    dQ = dimQ()
    dP = size(ΩFF, 1)
    k = size(GₚFF, 2) + N # of factors in the companion from
    p = Int((k - N) / dP)
    if S != []
        dh = length(S) # a time series length of the scenario, dh = 0 for an unconditional prediction
    else
        dh = 0
    end
    PCs, ~, Wₚ, Wₒ, mean_PCs = PCA(yields, p)

    if isempty(mean_macros)
        mean_macros = zeros(dP - dQ)
    end

    data = [PCs macros]
    T = size(data, 1)

    bτ_ = bτ(τₙ[end]; κQ)
    Bₓ_ = Bₓ(bτ_, τₙ)
    aτ_ = aτ(τₙ[end], bτ_, τₙ, Wₚ; kQ_infty, ΩPP=ΩFF[1:dQ, 1:dQ])
    Aₓ_ = Aₓ(aτ_, τₙ)
    T1X_ = T1X(Bₓ_, Wₚ)
    T1P_ = inv(T1X_)
    T0P_ = T0P(T1X_, Aₓ_, Wₚ, mean_PCs)
    Σᵣ = [Wₚ; Wₒ] \ [diagm(minimum(Σₒ) .* rand(dQ) .+ eps()) zeros(dQ, N - dQ); zeros(N - dQ, dQ) diagm(Σₒ)] / [Wₚ' Wₒ'] |> Symmetric

    if dh > 0
        ## Construct the Kalman filter parameters
        # Transition equation: F(t) = μT + G*F(t-1) + N(0,Ω), where F(t): dP*p+N vector
        μT = [KₚF
            zeros(N + dP * (p - 1))]
        G = [GₚFF[:, 1:dP] zeros(dP, N) GₚFF[:, dP+1:end]
            zeros(N, dP * p + N)
            I(dP) zeros(dP, dP * p - dP + N)
            zeros(dP * p - 2dP, dP + N) I(dP * p - 2dP) zeros(dP * p - 2dP, dP)]
        Ω = zeros(dP * p + N, dP * p + N)
        Ω[1:dP, 1:dP] = ΩFF
        Ω[dP+1:dP+N, dP+1:dP+N] = Σᵣ
        # Measurement equation: Y(t) = μM + H*F(t), where Y(t): N + (dP-dQ) vector
        μM = [Aₓ_ + Bₓ_ * T0P_
            zeros(dP - dQ, 1)]
        H = [Bₓ_*T1P_ zeros(N, dP - dQ) I(N) zeros(N, dP * p - dP)
            zeros(dP - dQ, dQ) I(dP - dQ) zeros(dP - dQ, dP * p - dP + N)]

        ## Kalman filtering step
        f_ttm = zeros(k, 1, dh)
        P_ttm = zeros(k, k, dh)
        mea_error = yields[end, :] - (Aₓ_ + Bₓ_ * T0P_) - Bₓ_ * T1P_ * data[end, 1:dQ]
        f_ll = data[end:-1:(end-p+1), :] |> (X -> vec(X')) |> x -> [x[1:dP]; mea_error; x[dP+1:end]]
        P_ll = zeros(k, k)
        for t = 1:dh

            f_tl = μT + G * f_ll
            P_tl = G * P_ll * G' + Ω

            St = S[t].combinations
            st = S[t].values
            # st = St*μM + St*H*F(t)
            if maximum(abs.(St)) > 0
                var_tl = (St * H) * P_tl * (St * H)' |> Symmetric
                e_tl = st - St * μM - St * H * f_tl
                Kalgain = P_tl * (St * H)' / var_tl
                f_tt = f_tl + Kalgain * e_tl
                P_tt = P_tl - Kalgain * St * H * P_tl
            else
                f_tt = deepcopy(f_tl)
                P_tt = deepcopy(P_tl)
            end
            idx = diag(P_tt) .< eps()
            P_tt[idx, :] .= 0
            P_tt[:, idx] .= 0

            f_ttm[:, :, t] = f_tt
            P_ttm[:, :, t] = P_tt

            f_ll = deepcopy(f_tt)
            P_ll = deepcopy(P_tt)

        end

        ## Backward recursion
        predicted_F = zeros(dh, dP + N)  # T by k
        predicted_yield = zeros(dh, N)

        # beta(T|T) sampling
        P_tt = P_ttm[1:dP+N, 1:dP+N, dh] |> Symmetric |> Array # k by k
        f_tt = f_ttm[1:dP+N, 1, dh] # k by 1

        ft = deepcopy(f_tt)
        idx = diag(P_tt) .> eps()
        while !isposdef(P_tt[idx, idx])
            P_tt[idx, idx] += eps() * I(sum(idx))
        end
        ft[idx] = MvNormal(f_tt[idx], P_tt[idx, idx]) |> rand
        predicted_F[dh, :] = ft
        predicted_yield[dh, :] = (Aₓ_ + Bₓ_ * T0P_) + Bₓ_ * T1P_ * ft[1:dQ] + ft[dP+1:end]

        for t in (dh-1):-1:1

            f_tt = f_ttm[:, 1, t]
            P_tt = P_ttm[:, :, t]

            GPG_Q = G[1:dP+N, :] * P_tt * G[1:dP+N, :]' + Ω[1:dP+N, 1:dP+N] |> Symmetric # P[t+1|t], k by k

            e_tl = predicted_F[t+1, :] - μT[1:dP+N] - G[1:dP+N, :] * f_tt

            PGG = P_tt * G[1:dP+N, :]' / GPG_Q
            f_tt1 = f_tt + PGG * e_tl |> x -> x[1:dP+N]

            PGP = PGG * G[1:dP+N, :] * P_tt
            P_tt1 = P_tt - PGP |> Symmetric |> x -> x[1:dP+N, 1:dP+N] |> Array

            # beta(t|t+1) sampling
            ft = deepcopy(f_tt1)
            idx = diag(P_tt1) .> eps()
            while !isposdef(P_tt1[idx, idx])
                P_tt1[idx, idx] += eps() * I(sum(idx))
            end
            ft[idx] = MvNormal(f_tt1[idx], P_tt1[idx, idx]) |> rand

            predicted_F[t, :] = ft
            predicted_yield[t, :] = (Aₓ_ + Bₓ_ * T0P_) + Bₓ_ * T1P_ * ft[1:dQ] + ft[dP+1:end]
        end
    end

    spanned_factors = Matrix{Float64}(undef, T + horizon, dP)
    spanned_yield = Matrix{Float64}(undef, T + horizon, N)
    spanned_factors[1:T, :] = data
    spanned_yield[1:T, :] = yields
    if dh > 0
        spanned_factors[(T+1):(T+dh), :] = predicted_F[:, 1:dP]
        spanned_yield[(T+1):(T+dh), :] = predicted_yield
    end
    for t in (T+dh+1):(T+horizon) # predicted period
        X = spanned_factors[t-1:-1:t-p, :] |> (X -> vec(X'))
        spanned_factors[t, :] = KₚF + GₚFF * X + rand(MvNormal(zeros(dP), ΩFF))
        mea_error = [Wₚ; Wₒ] \ [zeros(dQ); rand(MvNormal(zeros(N - dQ), Matrix(diagm(Σₒ))))]
        spanned_yield[t, :] = (Aₓ_ + Bₓ_ * T0P_) + Bₓ_ * T1P_ * spanned_factors[t, 1:dQ] + mea_error
    end
    if isempty(τ)
        predicted_TP = []
    else
        predicted_TP = Matrix{Float64}(undef, horizon, length(τ))
        for i in eachindex(τ)
            predicted_TP[:, i] = _termPremium(τ[i], spanned_factors[(T-p+1):end, 1:dQ], spanned_factors[(T-p+1):end, (dQ+1):end], bτ_, T0P_, T1X_; κQ, kQ_infty, KₚF, GₚFF, ΩPP=ΩFF[1:dQ, 1:dQ])[1]
        end
    end

    spanned_factors = spanned_factors[(end-horizon+1):end, :]
    for i in 1:dP-dQ
        spanned_factors[:, dQ+i] .+= mean_macros[i]
    end
    return spanned_yield[(end-horizon+1):end, :], spanned_factors, predicted_TP
end