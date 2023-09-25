"""
    conditional_forecasts(S::Vector, τ, horizon, saved_θ, yields, macros, τₙ; mean_macros::Vector=[], data_scale=1200)
# Input
scenarios, a result of the posterior sampler, and data 
- `S[t]` = conditioned scenario at time `size(yields, 1)+t`.
    - If we need an unconditional prediction, `S = []`.
    - If you are conditionaing a scenario, I assume S = Vector{Scenario}.
- `τ` is a vector of maturities that term premiums of interest has.
- `horizon`: maximum length of the predicted path. It should not be small than `length(S)`.
- `saved_θ`: the first output of function `posterior_sampler`.
- `mean_macros::Vector`: If you demeaned macro variables, you can input the mean of the macro variables. Then, the output will be generated in terms of the un-demeaned macro variables.
# Output
- `Vector{Forecast}(, iteration)`
- `t`'th rows in predicted `yields`, predicted `factors`, and predicted `TP` are the corresponding predicted value at time `size(yields, 1)+t`.
- Mathematically, it is a posterior samples from `future observation|past observation,scenario`.
"""
function conditional_forecasts(S, τ, horizon, saved_θ, yields, macros, τₙ; mean_macros::Vector=[], data_scale=1200)
    iteration = length(saved_θ)
    scenarios = Vector{Forecast}(undef, iteration)
    prog = Progress(iteration; dt=5, desc="conditional_forecasts...")
    Threads.@threads for iter in 1:iteration

        κQ = saved_θ[:κQ][iter]
        kQ_infty = saved_θ[:kQ_infty][iter]
        ϕ = saved_θ[:ϕ][iter]
        σ²FF = saved_θ[:σ²FF][iter]
        Σₒ = saved_θ[:Σₒ][iter]

        if isempty(S)
            spanned_yield, spanned_F, predicted_TP = _unconditional_forecasts(τ, horizon, yields, macros, τₙ; κQ, kQ_infty, ϕ, σ²FF, Σₒ, mean_macros, data_scale)
        else
            spanned_yield, spanned_F, predicted_TP = _conditional_forecasts(S, τ, horizon, yields, macros, τₙ; κQ, kQ_infty, ϕ, σ²FF, Σₒ, mean_macros, data_scale)
        end
        scenarios[iter] = Forecast(yields=deepcopy(spanned_yield), factors=deepcopy(spanned_F), TP=deepcopy(predicted_TP))

        next!(prog)
    end
    finish!(prog)

    return scenarios
end

"""
    _unconditional_forecasts(τ, horizon, yields, macros, τₙ; κQ, kQ_infty, ϕ, σ²FF, Σₒ, mean_macros, data_scale)
"""
function _unconditional_forecasts(τ, horizon, yields, macros, τₙ; κQ, kQ_infty, ϕ, σ²FF, Σₒ, mean_macros, data_scale)

    ## Construct GDTSM parameters
    ϕ0, C = ϕ_2_ϕ₀_C(; ϕ)
    ϕ0 = C \ ϕ0 # reduced form parameters
    KₚF = ϕ0[:, 1]
    GₚFF = ϕ0[:, 2:end]
    ΩFF = (C \ diagm(σ²FF)) / C' |> Symmetric

    N = length(τₙ)
    dQ = dimQ()
    dP = size(ΩFF, 1)
    p = Int(size(GₚFF, 2) / dP)
    PCs, ~, Wₚ, Wₒ, mean_PCs = PCA(yields, p)
    W = [Wₒ; Wₚ]
    W_inv = inv(W)

    if isempty(mean_macros)
        mean_macros = zeros(dP - dQ)
    end

    if isempty(macros)
        data = deepcopy(PCs)
    else
        data = [PCs macros]
    end
    T = size(data, 1)

    bτ_ = bτ(τₙ[end]; κQ)
    Bₓ_ = Bₓ(bτ_, τₙ)
    aτ_ = aτ(τₙ[end], bτ_, τₙ, Wₚ; kQ_infty, ΩPP=ΩFF[1:dQ, 1:dQ], data_scale)
    Aₓ_ = Aₓ(aτ_, τₙ)
    T1X_ = T1X(Bₓ_, Wₚ)
    T1P_ = inv(T1X_)
    T0P_ = T0P(T1X_, Aₓ_, Wₚ, mean_PCs)

    spanned_factors = Matrix{Float64}(undef, T + horizon, dP)
    spanned_yield = Matrix{Float64}(undef, T + horizon, N)
    spanned_factors[1:T, :] = data
    spanned_yield[1:T, :] = yields
    for t in (T+1):(T+horizon) # predicted period
        X = spanned_factors[t-1:-1:t-p, :] |> (X -> vec(X'))
        spanned_factors[t, :] = KₚF + GₚFF * X + rand(MvNormal(zeros(dP), ΩFF))
        mea_error = W_inv * [rand(MvNormal(zeros(N - dQ), Matrix(diagm(Σₒ)))); zeros(dQ)]
        spanned_yield[t, :] = (Aₓ_ + Bₓ_ * T0P_) + Bₓ_ * T1P_ * spanned_factors[t, 1:dQ] + mea_error
    end
    if isempty(τ)
        predicted_TP = []
    else
        predicted_TP = Matrix{Float64}(undef, horizon, length(τ))
        for i in eachindex(τ)
            predicted_TP[:, i] = _termPremium(τ[i], spanned_factors[(T-p+1):end, 1:dQ], spanned_factors[(T-p+1):end, (dQ+1):end], bτ_, T0P_, T1X_; κQ, kQ_infty, KₚF, GₚFF, ΩPP=ΩFF[1:dQ, 1:dQ], data_scale)[1]
        end
    end

    spanned_factors = spanned_factors[(end-horizon+1):end, :]
    for i in 1:dP-dQ
        spanned_factors[:, dQ+i] .+= mean_macros[i]
    end
    return spanned_yield[(end-horizon+1):end, :], spanned_factors, predicted_TP
end

"""
    _conditional_forecasts(S, τ, horizon, yields, macros, τₙ; κQ, kQ_infty, ϕ, σ²FF, Σₒ, mean_macros, data_scale)
"""
function _conditional_forecasts(S, τ, horizon, yields, macros, τₙ; κQ, kQ_infty, ϕ, σ²FF, Σₒ, mean_macros, data_scale)

    ## Construct GDTSM parameters
    ϕ0, C = ϕ_2_ϕ₀_C(; ϕ)
    ϕ0 = C \ ϕ0 # reduced form parameters
    KₚF = ϕ0[:, 1]
    GₚFF = ϕ0[:, 2:end]
    ΩFF = (C \ diagm(σ²FF)) / C' |> Symmetric

    N = length(τₙ)
    dQ = dimQ()
    dP = size(ΩFF, 1)
    k = size(GₚFF, 2) + N - dQ + dP # of factors in the companion from
    p = Int(size(GₚFF, 2) / dP)
    PCs, ~, Wₚ, Wₒ, mean_PCs = PCA(yields, p)
    W = [Wₒ; Wₚ]
    W_inv = inv(W)

    if isempty(mean_macros)
        mean_macros = zeros(dP - dQ)
    end

    if isempty(macros)
        data = deepcopy(PCs)
    else
        data = [PCs macros]
    end
    T = size(data, 1)
    dh = length(S)

    bτ_ = bτ(τₙ[end]; κQ)
    Bₓ_ = Bₓ(bτ_, τₙ)
    aτ_ = aτ(τₙ[end], bτ_, τₙ, Wₚ; kQ_infty, ΩPP=ΩFF[1:dQ, 1:dQ], data_scale)
    Aₓ_ = Aₓ(aτ_, τₙ)
    T1X_ = T1X(Bₓ_, Wₚ)
    T1P_ = inv(T1X_)
    T0P_ = T0P(T1X_, Aₓ_, Wₚ, mean_PCs)

    ## Construct the Kalman filter parameters
    # Transition equation: F(t) = μT + G*F(t-1) + N(0,Ω), where F(t): dP*p+N vector
    G_sub = [GₚFF[:, 1:dP] zeros(dP, N - dQ) GₚFF[:, dP+1:end]
        zeros(N - dQ, dP * p + N - dQ)
        I(dP) zeros(dP, dP * p - dP + N - dQ)
        zeros(dP * p - 2dP, dP + N - dQ) I(dP * p - 2dP) zeros(dP * p - 2dP, dP)]
    G = zeros(k, k)
    G[1:dP*p+N-dQ, 1:dP*p+N-dQ] = G_sub
    G[1:dP, end-dP+1:end] = diagm(KₚF)
    G[end-dP+1:end, end-dP+1:end] = I(dP)

    Ω = zeros(dP + N - dQ, dP + N - dQ)
    Ω[1:dP, 1:dP] = ΩFF
    Ω[dP+1:end, dP+1:end] = diagm(Σₒ)
    ΩFL = [I(dP) zeros(dP, N - dQ)
        zeros(N - dQ, dP) I(N - dQ)
        zeros(dP * p, dP + N - dQ)]
    # Measurement equation: Y(t) = μM + H*F(t), where Y(t): N + (dP-dQ) vector
    μM = [Aₓ_ + Bₓ_ * T0P_
        zeros(dP - dQ)]
    H = [Bₓ_*T1P_ zeros(N, dP - dQ) W_inv[:, 1:N-dQ] zeros(N, dP * p)
        zeros(dP - dQ, dQ) I(dP - dQ) zeros(dP - dQ, dP * p + N - dQ)]

    ## Kalman filtering & smoothing
    precFtm = Vector{Vector}(undef, dh)
    Ktm = Vector{Matrix}(undef, dh)
    W_mea_error = yields[end, :] - (Aₓ_ + Bₓ_ * T0P_) - Bₓ_ * T1P_ * data[end, 1:dQ] |> x -> W * x
    init_f_ll = data[end:-1:(end-p+1), :] |> (X -> vec(X')) |> x -> [x[1:dP]; W_mea_error[1:N-dQ]; x[dP+1:end]; ones(dP)]
    init_P_ll = zeros(k, k)

    function filtering_smoothing(S_)

        f_tt = deepcopy(init_f_ll)
        P_tt = deepcopy(init_P_ll)
        # Kalman filtering
        for t = 1:dh

            f_tl = G * f_tt
            P_tl = G * P_tt * G' + ΩFL * Ω * ΩFL'

            # st = St*μM + St*H*F(t)
            St = S_[t].combinations
            st = S_[t].values

            if maximum(abs.(St)) > 0
                var_tl = (St * H) * P_tl * (St * H)' |> Symmetric
                e_tl = st - St * μM - St * H * f_tl
                Kalgain = P_tl * (St * H)' / var_tl

                precFtm[t] = var_tl \ e_tl
                Ktm[t] = G * Kalgain

                f_tt = f_tl + Kalgain * e_tl
                P_tt = P_tl - Kalgain * St * H * P_tl
            else
                precFtm[t] = zeros(size(St, 1))
                Ktm[t] = zeros(k, size(St, 1))

                f_tt = deepcopy(f_tl)
                P_tt = deepcopy(P_tl)
            end
            idx = diag(P_tt) .< eps()
            P_tt[idx, :] .= 0
            P_tt[:, idx] .= 0

        end

        ## Backward recursion
        predicted_F = zeros(dh, k)  # T by k
        predicted_errors = zeros(dh, dP + N - dQ)  # T by k
        rt = zeros(k)
        r0 = 0

        for t in dh:-1:1
            precFt = precFtm[t]
            Kt = Ktm[t]
            St = S_[t].combinations

            ut = precFt - Kt' * rt
            predicted_errors[t, :] = Ω * ΩFL' * rt
            rt = (St * H)' * ut + G' * rt
            if t == 1
                r0 = deepcopy(rt)
            end
        end

        predicted_F[1, :] = G * init_f_ll + (G * init_P_ll * G' + ΩFL * Ω * ΩFL') * r0
        for t in 2:dh
            predicted_F[t, :] = G * predicted_F[t-1, :] + ΩFL * predicted_errors[t-1, :]
        end

        return predicted_F
    end

    # For generating posterior samples
    function data_generating()
        auxS = Vector{Scenario}(undef, dh)
        ftm = Matrix{Float64}(undef, dh, k)

        St = S[1].combinations
        ftm[1, :] = G * init_f_ll + ΩFL * rand(MvNormal(zeros(dP + N - dQ), Ω))
        if maximum(abs.(St)) > 0
            auxS[1] = Scenario(combinations=deepcopy(St), values=St * μM + St * H * ftm[1, :])
        else
            auxS[1] = Scenario(combinations=deepcopy(St), values=zeros(size(St, 1)))
        end

        for t in 2:dh
            St = S[t].combinations
            ftm[t, :] = G * ftm[t-1, :] + ΩFL * rand(MvNormal(zeros(dP + N - dQ), Ω))
            if maximum(abs.(St)) > 0
                auxS[t] = Scenario(combinations=deepcopy(St), values=St * μM + St * H * ftm[t, :])
            else
                auxS[t] = Scenario(combinations=deepcopy(St), values=zeros(size(St, 1)))
            end
        end

        return auxS, ftm
    end

    ## Do the simulation smoothing
    auxS, auxF = data_generating()
    aux_filteredF = filtering_smoothing(auxS)
    filteredF = filtering_smoothing(S)
    predicted_F = auxF - aux_filteredF + filteredF

    spanned_factors = Matrix{Float64}(undef, T + horizon, dP)
    spanned_yield = Matrix{Float64}(undef, T + horizon, N)
    spanned_factors[1:T, :] = data
    spanned_yield[1:T, :] = yields
    spanned_factors[(T+1):(T+dh), :] = predicted_F[:, 1:dP]
    for t in (T+1):(T+horizon) # predicted period
        if t > T + dh
            X = spanned_factors[t-1:-1:t-p, :] |> X -> vec(X')
            spanned_factors[t, :] = KₚF + GₚFF * X + rand(MvNormal(zeros(dP), ΩFF))
            mea_error = W_inv * [rand(MvNormal(zeros(N - dQ), Matrix(diagm(Σₒ)))); zeros(dQ)]
        else
            mea_error = W_inv * [predicted_F[t-T, dP+1:dP+N-dQ]; zeros(dQ)]
        end
        spanned_yield[t, :] = (Aₓ_ + Bₓ_ * T0P_) + Bₓ_ * T1P_ * spanned_factors[t, 1:dQ] + mea_error
    end
    if isempty(τ)
        predicted_TP = []
    else
        predicted_TP = Matrix{Float64}(undef, horizon, length(τ))
        for i in eachindex(τ)
            predicted_TP[:, i] = _termPremium(τ[i], spanned_factors[(T-p+1):end, 1:dQ], spanned_factors[(T-p+1):end, (dQ+1):end], bτ_, T0P_, T1X_; κQ, kQ_infty, KₚF, GₚFF, ΩPP=ΩFF[1:dQ, 1:dQ], data_scale)[1]
        end
    end

    spanned_factors = spanned_factors[(end-horizon+1):end, :]
    for i in 1:dP-dQ
        spanned_factors[:, dQ+i] .+= mean_macros[i]
    end
    return spanned_yield[(end-horizon+1):end, :], spanned_factors, predicted_TP
end

"""
    scenario_analysis(S::Vector, τ, horizon, saved_θ, yields, macros, τₙ; mean_macros::Vector=[], data_scale=1200)
# Input
scenarios, a result of the posterior sampler, and data 
- `S[t]` = conditioned scenario at time `size(yields, 1)+t`.
    - Set `S = []` if you need an unconditional prediction. 
    - If you are conditionaing a scenario, I assume S = Vector{Scenario}.
- `τ` is a vector of maturities that term premiums of interest has.
- `horizon`: maximum length of the predicted path. It should not be small than `length(S)`.
- `saved_θ`: the first output of function `posterior_sampler`.
- `mean_macros::Vector`: If you demeaned macro variables, you can input the mean of the macro variables. Then, the output will be generated in terms of the un-demeaned macro variables.
# Output
- `Vector{Forecast}(, iteration)`
- `t`'th rows in predicted `yields`, predicted `factors`, and predicted `TP` are the corresponding predicted value at time `size(yields, 1)+t`.
- Mathematically, it is a posterior distribution of `E[future obs|past obs, scenario, parameters]`.
"""
function scenario_analysis(S::Vector, τ, horizon, saved_θ, yields, macros, τₙ; mean_macros::Vector=[], data_scale=1200)
    iteration = length(saved_θ)
    scenarios = Vector{Forecast}(undef, iteration)
    prog = Progress(iteration; dt=5, desc="scenario_analysis...")
    Threads.@threads for iter in 1:iteration

        κQ = saved_θ[:κQ][iter]
        kQ_infty = saved_θ[:kQ_infty][iter]
        ϕ = saved_θ[:ϕ][iter]
        σ²FF = saved_θ[:σ²FF][iter]
        Σₒ = saved_θ[:Σₒ][iter]

        if isempty(S)
            spanned_yield, spanned_F, predicted_TP = _scenario_analysis_unconditional(τ, horizon, yields, macros, τₙ; κQ, kQ_infty, ϕ, σ²FF, Σₒ, mean_macros, data_scale)
        else
            spanned_yield, spanned_F, predicted_TP = _scenario_analysis(S, τ, horizon, yields, macros, τₙ; κQ, kQ_infty, ϕ, σ²FF, Σₒ, mean_macros, data_scale)
        end
        scenarios[iter] = Forecast(yields=deepcopy(spanned_yield), factors=deepcopy(spanned_F), TP=deepcopy(predicted_TP))
        next!(prog)
    end
    finish!(prog)

    return scenarios
end


"""
    _scenario_analysis(S, τ, horizon, yields, macros, τₙ; κQ, kQ_infty, ϕ, σ²FF, Σₒ, mean_macros, data_scale)
"""
function _scenario_analysis(S, τ, horizon, yields, macros, τₙ; κQ, kQ_infty, ϕ, σ²FF, Σₒ, mean_macros, data_scale)

    ## Construct GDTSM parameters
    ϕ0, C = ϕ_2_ϕ₀_C(; ϕ)
    ϕ0 = C \ ϕ0 # reduced form parameters
    KₚF = ϕ0[:, 1]
    GₚFF = ϕ0[:, 2:end]
    ΩFF = (C \ diagm(σ²FF)) / C' |> Symmetric

    N = length(τₙ)
    dQ = dimQ()
    dP = size(ΩFF, 1)
    k = size(GₚFF, 2) + N - dQ + dP # of factors in the companion from
    p = Int(size(GₚFF, 2) / dP)
    dh = length(S) # a time series length of the scenario, dh = 0 for an unconditional prediction

    PCs, ~, Wₚ, Wₒ, mean_PCs = PCA(yields, p)
    W = [Wₒ; Wₚ]
    W_inv = inv(W)

    if isempty(mean_macros)
        mean_macros = zeros(dP - dQ)
    end

    if isempty(macros)
        data = deepcopy(PCs)
    else
        data = [PCs macros]
    end
    T = size(data, 1)

    bτ_ = bτ(τₙ[end]; κQ)
    Bₓ_ = Bₓ(bτ_, τₙ)
    aτ_ = aτ(τₙ[end], bτ_, τₙ, Wₚ; kQ_infty, ΩPP=ΩFF[1:dQ, 1:dQ], data_scale)
    Aₓ_ = Aₓ(aτ_, τₙ)
    T1X_ = T1X(Bₓ_, Wₚ)
    T1P_ = inv(T1X_)
    T0P_ = T0P(T1X_, Aₓ_, Wₚ, mean_PCs)

    ## Construct the Kalman filter parameters
    # Transition equation: F(t) = μT + G*F(t-1) + N(0,Ω), where F(t): dP*p+N vector
    G_sub = [GₚFF[:, 1:dP] zeros(dP, N - dQ) GₚFF[:, dP+1:end]
        zeros(N - dQ, dP * p + N - dQ)
        I(dP) zeros(dP, dP * p - dP + N - dQ)
        zeros(dP * p - 2dP, dP + N - dQ) I(dP * p - 2dP) zeros(dP * p - 2dP, dP)]
    G = zeros(k, k)
    G[1:dP*p+N-dQ, 1:dP*p+N-dQ] = G_sub
    G[1:dP, end-dP+1:end] = diagm(KₚF)
    G[end-dP+1:end, end-dP+1:end] = I(dP)

    Ω = zeros(dP + N - dQ, dP + N - dQ)
    Ω[1:dP, 1:dP] = ΩFF
    Ω[dP+1:end, dP+1:end] = diagm(Σₒ)
    ΩFL = [I(dP) zeros(dP, N - dQ)
        zeros(N - dQ, dP) I(N - dQ)
        zeros(dP * p, dP + N - dQ)]
    # Measurement equation: Y(t) = μM + H*F(t), where Y(t): N + (dP-dQ) vector
    μM = [Aₓ_ + Bₓ_ * T0P_
        zeros(dP - dQ)]
    H = [Bₓ_*T1P_ zeros(N, dP - dQ) W_inv[:, 1:N-dQ] zeros(N, dP * p)
        zeros(dP - dQ, dQ) I(dP - dQ) zeros(dP - dQ, dP * p + N - dQ)]

    ## initializing
    # for conditional prediction
    precFtm = Vector{Vector}(undef, dh)
    Ktm = Vector{Matrix}(undef, dh)
    W_mea_error = yields[end, :] - (Aₓ_ + Bₓ_ * T0P_) - Bₓ_ * T1P_ * data[end, 1:dQ] |> x -> W * x
    init_f_ll = data[end:-1:(end-p+1), :] |> (X -> vec(X')) |> x -> [x[1:dP]; W_mea_error[1:N-dQ]; x[dP+1:end]; ones(dP)]
    init_P_ll = zeros(k, k)

    ## Conditional prediction
    function filtering_smoothing(S_)

        f_tt = deepcopy(init_f_ll)
        P_tt = deepcopy(init_P_ll)
        # Kalman filtering
        for t = 1:dh

            f_tl = G * f_tt
            P_tl = G * P_tt * G' + ΩFL * Ω * ΩFL'

            # st = St*μM + St*H*F(t)
            St = S_[t].combinations
            st = S_[t].values

            if maximum(abs.(St)) > 0
                var_tl = (St * H) * P_tl * (St * H)' |> Symmetric
                e_tl = st - St * μM - St * H * f_tl
                Kalgain = P_tl * (St * H)' / var_tl

                precFtm[t] = var_tl \ e_tl
                Ktm[t] = G * Kalgain

                f_tt = f_tl + Kalgain * e_tl
                P_tt = P_tl - Kalgain * St * H * P_tl
            else
                precFtm[t] = zeros(size(St, 1))
                Ktm[t] = zeros(k, size(St, 1))

                f_tt = deepcopy(f_tl)
                P_tt = deepcopy(P_tl)
            end
            idx = diag(P_tt) .< eps()
            P_tt[idx, :] .= 0
            P_tt[:, idx] .= 0

        end

        ## Backward recursion
        predicted_F = zeros(dh, k)  # T by k
        predicted_errors = zeros(dh, dP + N - dQ)  # T by k
        rt = zeros(k)
        r0 = 0

        for t in dh:-1:1
            precFt = precFtm[t]
            Kt = Ktm[t]
            St = S_[t].combinations

            ut = precFt - Kt' * rt
            predicted_errors[t, :] = Ω * ΩFL' * rt
            rt = (St * H)' * ut + G' * rt
            if t == 1
                r0 = deepcopy(rt)
            end
        end

        predicted_F[1, :] = G * init_f_ll + (G * init_P_ll * G' + ΩFL * Ω * ΩFL') * r0
        for t in 2:dh
            predicted_F[t, :] = G * predicted_F[t-1, :] + ΩFL * predicted_errors[t-1, :]
        end

        return predicted_F
    end
    predicted_F = filtering_smoothing(S)

    spanned_factors = Matrix{Float64}(undef, T + horizon, dP)
    spanned_yield = Matrix{Float64}(undef, T + horizon, N)
    spanned_factors[1:T, :] = data
    spanned_yield[1:T, :] = yields
    spanned_factors[(T+1):(T+dh), :] = predicted_F[:, 1:dP]
    for t in (T+1):(T+horizon) # predicted period
        if t > T + dh
            X = spanned_factors[t-1:-1:t-p, :] |> X -> vec(X')
            spanned_factors[t, :] = KₚF + GₚFF * X
        end
        spanned_yield[t, :] = (Aₓ_ + Bₓ_ * T0P_) + Bₓ_ * T1P_ * spanned_factors[t, 1:dQ]
    end
    if isempty(τ)
        predicted_TP = []
    else
        predicted_TP = Matrix{Float64}(undef, horizon, length(τ))
        for i in eachindex(τ)
            predicted_TP[:, i] = _termPremium(τ[i], spanned_factors[(T-p+1):end, 1:dQ], spanned_factors[(T-p+1):end, (dQ+1):end], bτ_, T0P_, T1X_; κQ, kQ_infty, KₚF, GₚFF, ΩPP=ΩFF[1:dQ, 1:dQ], data_scale)[1]
        end
    end

    spanned_factors = spanned_factors[(end-horizon+1):end, :]
    for i in 1:dP-dQ
        spanned_factors[:, dQ+i] .+= mean_macros[i]
    end
    return spanned_yield[(end-horizon+1):end, :], spanned_factors, predicted_TP
end

"""
    _scenario_analysis_unconditional(τ, horizon, yields, macros, τₙ; κQ, kQ_infty, ϕ, σ²FF, Σₒ, mean_macros, data_scale)
"""
function _scenario_analysis_unconditional(τ, horizon, yields, macros, τₙ; κQ, kQ_infty, ϕ, σ²FF, Σₒ, mean_macros, data_scale)

    ## Construct GDTSM parameters
    ϕ0, C = ϕ_2_ϕ₀_C(; ϕ)
    ϕ0 = C \ ϕ0 # reduced form parameters
    KₚF = ϕ0[:, 1]
    GₚFF = ϕ0[:, 2:end]
    ΩFF = (C \ diagm(σ²FF)) / C' |> Symmetric

    N = length(τₙ)
    dQ = dimQ()
    dP = size(ΩFF, 1)
    k = size(GₚFF, 2) + N - dQ + dP # of factors in the companion from
    p = Int(size(GₚFF, 2) / dP)

    PCs, ~, Wₚ, ~, mean_PCs = PCA(yields, p)

    if isempty(mean_macros)
        mean_macros = zeros(dP - dQ)
    end

    if isempty(macros)
        data = deepcopy(PCs)
    else
        data = [PCs macros]
    end
    T = size(data, 1)

    bτ_ = bτ(τₙ[end]; κQ)
    Bₓ_ = Bₓ(bτ_, τₙ)
    aτ_ = aτ(τₙ[end], bτ_, τₙ, Wₚ; kQ_infty, ΩPP=ΩFF[1:dQ, 1:dQ], data_scale)
    Aₓ_ = Aₓ(aτ_, τₙ)
    T1X_ = T1X(Bₓ_, Wₚ)
    T1P_ = inv(T1X_)
    T0P_ = T0P(T1X_, Aₓ_, Wₚ, mean_PCs)

    ## Construct the Kalman filter parameters
    # Transition equation: F(t) = μT + G*F(t-1) + N(0,Ω), where F(t): dP*p+N vector
    G_sub = [GₚFF[:, 1:dP] zeros(dP, N - dQ) GₚFF[:, dP+1:end]
        zeros(N - dQ, dP * p + N - dQ)
        I(dP) zeros(dP, dP * p - dP + N - dQ)
        zeros(dP * p - 2dP, dP + N - dQ) I(dP * p - 2dP) zeros(dP * p - 2dP, dP)]
    G = zeros(k, k)
    G[1:dP*p+N-dQ, 1:dP*p+N-dQ] = G_sub
    G[1:dP, end-dP+1:end] = diagm(KₚF)
    G[end-dP+1:end, end-dP+1:end] = I(dP)

    Ω = zeros(dP + N - dQ, dP + N - dQ)
    Ω[1:dP, 1:dP] = ΩFF
    Ω[dP+1:end, dP+1:end] = diagm(Σₒ)
    ΩFL = [I(dP) zeros(dP, N - dQ)
        zeros(N - dQ, dP) I(N - dQ)
        zeros(dP * p, dP + N - dQ)]

    ## initializing for unconditional_prediction
    f_ttm_u = zeros(horizon, k)
    P_ttm_u = zeros(k, k, horizon)
    f_ll_u = deepcopy(init_f_ll)
    P_ll_u = deepcopy(init_P_ll)
    yields_u = zeros(horizon, N)

    ## unconditional prediction
    for t = 1:horizon

        f_ttm_u[t, :] = G * f_ll_u
        P_ttm_u[:, :, t] = G * P_ll_u * G' + ΩFL * Ω * ΩFL'
        yields_u[t, :] = (Aₓ_ + Bₓ_ * T0P_) + Bₓ_ * T1P_ * f_ttm_u[t, 1:dQ]

        f_ll_u = f_ttm_u[t, :]
        P_ll_u = P_ttm_u[:, :, t]

    end

    spanned_factors_u = Matrix{Float64}(undef, T + horizon, dP)
    spanned_yield_u = Matrix{Float64}(undef, T + horizon, N)
    spanned_factors_u[1:T, :] = data
    spanned_yield_u[1:T, :] = yields
    spanned_factors_u[(T+1):end, :] = f_ttm_u[:, 1:dP]
    spanned_yield_u[(T+1):end, :] = yields_u

    if isempty(τ)
        predicted_TP_u = []
    else
        predicted_TP_u = Matrix{Float64}(undef, horizon, length(τ))
        for i in eachindex(τ)
            predicted_TP_u[:, i] = _termPremium(τ[i], spanned_factors_u[(T-p+1):end, 1:dQ], spanned_factors_u[(T-p+1):end, (dQ+1):end], bτ_, T0P_, T1X_; κQ, kQ_infty, KₚF, GₚFF, ΩPP=ΩFF[1:dQ, 1:dQ], data_scale)[1]
        end
    end

    spanned_factors_u = spanned_factors_u[(end-horizon+1):end, :]
    for i in 1:dP-dQ
        spanned_factors_u[:, dQ+i] .+= mean_macros[i]
    end
    return spanned_yield_u[(end-horizon+1):end, :], spanned_factors_u, predicted_TP_u
end