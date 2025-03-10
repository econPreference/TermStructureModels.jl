"""
    conditional_forecast(S::Vector, τ, horizon, saved_params, yields, macros, tau_n; baseline=[], mean_macros::Vector=[], data_scale=1200)
# Input
scenarios, a result of the posterior sampler, and data 
- `S[t]` = conditioned scenario at time `size(yields, 1)+t`.
    - If we need an unconditional prediction, `S = []`.
    - If you are conditionaing a scenario, I assume S = Vector{Scenario}.
-  `τ` is a vector. The term premium of `τ[i]`-bond is forecasted for each i.
    - If `τ` is set to `[]`, the term premium is not forecasted. 
- `horizon`: maximum length of the predicted path. It should not be small than `length(S)`.
- `saved_params`: the first output of function `posterior_sampler`.
- `baseline::Vector{Forecast}`: the output of `conditional_forecast` when `S` is empty. 
- `mean_macros::Vector`: If you demeaned macro variables, you can input the mean of the macro variables. Then, the output will be generated in terms of the un-demeaned macro variables.
- If `mean_macros` was used as an input when deriving `baseline` with this function, `mean_macros` should also be included as an input when using `baseline` as an input. Conversely, if `mean_macros` was not used as an input when deriving `baseline`, it should not be included as an input when using `baseline`.
# Output
- `Vector{Forecast}(, iteration)`
- `t`'th rows in predicted `yields`, predicted `factors`, predicted `TP`, and predicted `EH` are the corresponding predicted value at time `size(yields, 1)+t`.
- Mathematically, it is a posterior samples from `future observation|past observation,scenario`.
"""
function conditional_forecast(S::Vector, τ, horizon, saved_params, yields, macros, tau_n; baseline=[], mean_macros::Vector=[], data_scale=1200)
    iteration = length(saved_params)
    scenarios = Vector{Forecast}(undef, iteration)
    prog = Progress(iteration; dt=5, desc="conditional_forecast...")
    Threads.@threads for iter in 1:iteration

        kappaQ = saved_params[:kappaQ][iter]
        kQ_infty = saved_params[:kQ_infty][iter]
        phi = saved_params[:phi][iter]
        varFF = saved_params[:varFF][iter]
        SigmaO = saved_params[:SigmaO][iter]

        if isempty(baseline)
            baseline_forecast = []
        else
            dQ = dimQ() + size(yields, 2) - length(tau_n)
            idx_τ = findall(x -> x ∈ τ, tau_n)
            baseline_forecast = [baseline[iter][:yields] baseline[iter][:factors][:, dQ+1:end] baseline[iter][:yields][:, idx_τ] - baseline[iter][:TP]]
        end

        if isempty(S)
            spanned_yield, spanned_F, predicted_TP, predicted_EH = _unconditional_forecast(τ, horizon, yields, macros, tau_n; kappaQ, kQ_infty, phi, varFF, SigmaO, mean_macros, data_scale)
        else
            spanned_yield, spanned_F, predicted_TP, predicted_EH = _conditional_forecast(S, τ, horizon, yields, macros, tau_n; kappaQ, kQ_infty, phi, varFF, SigmaO, baseline_forecast, mean_macros, data_scale)
        end
        scenarios[iter] = Forecast(yields=copy(spanned_yield), factors=copy(spanned_F), TP=copy(predicted_TP), EH=copy(predicted_EH))

        next!(prog)
    end
    finish!(prog)

    return scenarios
end

"""
    _unconditional_forecast(τ, horizon, yields, macros, tau_n; kappaQ, kQ_infty, phi, varFF, SigmaO, mean_macros, data_scale)
"""
function _unconditional_forecast(τ, horizon, yields, macros, tau_n; kappaQ, kQ_infty, phi, varFF, SigmaO, mean_macros, data_scale)

    ## Construct TSM parameters
    phi0, C = phi_2_phi₀_C(; phi)
    phi0 = C \ phi0 # reduced form parameters
    KPF = phi0[:, 1]
    GPFF = phi0[:, 2:end]
    OmegaFF = (C \ diagm(varFF)) / C' |> Symmetric

    N = length(tau_n)
    dQ = dimQ() + size(yields, 2) - length(tau_n)
    dP = size(OmegaFF, 1)
    p = Int(size(GPFF, 2) / dP)
    PCs, ~, Wₚ, Wₒ, mean_PCs = PCA(yields, p; dQ)
    W = [Wₒ; Wₚ]
    W_inv = inv(W)

    if isempty(mean_macros)
        mean_macros = zeros(dP - dQ)
    end

    if isempty(macros)
        data = copy(PCs)
    else
        data = [PCs macros]
    end
    T = size(data, 1)

    bτ_ = bτ(tau_n[end]; kappaQ, dQ)
    Bₓ_ = Bₓ(bτ_, tau_n)
    aτ_ = aτ(tau_n[end], bτ_, tau_n, Wₚ; kQ_infty, ΩPP=OmegaFF[1:dQ, 1:dQ], data_scale)
    Aₓ_ = Aₓ(aτ_, tau_n)
    T1X_ = T1X(Bₓ_, Wₚ)
    T1P_ = inv(T1X_)
    T0P_ = T0P(T1X_, Aₓ_, Wₚ, mean_PCs)

    spanned_factors = Matrix{Float64}(undef, T + horizon, dP)
    spanned_yield = Matrix{Float64}(undef, T + horizon, N)
    spanned_factors[1:T, :] = data
    spanned_yield[1:T, :] = yields
    for t in (T+1):(T+horizon) # predicted period
        X = spanned_factors[t-1:-1:t-p, :] |> (X -> vec(X'))
        spanned_factors[t, :] = KPF + GPFF * X + rand(MvNormal(zeros(dP), OmegaFF))
        mea_error = W_inv * [rand(MvNormal(zeros(N - dQ), Matrix(diagm(SigmaO)))); zeros(dQ)]
        spanned_yield[t, :] = (Aₓ_ + Bₓ_ * T0P_) + Bₓ_ * T1P_ * spanned_factors[t, 1:dQ] + mea_error
    end
    if isempty(τ)
        predicted_TP = []
        predicted_EH = []
    else
        predicted_TP = Matrix{Float64}(undef, horizon, length(τ))
        for i in eachindex(τ)
            predicted_TP[:, i] = _termPremium(τ[i], spanned_factors[(T-p+1):end, 1:dQ], spanned_factors[(T-p+1):end, (dQ+1):end], bτ_, T0P_, T1X_; kappaQ, kQ_infty, KPF, GPFF, ΩPP=OmegaFF[1:dQ, 1:dQ], data_scale)[1]
        end
        predicted_EH = spanned_yield[(end-horizon+1):end, findall(x -> x ∈ τ, tau_n)] - predicted_TP
    end

    spanned_factors = spanned_factors[(end-horizon+1):end, :]
    for i in 1:dP-dQ
        spanned_factors[:, dQ+i] .+= mean_macros[i]
    end
    return spanned_yield[(end-horizon+1):end, :], spanned_factors, predicted_TP, predicted_EH
end

"""
    _conditional_forecast(S, τ, horizon, yields, macros, tau_n; kappaQ, kQ_infty, phi, varFF, SigmaO, baseline, mean_macros, data_scale)
"""
function _conditional_forecast(S, τ, horizon, yields, macros, tau_n; kappaQ, kQ_infty, phi, varFF, SigmaO, baseline_forecast, mean_macros, data_scale)

    ## Construct TSM parameters
    phi0, C = phi_2_phi₀_C(; phi)
    phi0 = C \ phi0 # reduced form parameters
    KPF = phi0[:, 1]
    GPFF = phi0[:, 2:end]
    OmegaFF = (C \ diagm(varFF)) / C' |> Symmetric

    N = length(tau_n)
    dQ = dimQ() + size(yields, 2) - length(tau_n)
    dP = size(OmegaFF, 1)
    k = size(GPFF, 2) + N - dQ + dP # of factors in the companion from
    p = Int(size(GPFF, 2) / dP)
    PCs, ~, Wₚ, Wₒ, mean_PCs = PCA(yields, p)
    W = [Wₒ; Wₚ]
    W_inv = inv(W)

    if isempty(mean_macros)
        mean_macros = zeros(dP - dQ)
    end

    if isempty(macros)
        data = copy(PCs)
    else
        data = [PCs macros]
    end
    T = size(data, 1)
    dh = length(S)

    bτ_ = bτ(tau_n[end]; kappaQ, dQ)
    Bₓ_ = Bₓ(bτ_, tau_n)
    aτ_ = aτ(tau_n[end], bτ_, tau_n, Wₚ; kQ_infty, ΩPP=OmegaFF[1:dQ, 1:dQ], data_scale)
    Aₓ_ = Aₓ(aτ_, tau_n)
    T1X_ = T1X(Bₓ_, Wₚ)
    T1P_ = inv(T1X_)
    T0P_ = T0P(T1X_, Aₓ_, Wₚ, mean_PCs)

    ## Construct the Kalman filter parameters
    # Transition equation: F(t) = μT + G*F(t-1) + N(0,Ω), where F(t): dP*p+N vector
    if p == 1
        G_sub = [GPFF zeros(dP, N - dQ)
            zeros(N - dQ, dP + N - dQ)]
    else
        G_sub = [GPFF[:, 1:dP] zeros(dP, N - dQ) GPFF[:, dP+1:end]
            zeros(N - dQ, dP * p + N - dQ)
            I(dP) zeros(dP, dP * p - dP + N - dQ)
            zeros(dP * p - 2dP, dP + N - dQ) I(dP * p - 2dP) zeros(dP * p - 2dP, dP)]
    end
    G = zeros(k, k)
    G[1:dP*p+N-dQ, 1:dP*p+N-dQ] = G_sub
    G[1:dP, end-dP+1:end] = diagm(KPF)
    G[end-dP+1:end, end-dP+1:end] = I(dP)

    Ω = zeros(dP + N - dQ, dP + N - dQ)
    Ω[1:dP, 1:dP] = OmegaFF
    Ω[dP+1:end, dP+1:end] = diagm(SigmaO)
    ΩFL = [I(dP) zeros(dP, N - dQ)
        zeros(N - dQ, dP) I(N - dQ)
        zeros(dP * p, dP + N - dQ)]
    # Measurement equation: Y(t) = μM + H*F(t), where Y(t): N + (dP-dQ) vector
    μM = [Aₓ_ + Bₓ_ * T0P_
        zeros(dP - dQ)]
    H = [Bₓ_*T1P_ zeros(N, dP - dQ) W_inv[:, 1:N-dQ] zeros(N, dP * p)
        zeros(dP - dQ, dQ) I(dP - dQ) zeros(dP - dQ, dP * p + N - dQ)]

    # Measurement equation: term premium
    if length(τ) > 0
        if p == 1
            GP = [GPFF diagm(KPF)
                zeros(dP, dP) I(dP)]
        else
            GP = [GPFF diagm(KPF)
                I(dP * p - dP) zeros(dP * p - dP, 2dP)
                zeros(dP, dP * p) I(dP)]
        end
        Gpower = Array{Float64}(undef, size(GP, 1), size(GP, 2), length(τ))
        Gpower_ind = I(size(GP, 1))
        Gpower_sum = I(size(GP, 1))
        for i in 1:τ[end]
            if i ∈ τ
                idx = findall(x -> x == i, τ)
                Gpower[:, :, idx] = Gpower_sum ./ i
            end
            Gpower_ind *= GP
            Gpower_sum += Gpower_ind
        end

        eh_const = sum(T0P_)
        append!(μM, ones(length(τ)) * eh_const)

        eh_fl = Matrix{Float64}(undef, length(τ), k)
        for i in axes(eh_fl, 1)
            eh_fl[i, :] = sum(T1P_, dims=1) * [I(dQ) zeros(dQ, p * dP + dP - dQ)] * Gpower[:, :, i] |> x -> [x[1, 1:dP]; zeros(N - dQ); x[1, dP+1:end]]
            H = vcat(H, eh_fl[i, :]')
        end
    end

    ## Kalman filtering & smoothing
    precFtm = Vector{Vector}(undef, dh)
    Ktm = Vector{Matrix}(undef, dh)
    W_mea_error = yields[end, :] - (Aₓ_ + Bₓ_ * T0P_) - Bₓ_ * T1P_ * data[end, 1:dQ] |> x -> W * x
    init_f_ll = data[end:-1:(end-p+1), :] |> (X -> vec(X')) |> x -> [x[1:dP]; W_mea_error[1:N-dQ]; x[dP+1:end]; ones(dP)]
    init_P_ll = zeros(k, k)

    function filtering_smoothing(S_)

        f_tt = copy(init_f_ll)
        P_tt = copy(init_P_ll)
        # Kalman filtering
        for t = 1:dh

            f_tl = G * f_tt
            P_tl = G * P_tt * G' + ΩFL * Ω * ΩFL'

            # st = St*μM + St*H*F(t)
            St = S_[t].combinations
            st = S_[t].values
            if !isempty(baseline_forecast)
                st += St * (baseline_forecast[t, :] - [zeros(length(tau_n)); mean_macros; zeros(length(τ))])
            end
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

                f_tt = copy(f_tl)
                P_tt = copy(P_tl)
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
                r0 = copy(rt)
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
            auxS[1] = Scenario(combinations=copy(St), values=St * μM + St * H * ftm[1, :])
        else
            auxS[1] = Scenario(combinations=copy(St), values=zeros(size(St, 1)))
        end

        for t in 2:dh
            St = S[t].combinations
            ftm[t, :] = G * ftm[t-1, :] + ΩFL * rand(MvNormal(zeros(dP + N - dQ), Ω))
            if maximum(abs.(St)) > 0
                auxS[t] = Scenario(combinations=copy(St), values=St * μM + St * H * ftm[t, :])
            else
                auxS[t] = Scenario(combinations=copy(St), values=zeros(size(St, 1)))
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
            spanned_factors[t, :] = KPF + GPFF * X + rand(MvNormal(zeros(dP), OmegaFF))
            mea_error = W_inv * [rand(MvNormal(zeros(N - dQ), Matrix(diagm(SigmaO)))); zeros(dQ)]
        else
            mea_error = W_inv * [predicted_F[t-T, dP+1:dP+N-dQ]; zeros(dQ)]
        end
        spanned_yield[t, :] = (Aₓ_ + Bₓ_ * T0P_) + Bₓ_ * T1P_ * spanned_factors[t, 1:dQ] + mea_error
    end
    if isempty(τ)
        predicted_EH = []
        predicted_TP = []
    else
        predicted_EH = eh_const .+ predicted_F * eh_fl'
        predicted_TP = spanned_yield[(end-horizon+1):end, findall(x -> x ∈ τ, tau_n)] - predicted_EH
    end

    spanned_factors = spanned_factors[(end-horizon+1):end, :]
    for i in 1:dP-dQ
        spanned_factors[:, dQ+i] .+= mean_macros[i]
    end
    return spanned_yield[(end-horizon+1):end, :], spanned_factors, predicted_TP, predicted_EH
end

"""
    conditional_expectation(S::Vector, τ, horizon, saved_params, yields, macros, tau_n; baseline=[], mean_macros::Vector=[], data_scale=1200)
# Input
scenarios, a result of the posterior sampler, and data 
- `S[t]` = conditioned scenario at time `size(yields, 1)+t`.
    - Set `S = []` if you need an unconditional prediction. 
    - If you are conditionaing a scenario, I assume S = Vector{Scenario}.
- `τ` is a vector of maturities that term premiums of interest has.
- `horizon`: maximum length of the predicted path. It should not be small than `length(S)`.
- `saved_params`: the first output of function `posterior_sampler`.
- `baseline::Vector{Forecast}`: the output of `conditional_expectation` when `S` is empty.
- `mean_macros::Vector`: If you demeaned macro variables, you can input the mean of the macro variables. Then, the output will be generated in terms of the un-demeaned macro variables.
- If `mean_macros` was used as an input when deriving `baseline` with this function, `mean_macros` should also be included as an input when using `baseline` as an input. Conversely, if `mean_macros` was not used as an input when deriving `baseline`, it should not be included as an input when using `baseline`.
# Output
- `Vector{Forecast}(, iteration)`
- `t`'th rows in predicted `yields`, predicted `factors`, predicted `TP`, and predicted `EH` are the corresponding predicted value at time `size(yields, 1)+t`.
- Mathematically, it is a posterior distribution of `E[future obs|past obs, scenario, parameters]`.
"""
function conditional_expectation(S::Vector, τ, horizon, saved_params, yields, macros, tau_n; baseline=[], mean_macros::Vector=[], data_scale=1200)
    iteration = length(saved_params)
    scenarios = Vector{Forecast}(undef, iteration)
    prog = Progress(iteration; dt=5, desc="conditional_expectation...")
    Threads.@threads for iter in 1:iteration

        kappaQ = saved_params[:kappaQ][iter]
        kQ_infty = saved_params[:kQ_infty][iter]
        phi = saved_params[:phi][iter]
        varFF = saved_params[:varFF][iter]
        SigmaO = saved_params[:SigmaO][iter]

        if isempty(baseline)
            baseline_expectation = []
        else
            dQ = dimQ() + size(yields, 2) - length(tau_n)
            idx_τ = findall(x -> x ∈ τ, tau_n)
            baseline_expectation = [baseline[iter][:yields] baseline[iter][:factors][:, dQ+1:end] baseline[iter][:yields][:, idx_τ] - baseline[iter][:TP]]
        end

        if isempty(S)
            spanned_yield, spanned_F, predicted_TP, predicted_EH = _unconditional_expectation(τ, horizon, yields, macros, tau_n; kappaQ, kQ_infty, phi, varFF, SigmaO, mean_macros, data_scale)
        else
            spanned_yield, spanned_F, predicted_TP, predicted_EH = _conditional_expectation(S, τ, horizon, yields, macros, tau_n; kappaQ, kQ_infty, phi, varFF, SigmaO, baseline_expectation, mean_macros, data_scale)
        end
        scenarios[iter] = Forecast(yields=copy(spanned_yield), factors=copy(spanned_F), TP=copy(predicted_TP), EH=copy(predicted_EH))
        next!(prog)
    end
    finish!(prog)

    return scenarios
end


"""
    _conditional_expectation(S, τ, horizon, yields, macros, tau_n; kappaQ, kQ_infty, phi, varFF, SigmaO, baseline, mean_macros, data_scale)
"""
function _conditional_expectation(S, τ, horizon, yields, macros, tau_n; kappaQ, kQ_infty, phi, varFF, SigmaO, baseline_expectation, mean_macros, data_scale)

    ## Construct TSM parameters
    phi0, C = phi_2_phi₀_C(; phi)
    phi0 = C \ phi0 # reduced form parameters
    KPF = phi0[:, 1]
    GPFF = phi0[:, 2:end]
    OmegaFF = (C \ diagm(varFF)) / C' |> Symmetric

    N = length(tau_n)
    dQ = dimQ() + size(yields, 2) - length(tau_n)
    dP = size(OmegaFF, 1)
    k = size(GPFF, 2) + N - dQ + dP # of factors in the companion from
    p = Int(size(GPFF, 2) / dP)
    dh = length(S) # a time series length of the scenario, dh = 0 for an unconditional prediction

    PCs, ~, Wₚ, Wₒ, mean_PCs = PCA(yields, p)
    W = [Wₒ; Wₚ]
    W_inv = inv(W)

    if isempty(mean_macros)
        mean_macros = zeros(dP - dQ)
    end

    if isempty(macros)
        data = copy(PCs)
    else
        data = [PCs macros]
    end
    T = size(data, 1)

    bτ_ = bτ(tau_n[end]; kappaQ, dQ)
    Bₓ_ = Bₓ(bτ_, tau_n)
    aτ_ = aτ(tau_n[end], bτ_, tau_n, Wₚ; kQ_infty, ΩPP=OmegaFF[1:dQ, 1:dQ], data_scale)
    Aₓ_ = Aₓ(aτ_, tau_n)
    T1X_ = T1X(Bₓ_, Wₚ)
    T1P_ = inv(T1X_)
    T0P_ = T0P(T1X_, Aₓ_, Wₚ, mean_PCs)

    ## Construct the Kalman filter parameters
    # Transition equation: F(t) = μT + G*F(t-1) + N(0,Ω), where F(t): dP*p+N vector
    if p == 1
        G_sub = [GPFF zeros(dP, N - dQ)
            zeros(N - dQ, dP + N - dQ)]
    else
        G_sub = [GPFF[:, 1:dP] zeros(dP, N - dQ) GPFF[:, dP+1:end]
            zeros(N - dQ, dP * p + N - dQ)
            I(dP) zeros(dP, dP * p - dP + N - dQ)
            zeros(dP * p - 2dP, dP + N - dQ) I(dP * p - 2dP) zeros(dP * p - 2dP, dP)]
    end
    G = zeros(k, k)
    G[1:dP*p+N-dQ, 1:dP*p+N-dQ] = G_sub
    G[1:dP, end-dP+1:end] = diagm(KPF)
    G[end-dP+1:end, end-dP+1:end] = I(dP)

    Ω = zeros(dP + N - dQ, dP + N - dQ)
    Ω[1:dP, 1:dP] = OmegaFF
    Ω[dP+1:end, dP+1:end] = diagm(SigmaO)
    ΩFL = [I(dP) zeros(dP, N - dQ)
        zeros(N - dQ, dP) I(N - dQ)
        zeros(dP * p, dP + N - dQ)]
    # Measurement equation: Y(t) = μM + H*F(t), where Y(t): N + (dP-dQ) vector
    μM = [Aₓ_ + Bₓ_ * T0P_
        zeros(dP - dQ)]
    H = [Bₓ_*T1P_ zeros(N, dP - dQ) W_inv[:, 1:N-dQ] zeros(N, dP * p)
        zeros(dP - dQ, dQ) I(dP - dQ) zeros(dP - dQ, dP * p + N - dQ)]

    # Measurement equation: term premium
    if length(τ) > 0
        if p == 1
            GP = [GPFF diagm(KPF)
                zeros(dP, dP) I(dP)]
        else
            GP = [GPFF diagm(KPF)
                I(dP * p - dP) zeros(dP * p - dP, 2dP)
                zeros(dP, dP * p) I(dP)]
        end
        Gpower = Array{Float64}(undef, size(GP, 1), size(GP, 2), length(τ))
        Gpower_ind = I(size(GP, 1))
        Gpower_sum = I(size(GP, 1))
        for i in 1:τ[end]
            if i ∈ τ
                idx = findall(x -> x == i, τ)
                Gpower[:, :, idx] = Gpower_sum ./ i
            end
            Gpower_ind *= GP
            Gpower_sum += Gpower_ind
        end

        eh_const = sum(T0P_)
        append!(μM, ones(length(τ)) * eh_const)

        eh_fl = Matrix{Float64}(undef, length(τ), k)
        for i in axes(eh_fl, 1)
            eh_fl[i, :] = sum(T1P_, dims=1) * [I(dQ) zeros(dQ, p * dP + dP - dQ)] * Gpower[:, :, i] |> x -> [x[1, 1:dP]; zeros(N - dQ); x[1, dP+1:end]]
            H = vcat(H, eh_fl[i, :]')
        end
    end

    ## initializing
    # for conditional prediction
    precFtm = Vector{Vector}(undef, dh)
    Ktm = Vector{Matrix}(undef, dh)
    W_mea_error = yields[end, :] - (Aₓ_ + Bₓ_ * T0P_) - Bₓ_ * T1P_ * data[end, 1:dQ] |> x -> W * x
    init_f_ll = data[end:-1:(end-p+1), :] |> (X -> vec(X')) |> x -> [x[1:dP]; W_mea_error[1:N-dQ]; x[dP+1:end]; ones(dP)]
    init_P_ll = zeros(k, k)

    ## Conditional prediction
    function filtering_smoothing(S_)

        f_tt = copy(init_f_ll)
        P_tt = copy(init_P_ll)
        # Kalman filtering
        for t = 1:dh

            f_tl = G * f_tt
            P_tl = G * P_tt * G' + ΩFL * Ω * ΩFL'

            # st = St*μM + St*H*F(t)
            St = S_[t].combinations
            st = S_[t].values
            if !isempty(baseline_expectation)
                st += St * (baseline_expectation[t, :] - [zeros(length(tau_n)); mean_macros; zeros(length(τ))])
            end

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

                f_tt = copy(f_tl)
                P_tt = copy(P_tl)
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
                r0 = copy(rt)
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
            spanned_factors[t, :] = KPF + GPFF * X
            mea_error = zeros(N)
        else
            mea_error = W_inv * [predicted_F[t-T, dP+1:dP+N-dQ]; zeros(dQ)]
        end

        spanned_yield[t, :] = (Aₓ_ + Bₓ_ * T0P_) + Bₓ_ * T1P_ * spanned_factors[t, 1:dQ] + mea_error
    end
    if isempty(τ)
        predicted_EH = []
        predicted_TP = []
    else
        predicted_EH = eh_const .+ predicted_F * eh_fl'
        predicted_TP = spanned_yield[(end-horizon+1):end, findall(x -> x ∈ τ, tau_n)] - predicted_EH
    end

    spanned_factors = spanned_factors[(end-horizon+1):end, :]
    for i in 1:dP-dQ
        spanned_factors[:, dQ+i] .+= mean_macros[i]
    end
    return spanned_yield[(end-horizon+1):end, :], spanned_factors, predicted_TP, predicted_EH
end

"""
    _unconditional_expectation(τ, horizon, yields, macros, tau_n; kappaQ, kQ_infty, phi, varFF, SigmaO, mean_macros, data_scale)
"""
function _unconditional_expectation(τ, horizon, yields, macros, tau_n; kappaQ, kQ_infty, phi, varFF, SigmaO, mean_macros, data_scale)

    ## Construct TSM parameters
    phi0, C = phi_2_phi₀_C(; phi)
    phi0 = C \ phi0 # reduced form parameters
    KPF = phi0[:, 1]
    GPFF = phi0[:, 2:end]
    OmegaFF = (C \ diagm(varFF)) / C' |> Symmetric

    N = length(tau_n)
    dQ = dimQ() + size(yields, 2) - length(tau_n)
    dP = size(OmegaFF, 1)
    k = size(GPFF, 2) + N - dQ + dP # of factors in the companion from
    p = Int(size(GPFF, 2) / dP)

    PCs, ~, Wₚ, Wₒ, mean_PCs = PCA(yields, p)
    W = [Wₒ; Wₚ]

    if isempty(mean_macros)
        mean_macros = zeros(dP - dQ)
    end

    if isempty(macros)
        data = copy(PCs)
    else
        data = [PCs macros]
    end
    T = size(data, 1)

    bτ_ = bτ(tau_n[end]; kappaQ, dQ)
    Bₓ_ = Bₓ(bτ_, tau_n)
    aτ_ = aτ(tau_n[end], bτ_, tau_n, Wₚ; kQ_infty, ΩPP=OmegaFF[1:dQ, 1:dQ], data_scale)
    Aₓ_ = Aₓ(aτ_, tau_n)
    T1X_ = T1X(Bₓ_, Wₚ)
    T1P_ = inv(T1X_)
    T0P_ = T0P(T1X_, Aₓ_, Wₚ, mean_PCs)

    ## Construct the Kalman filter parameters
    # Transition equation: F(t) = μT + G*F(t-1) + N(0,Ω), where F(t): dP*p+N vector
    if p == 1
        G_sub = [GPFF zeros(dP, N - dQ)
            zeros(N - dQ, dP + N - dQ)]
    else
        G_sub = [GPFF[:, 1:dP] zeros(dP, N - dQ) GPFF[:, dP+1:end]
            zeros(N - dQ, dP * p + N - dQ)
            I(dP) zeros(dP, dP * p - dP + N - dQ)
            zeros(dP * p - 2dP, dP + N - dQ) I(dP * p - 2dP) zeros(dP * p - 2dP, dP)]
    end
    G = zeros(k, k)
    G[1:dP*p+N-dQ, 1:dP*p+N-dQ] = G_sub
    G[1:dP, end-dP+1:end] = diagm(KPF)
    G[end-dP+1:end, end-dP+1:end] = I(dP)

    Ω = zeros(dP + N - dQ, dP + N - dQ)
    Ω[1:dP, 1:dP] = OmegaFF
    Ω[dP+1:end, dP+1:end] = diagm(SigmaO)
    ΩFL = [I(dP) zeros(dP, N - dQ)
        zeros(N - dQ, dP) I(N - dQ)
        zeros(dP * p, dP + N - dQ)]

    ## initializing for unconditional_prediction
    W_mea_error = yields[end, :] - (Aₓ_ + Bₓ_ * T0P_) - Bₓ_ * T1P_ * data[end, 1:dQ] |> x -> W * x
    f_ttm_u = zeros(horizon, k)
    P_ttm_u = zeros(k, k, horizon)
    f_ll_u = data[end:-1:(end-p+1), :] |> (X -> vec(X')) |> x -> [x[1:dP]; W_mea_error[1:N-dQ]; x[dP+1:end]; ones(dP)]
    P_ll_u = zeros(k, k)
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
        predicted_EH_u = []
    else
        predicted_TP_u = Matrix{Float64}(undef, horizon, length(τ))
        for i in eachindex(τ)
            predicted_TP_u[:, i] = _termPremium(τ[i], spanned_factors_u[(T-p+1):end, 1:dQ], spanned_factors_u[(T-p+1):end, (dQ+1):end], bτ_, T0P_, T1X_; kappaQ, kQ_infty, KPF, GPFF, ΩPP=OmegaFF[1:dQ, 1:dQ], data_scale)[1]
        end
        predicted_EH_u = spanned_yield_u[(end-horizon+1):end, findall(x -> x ∈ τ, tau_n)] - predicted_TP_u
    end

    spanned_factors_u = spanned_factors_u[(end-horizon+1):end, :]
    for i in 1:dP-dQ
        spanned_factors_u[:, dQ+i] .+= mean_macros[i]
    end
    return spanned_yield_u[(end-horizon+1):end, :], spanned_factors_u, predicted_TP_u, predicted_EH_u
end