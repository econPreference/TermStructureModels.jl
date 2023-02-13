"""
This function generate a log likelihood of the measurement equation.

    * Inputs: yields should excludes their initial conditions. τₙ is maturities in "yields".

    * Output: the measurement equation part in the log likelihood
===
"""
function loglik_mea(yields, τₙ, p; κQ, kQ_infty, ϕ, σ²FF, Σₒ)

    PCs, OCs, Wₚ, Wₒ = PCA(yields, p)
    PCs = PCs[(p+1):end, :]
    OCs = OCs[(p+1):end, :]

    bτ_ = bτ(τₙ[end]; κQ)
    Bₓ_ = Bₓ(bτ_, τₙ)
    T1X_ = T1X(Bₓ_; Wₚ)
    Bₚ_ = Bₚ(Bₓ_, T1X_; Wₒ)

    ΩPP = ϕ_σ²FF_2_ΩPP(; ϕ, σ²FF)
    aτ_ = aτ(τₙ[end], bτ_, τₙ; kQ_infty, ΩPP, Wₚ)
    Aₓ_ = Aₓ(aτ_, τₙ)
    T0P_ = T0P(T1X_, Aₓ_; Wₚ)
    Aₚ_ = Aₚ(Aₓ_, Bₓ_, T0P_; Wₒ)

    T = size(OCs)[1]
    mea = MvNormal(diagm(Σₒ))
    residuals = (OCs' - (Aₚ_ .+ Bₚ_ * PCs'))'

    logpdf_ = 0
    for t = 1:T
        logpdf_ += logpdf(mea, residuals[t, :])
    end

    return logpdf_
end

"""
It calculate log likelihood of the transition equation. 
    
    * Input: All data should contains initial conditions.

    * Output: log likelihood of the transition equation.
===
"""
function loglik_tran(PCs, macros; ϕ, σ²FF)

    dP = length(σ²FF)
    p = ((size(ϕ)[2] - 1) / dP) - 1
    T = size(PCs)[1]

    yϕ, Xϕ = yϕ_Xϕ(PCs, macros, p)

    logpdf_ = Vector{Float64}(undef, dP)
    for i in 1:dP
        logpdf_[i] = logpdf(MvNormal(Xϕ * (ϕ[i, :]), σ²FF[i] * I(T)), yϕ[:, i])
    end

    return sum(logpdf_)
end

"""
This function generate the dependent variable and the corresponding regressors in the orthogonal transition equation.

    * Input: PCs and macros data should contain initial conditions. p is the transition equation lag.

    * Output: yϕ and Xϕ is a full matrix. The regressors that should have been eliminated from Xϕ is excluded by the specific form of ϕ (in which zero coefficiet will do that).
=== 
"""
function yϕ_Xϕ(PCs, macros, p)
    T = size(PCs)[1]
    data = [PCs macros]
    dP = size(data)[2]

    yϕ = data[(p+1):T, :]

    Xϕ_lag = Matrix{Float64}(undef, T - p, dP * p)
    Xϕ_contem = Matrix{Float64}(undef, T - p, dP)
    for t in (p+1):T
        Xϕ_lag[t-p, :] = vec(data[(t-1):-1:(t-p), :]')'
        Xϕ_contem[t-p, :] = -data[t, :]
    end
    Xϕ = [ones(T - p) Xϕ_lag Xϕ_contem]

    return yϕ, Xϕ, [ones(T - p) Xϕ_lag], Xϕ_contem
end

"""
This function generate a matrix decomposition, called LDLt. X = L*D*L', where L is a lower triangular matrix and D is a diagonal.
How to conduct it can be found at [Wikipedia](https://en.wikipedia.org/wiki/Cholesky_decomposition#LDL_decomposition).

    * Input: Decomposed Object, X

    * Output: Decomposed Result, X = L*D*L'
===
"""
function LDLt(X)
    C = cholesky(X).L
    S = diagm(diag(C))
    L = C / S
    D = Diagonal(S^2)

    return L, D
end

"""
It construct ΩPP from statistical parameters.
"""
function ϕ_σ²FF_2_ΩPP(; ϕ, σ²FF)

    dQ = dimQ()
    dP = length(σ²FF)
    ~, C = ϕ_2_ϕ₀_C(; ϕ)

    CQQ = C[1:dQ, 1:dQ]
    return (CQQ \ diagm(σ²FF[1:dQ])) / CQQ'

end

function ϕ_2_ϕ₀_C(; ϕ)

    dP = size(ϕ)[1]
    ϕ0 = ϕ[:, 1:(end-dP)]
    C0 = ϕ[:, (end-dP+1):end]
    C = C0 + I(dP)

    return ϕ0, C, C0
end

function isstationary(GₚFF)
    dP = size(GₚFF)[1]
    p = Int(size(GₚFF)[2] / dP)

    G = [GₚFF
        I(dP * (p - 1)) zeros(dP * (p - 1), dP)]

    return maximum(abs.(eigen(G).values)) < 1
end
