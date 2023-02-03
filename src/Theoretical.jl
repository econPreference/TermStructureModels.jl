
module Theoretical
#This module includes analytical results of our no-arbitrage Gaussian dynamic term structure model. For a specific theoretical settings, see our paper.

"""
κQ governs a conditional mean of the Q-dynamics of X, and its slope matrix has a restricted form. This function shows that restricted form.

    * Input: the DNS decay parameter

    * Output: slope matrix of the Q-conditional mean of X
        * If there is no argument, it returns the reduced dimension of yields
===
"""
function GQ_XX(; κQ)
    X = [1 0 0
        0 exp(-κQ) 1
        0 0 exp(-κQ)]
    return X
end
function GQ_XX()
    return 3
end

"""
It solves the difference equation for b_τ, where ``P_{\\tau,t}=\\exp[-a_{\\tau}-b_{\\tau}^{\\prime}X_{t}].`` 
    
    * Input
        * κQ: the DNS decay parameter
        * N: the DE solved for one to N maturities
        
    * Output: Solved factor loadings are saved with dQ by N matrix. Therefore, each column vector in the output is a factor loading for a specific maturity.
===
"""
function bτ(N; κQ)
    dQ = GQ_XX()
    GQ_XX_ = GQ_XX(; κQ)
    ι = ones(dQ)

    b = ones(dQ, N) # derived factor loadings
    for i in 2:N
        b[:, i] = ι + GQ_XX_' * b[:, i-1]
    end

    return b
end

"""
Suppose that Rₜ a yield vector. Then, for a bond market factor Xₜ, 
```math
R_{t}=\\mathcal{A}_{X}+\\mathcal{B}{}_{X}^{\\prime}X_{t}+\\begin{bmatrix}O_{d_{\\mathbb{Q}}\\times1}\\\
e_{\\mathcal{O},t}
\\end{bmatrix}
```

    * Input: Output of function bτ, and observed maturity τₙ.

    * Output: Bₓ
===
"""
function Bₓ(bτ_, τₙ)
    return bτ_[:, τₙ] ./ τₙ
end


"""
The affine transformation from latent X to obseved PC P is P = T0X + T1X*X.

    * Input: Output of function Bₓ and PC rotation matrix Wₚ

    * Output: T1X
===
"""
function T1X(Bₓ_; Wₚ)
    return Wₚ * Bₓ_'
end

"""
It solves the difference equation for a_τ, where ``P_{\\tau,t}=\\exp[-a_{\\tau}-b_{\\tau}^{\\prime}X_{t}].``
    
    * Input
        * N: the DE solved for one to N maturities
        * Output of function bτ and observed maturity τₙ.
        
    * Output: Solved factor loadings are saved with dQ by N matrix. Therefore, each column vector in the output is a factor loading for a specific maturity.
===
"""
function aτ(N, bτ_, τₙ; kQ_infty, ΩPP, Wₚ)

    a = zeros(N)
    T1X_ = T1X(Bₓ(bτ_, τₙ); Wₚ)
    for i in 2:N
        a[i] = a[i-1] - Jensens(i, bτ_, T1X_; ΩPP) + (i - 1) * kQ_infty
    end

    return a
end

"""
This function evaluate the Jensen's Ineqaulity term. 
"""
function Jensens(τ, bτ_, T1X_; ΩPP, annual=1)
    J = 0.5 * ΩPP
    J = (T1X_ \ J) / (T1X_')
    J = bτ_[:, τ-1]' * J * bτ_[:, τ-1]
    if annual == 1
        J /= 1200
    end
    return J
end

"""
Suppose that Rₜ a yield vector. Then, for a bond market factor Xₜ, 
```math
R_{t}=\\mathcal{A}_{X}+\\mathcal{B}{}_{X}^{\\prime}X_{t}+\\begin{bmatrix}O_{d_{\\mathbb{Q}}\\times1}\\\
e_{\\mathcal{O},t}
\\end{bmatrix}
```

    * Input: Output of function aτ, and observed maturity τₙ.

    * Output: Aₓ
===
"""
function Aₓ(aτ_, τₙ)
    return aτ_[τₙ] ./ τₙ
end

"""
The affine transformation from obseved PC P to latent X is X = T0P + inv(T1X)*P.

    * Input: Output of function Bₓ and PC rotation matrix Wₚ

    * Output: T1X
===
"""
function T0P(T1X_, Aₓ_; Wₚ)
    return -T1X_ \ Wₚ * Aₓ_
end

"""
Suppose that Oₜ remaining PCs. Then, for main dQ PCs Pₜ, 
```math
\\mathcal{O}_{t}=\\mathcal{A}_{\\mathcal{P}}+\\mathcal{B}_{\\mathcal{P}}\\mathcal{P}_{t}+\\mathcal{N}(O_{(N-d_{\\mathbb{Q}})\\times1},\\Sigma_{\\mathcal{O}})
```

    * Input: Output of function {Aₓ, Bₓ, T0P}, and the rotation matrix for Oₜ.

    * Output: Aₚ
===
"""
function Aₚ(Aₓ_, Bₓ_, T0P_; Wₒ)
    return Wₒ * (Aₓ_ + Bₓ_'T0P_)
end

"""
Suppose that Oₜ remaining PCs. Then, for main dQ PCs Pₜ, 
```math
\\mathcal{O}_{t}=\\mathcal{A}_{\\mathcal{P}}+\\mathcal{B}_{\\mathcal{P}}\\mathcal{P}_{t}+\\mathcal{N}(O_{(N-d_{\\mathbb{Q}})\\times1},\\Sigma_{\\mathcal{O}})
```

    * Input: Output of function {Bₓ, T1X}, and the rotation matrix for Oₜ.

    * Output: Bₚ
===
"""
function Bₚ(Bₓ_, T1X_; Wₒ)
    return (Wₒ * Bₓ_') / T1X_
end

"""
This function caluclat the term premium estimates. 

    * Input: τ is a target maturity. bτ and T1X is calculated from the corresponding functions. yields and macros are the data. GP_F is dP by p*dP matrix that contains slope coefficients of the transition equation. 
        * Remember that data should contains initial conditions, that is t = 0,1,...,p-1. 

    * Output: 
        * TP: term premium of maturity τ
        * TV_TP: contributions of each dependent variable on TP at each time t
        * const_TP: Constant part of TP
        * Jensens_: Jensen's Ineqaulity part in TP
===
"""
function TP(τ, PCs, macros, bτ_, T1X_; κQ, kQ_infty, KₚP, GP_F, ΩPP)

    # Jensen's Ineqaulity term
    Jensens_ = 0
    for i = 1:(τ-1)
        Jensens_ += Jensens(i + 1, bτ_, T1X_; ΩPP)
    end
    Jensens_ /= -τ

    # Constant term
    dQ = GQ_XX()
    KₚQ = zeros(dQ)
    KₚQ[1] = kQ_infty
    λₚ = KₚP - KₚQ
    const_TP = 0
    for i = 1:(τ-1)
        const_TP += bτ_[:, τ-i]' * (T1X_ \ λₚ)
    end
    const_TP /= -τ

    # Time-varying part
    dP = size(GP_F)[1]
    p = size(GP_F)[2] / dP
    T = size(PCs)[1] # time series length including intial conditions
    TV_TP = Matrix{Float64}(undef, T - p, dP) # saving place

    GQ_PP = T1X_ * GQ_XX(; κQ) / T1X_
    Λ_PF = GP_F[1:dQ, :]
    Λ_PF[1:dQ, 1:dQ] -= GQ_PP
    for t = (p+1):T
        # prediction part
        lag_PCs = PCs[t:-1:(t-p+1), :]
        lag_macros = macros[t:-1:(t-p+1), :]
        for horizon = 1:(τ-2)
            regressors = vec([lag_PCs[1:p, :]'
                lag_macros[1:p, :]'])
            predicted = (GP_F * regressors)'
            pushfirst!(lag_PCs, predicted[1:dQ])
            pushfirst!(lag_macros, predicted[(dQ+1):end])
        end
        lag_PCs = lag_PCs[end:-1:1, :]
        lag_macros = lag_macros[end:-1:1, :]
        lag_X = [lag_PCs lag_macros]

        # Calculate TP
        for i = 1:(τ-1), l = 1:p
            weight = bτ_[:, τ-i]' * (T1X_ \ Λ_PF)
            for j = 1:dP
                TV_TP[t-p, j] += weight[j] * lag_X[i-l+1, j]
            end
        end
    end
    TV_TP /= -τ

    TP = sum(TV_TP, dims=2) .+ Jensens_ .+ const_TP

    return TP, TV_TP, const_TP, Jensens_
end

export GQ_XX, bτ, Bₓ, T1X, aτ, Jensens, Aₓ, T0P, Aₚ, Bₚ, TP

end