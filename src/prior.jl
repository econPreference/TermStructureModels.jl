"""
    prior_σ²FF(; ν0, Ω0::Vector)
We translate the Inverse-Wishart prior to a series of the Normal-Inverse-Gamma (NIG) prior distributions. If the dimension is dₚ, there are dₚ NIG prior distributions. This function generates Inverse-Gamma priors.  
# Output: 
- prior of `σ²FF` in the LDLt decomposition,` ΩFF = inv(C)*diagm(σ²FF)*inv(C)'`
- Each element in the output follows Inverse-Gamma priors.
"""
function prior_σ²FF(; ν0, Ω0::Vector)

    dP = length(Ω0) # dimension
    σ²FF = Vector{Any}(undef, dP) # Diagonal matrix in the LDLt decomposition that follows a IG distribution

    for i in eachindex(σ²FF)
        σ²FF[i] = InverseGamma((ν0 + i - dP) / 2, Ω0[i] / 2)
    end

    return σ²FF
end

"""
    prior_C(; Ω0::Vector)
We translate the Inverse-Wishart prior to a series of the Normal-Inverse-Gamma (NIG) prior distributions. If the dimension is dₚ, there are dₚ NIG prior distributions. This function generates Normal priors.  
# Output: 
- unscaled prior of `C` in the LDLt decomposition, `ΩFF = inv(C)*diagm(σ²FF)*inv(C)'`
# Important note
prior variance for `C[i,:] = σ²FF[i]*variance of output[i,:]`
"""
function prior_C(; Ω0::Vector)

    dP = length(Ω0) # dimension
    C = Matrix{Any}(undef, dP, dP) # A lower triangular Matrix in the LDLt that follows a Normal

    C[1, 1] = Dirac(1)
    for j in 2:dP
        C[1, j] = Dirac(0)
    end
    for i in 2:dP
        for j in 1:(i-1)
            C[i, j] = Normal(0, sqrt(1 / Ω0[j]))
        end
        C[i, i] = Dirac(1)
        for j in (i+1):dP
            C[i, j] = Dirac(0)
        end
    end

    return C
end

"""
    prior_ϕ0(μϕ_const, ρ::Vector, prior_κQ_, τₙ, Wₚ; ψ0, ψ, q, ν0, Ω0, fix_const_PC1)
This part derives the prior distribution for coefficients of the lagged regressors in the orthogonalized VAR. 
# Input 
- `prior_κQ_` is a output of function `prior_κQ`.
- When `fix_const_PC1==true`, the first element in a constant term in our orthogonalized VAR is fixed to its prior mean during the posterior sampling.
# Output
- Normal prior distributions on the slope coefficient of lagged variables and intercepts in the orthogonalized equation. 
- `Output[:,1]` for intercepts, `Output[:,1+1:1+dP]` for the first lag, `Output[:,1+dP+1:1+2*dP]` for the second lag, and so on.
# Important note
prior variance for `ϕ[i,:]` = `σ²FF[i]*var(output[i,:])`
"""
function prior_ϕ0(μϕ_const, ρ::Vector, prior_κQ_, τₙ, Wₚ; ψ0, ψ, q, ν0, Ω0, fix_const_PC1)

    dP, dPp = size(ψ) # dimension & #{regressors}
    p = Int(dPp / dP) # the number of lags
    ϕ0 = Matrix{Any}(undef, dP, dPp + 1)
    dQ = dimQ() # reduced dimension of yields

    κQ_candidate = support(prior_κQ_)
    κQ_prob = probs(prior_κQ_)
    GQ_XX_mean = zeros(dQ, dQ)

    for i in eachindex(κQ_candidate)
        κQ = κQ_candidate[i]
        bτ_ = bτ(τₙ[end]; κQ)
        Bₓ_ = Bₓ(bτ_, τₙ)
        T1X_ = T1X(Bₓ_, Wₚ)
        GQ_XX_mean += (T1X_ * GQ_XX(; κQ) / T1X_) .* κQ_prob[i]
    end

    for i in 1:dQ
        if i == 1 && fix_const_PC1
            ϕ0[i, 1] = Normal(μϕ_const[i], sqrt(ψ0[i] * 1e-10))
        else
            ϕ0[i, 1] = Normal(μϕ_const[i], sqrt(ψ0[i] * q[4, 1]))
        end
        for l = 1:1
            for j in 1:dQ
                ϕ0[i, 1+dP*(l-1)+j] = Normal(GQ_XX_mean[i, j], sqrt(ψ[i, dP*(l-1)+j] * Minnesota(l, i, j; q, ν0, Ω0)))
            end
            for j in (dQ+1):dP
                ϕ0[i, 1+dP*(l-1)+j] = Normal(0, sqrt(ψ[i, dP*(l-1)+j] * Minnesota(l, i, j; q, ν0, Ω0)))
            end
        end
        for l = 2:p
            for j in 1:dP
                ϕ0[i, 1+dP*(l-1)+j] = Normal(0, sqrt(ψ[i, dP*(l-1)+j] * Minnesota(l, i, j; q, ν0, Ω0)))
            end
        end
    end
    for i in (dQ+1):dP
        ϕ0[i, 1] = Normal(μϕ_const[i], sqrt(ψ0[i] * q[4, 2]))
        for l = 1:p
            for j in 1:dP
                if i == j && l == 1
                    ϕ0[i, 1+dP*(l-1)+j] = Normal(ρ[i-dQ], sqrt(ψ[i, dP*(l-1)+j] * Minnesota(l, i, j; q, ν0, Ω0)))
                else
                    ϕ0[i, 1+dP*(l-1)+j] = Normal(0, sqrt(ψ[i, dP*(l-1)+j] * Minnesota(l, i, j; q, ν0, Ω0)))
                end
            end
        end
    end

    return ϕ0
end

"""
    Minnesota(l, i, j; q, ν0, Ω0)
It return unscaled prior variance of the Minnesota prior.
# Input 
- lag `l`, dependent variable `i`, regressor `j` in the VAR(`p`)
- `q[:,1]` and `q[:,2]` are [own, cross, lag, intercept] shrikages for the first `dQ` and remaining `dP-dQ` equations, respectively.
- `ν0`(d.f.), `Ω0`(scale): Inverse-Wishart prior for the error-covariance matrix of VAR(`p`).
# Output
- Minnesota part in the prior variance
"""
function Minnesota(l, i, j; q, ν0, Ω0)

    dP = length(Ω0) # dimension

    if i < dimQ() + 1
        if i == j
            Minn_var = q[1, 1]
        else
            Minn_var = q[2, 1]
        end
        Minn_var /= l^q[3, 1]
        Minn_var *= ν0 - dP - 1
        Minn_var /= Ω0[j]
    else
        if i == j
            Minn_var = q[1, 2]
        else
            Minn_var = q[2, 2]
        end
        Minn_var /= l^q[3, 2]
        Minn_var *= ν0 - dP - 1
        Minn_var /= Ω0[j]
    end

    return Minn_var
end

"""
    prior_κQ(medium_τ, pr)
The function derive the maximizer decay parameter `κQ` that maximize the curvature factor loading at each candidate medium-term maturity. And then, it impose a discrete prior distribution on the maximizers with a prior probability vector `pr`.
# Input 
- `medium_τ::Vector`(candidate medium maturities, # of candidates)
- `pr::Vector`(probability, # of candidates)
# Output
- discrete prior distribution that has a support of the maximizers `κQ`
"""
function prior_κQ(medium_τ, pr) # Default candidates are one to five years

    medium_τN = length(medium_τ) # the number of candidates
    κQ_candidate = Vector{Float64}(undef, medium_τN) # support of the prior

    for i in eachindex(medium_τ) # calculate the maximizer κQ
        obj(κQ) = dcurvature_dτ(medium_τ[i]; κQ)
        κQ_candidate[i] = fzero(obj, 0.001, 2)
    end

    return DiscreteNonParametric(κQ_candidate, pr)

end

"""
    dcurvature_dτ(τ; κQ)
This function calculate the first derivative of the curvature factor loading w.r.t. the maturity.
# Input
- `κQ`: The decay parameter
- `τ`: The maturity that the derivative is calculated
# Output
- the first derivative of the curvature factor loading w.r.t. the maturity
"""
function dcurvature_dτ(τ; κQ)
    derivative = (κQ * τ + 1) * exp(-κQ * τ)
    derivative -= 1
    derivative /= κQ * (τ^2)
    derivative += exp(-κQ * τ) * κQ
    return derivative
end

"""
    prior_γ(yields, p)
There is a hierarchcal structure in the measurement equation. The prior means of the measurement errors are `γ[i]` and each `γ[i]` follows Gamma(1,`γ_bar`) distribution. This function decides `γ_bar` empirically. OLS is used to estimate the measurement equation and then a variance of residuals is calculated for each maturities. An inverse of the average residual variances is set to `γ_bar`.
# Output
- hyperparameter `γ_bar`
"""
function prior_γ(yields, p)
    yields = yields[p+1:end, :]

    PCs, OCs = PCA(yields, 0)
    T = size(OCs, 1)

    res_var = Vector{Float64}(undef, size(OCs, 2))
    for i in axes(OCs, 2)
        y = OCs[:, i]
        X = [ones(T) PCs]
        res_var[i] = var(y - X * ((X'X) \ (X'y)))
    end

    return 1 / mean(res_var)
end
