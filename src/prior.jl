#############################
## VAR transition equation ##
#############################
# error covariance of reduced form VAR, ΩFF ~ Inverse-Wishart(ν0, diagm[Ω0])
# Reduced form VAR coefficients ~ Normal
# However, we orthogonalize the reduced form VAR and estimate C, ϕ and σ²FF,
# where ΩFF = LDLt decompopsition, C\diagm(σ²FF)/C'
# ϕ = coefficients of the orthogonalized VAR
#############################
"""
prior_σ²FF(; ν0, Ω0::Vector)
* We translate the Inverse-Wishart prior to a series of the Normal-Inverse-Gamma (NIG) prior distributions. If the dimension is dₚ, there are dₚ NIG prior distributions. This function generates Inverse-Gamma priors.  
* Input: the InverseWishart prior
    - ν0: the degree of freedom
    - Ω0: the scale parameter
* Output: prior of σ²FF in the LDLt decomposition, ΩFF = inv(C)*diagm(σ²FF)*inv(C)'
    - Each element in the output follows Inverse-Gamma priors.
"""
function prior_σ²FF(; ν0, Ω0::Vector)

    dP = length(Ω0) # dimension
    σ²FF = Vector{Any}(undef, dP) # Diagonal matrix in the LDLt decomposition that follows a IG distribution

    for i in 1:dP
        σ²FF[i] = InverseGamma((ν0 + i - dP) / 2, Ω0[i] / 2)
    end

    return σ²FF
end

"""
prior_C(; Ω0::Vector)
* We translate the Inverse-Wishart prior to a series of the Normal-Inverse-Gamma (NIG) prior distributions. If the dimension is dₚ, there are dₚ NIG prior distributions. This function generates Normal priors.  
* Input: the scale parameter, Ω0
* Output: unscaled prior of C in the LDLt decomposition, ΩFF = inv(C)*diagm(σ²FF)*inv(C)'
* Important note!!
    - prior variance for C[i,:] = σ²FF[i]*variance of output[i,:]
    - Unlike MvNormal, the second arg of "Normal" is a standard deviation.
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
prior_ϕ0(ρ::Vector, prior_κQ_; ψ0, ψ, q, ν0, Ω0)
* This part derives the prior distribution for coefficients of the lagged regressors in the orthogonalized VAR. 
* Input:
    - ρ = Vector{Float64}(0 or near 1, dP-dQ) benchmark persistencies of macro variables. For growth variables and level variables, ρ[i] should be 0 and nearly 1, respectively.
    - ψ0: sparsity on the intercept 
    - ψ: sparsity on the slope coefficients
    - q: Minnesota shrikages with vector dimension 4 (own, cross, lag, intercept shrikages)
    - ν0(d.f.), Ω0(scale): hyperparameters for Inverse-Wishart prior on error-covariance in the reduced form VAR, ΩFF
* Output: Normal prior distributions on the slope coefficient of lagged variables and intercepts in the orthogonalized equation. 
* Important note!!
    - prior variance for ϕ[i,:] = σ²FF[i]*variance of output[i,:]
    - Unlike MvNormal, the second arg of "Normal" is a standard deviation.
"""
function prior_ϕ0(ρ::Vector, prior_κQ_; ψ0, ψ, q, ν0, Ω0)

    dP, dPp = size(ψ) # dimension & #{regressors}
    p = Int(dPp / dP) # the number of lags
    ϕ0 = Matrix{Any}(undef, dP, dPp + 1)
    dQ = dimQ() # reduced dimension of yields

    κQ_candidate = support(prior_κQ_)
    κQ_prob = probs(prior_κQ_)
    GQ_XX_mean = zeros(dQ, dQ)
    for i in eachindex(κQ_candidate)
        κQ = κQ_candidate[i]
        GQ_XX_mean += GQ_XX(; κQ) .* κQ_prob[i]
    end

    for i in 1:dQ
        ϕ0[i, 1] = Normal(0, sqrt(ψ0[i] * q[4]))
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
        ϕ0[i, 1] = Normal(0, sqrt(ψ0[i] * q[4]))
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
* It return unscaled prior variance of the Minnesota prior.
* Input: 
    - lag l, dependent variable i, regressor j in the VAR(p)
    - q = [own, cross, lag, intercept] shrikages
    - ν0(d.f.), Ω0(scale): Inverse-Wishart prior for the error-covariance matrix of VAR(p)
* Output: corresponding Minnesota variance
    - To scale it, the return has to be divided by the corresponding error variance.
"""
function Minnesota(l, i, j; q, ν0, Ω0)

    dP = length(Ω0) # dimension

    if i == j
        Minn_var = q[1]
    else
        Minn_var = q[2]
    end
    Minn_var /= l^q[3]
    Minn_var *= ν0 - dP - 1
    Minn_var /= Ω0[j]

    return Minn_var
end

###########################
## DNS decay parameter κQ##
###########################
"""
prior_κQ(medium_τ)
* The function derive the maximizer decay parameter κQ that maximize the curvature factor loading at each candidate medium-term maturity.
* Input: Vector{Float64}(candidate medium maturities, # of candidates)
* Output: uniform prior distribution that has a support of the maximizer κQ
"""
function prior_κQ(medium_τ) # Default candidates are one to five years

    N = length(medium_τ) # the number of candidates
    κQ_candidate = Vector{Float64}(undef, N) # support of the prior

    for i in 1:N # calculate the maximizer κQ
        obj(κQ) = 1200 * dcurvature_dτ(medium_τ[i]; κQ)
        κQ_candidate[i] = fzero(obj, 0.001, 0.2)
    end

    return DiscreteNonParametric(κQ_candidate, ones(N) / N)

end

"""
dcurvature_dτ(τ; κQ)
* This function calculate the first derivative of the curvature factor loading w.r.t. the maturity.
* Input:
    - κQ: The decay parameter
    - τ: The maturity that the derivative is calculated
* Output: the first derivative of the curvature factor loading w.r.t. the maturity
"""
function dcurvature_dτ(τ; κQ)
    derivative = (κQ * τ + 1) * exp(-κQ * τ)
    derivative -= 1
    derivative /= κQ * (τ^2)
    derivative += exp(-κQ * τ) * κQ
    return derivative
end

#################################
## Population Measurement error##
#################################
"""
prior_γ(yields)
* There is a hierarchcal structure in the measurement equation. The prior means of the measurement errors are γᵢ and each γᵢ follows Gamma(1,γ_bar) distribution. This function decides γ_bar empirically. OLS is used to estimate the measurement equation and then a variance of residuals is calculated for each maturities. An inverse of the average residual variances is set to γ_bar.
* Input: yield data in which each column shows time-series of a specific bond yield. Here, yields do not contain initial conditions. 
* Output: hyperparameter γ_bar
"""
function prior_γ(yields)
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
