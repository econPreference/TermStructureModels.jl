#######################
##Prior Distributions##
#######################
#This section contains for the prior distribution of the deep-parameters. Several hyperparameters comes into this section, and it formulate the specific prior distribution setting.

#    * Input: 
#        * hyperparameters that comes from hyperparameters.jl
#        * medium_τ: candidate maturities that represent medium-term bond yields

#    * Output: Prior distributions for
#        * ϕᵢ, ``\sigma_{\mathcal{FF},i}^{2}``: parameter that chracterize the transition equation. ϕᵢ is decomposed into C[i,:] and ϕ0ᵢ, that are contemporaneous and lagged slope coefficients, respectively.
#        * κQ: the DNS decay parameters
#        * γᵢ: the level of pricing errors

#############################
## VAR transition equation ##
#############################
"""
We translate the Inverse-Wishart prior to a series of the Normal-InverseGamma (NIG) prior distributions. If the dimension is dₚ, there are dₚ NIG prior distributions. This part shows the one that follow IG priors.  

    * Input:
        * ν0: the degree of freedom
        * Ω0: the scale parameter

    * Output: Diagonal Matrix in the LDLt decomposition that each diagonal follows IG
===
"""
function prior_σ²FF(; ν0, Ω0::Vector)

    dP, ~ = size(Ω0) # dimension
    σ²FF = Vector{Any}(undef, dP) # Diagonal matrix in the LDLt decomposition that follows a IG distribution

    for i in 1:dP
        σ²FF[i] = InverseGamma((ν0 + i - dP) / 2, Ω0[i] / 2)
    end

    return σ²FF
end

"""
We translate the Inverse-Wishart prior to a series of the Normal-InverseGamma (NIG) prior distributions. If the dimension is dₚ, there are dₚ NIG prior distributions. This part shows the one that follows Normals

    * Input:
        * Ω0: the scale parameter

    * Output: The LowerTriangular Part in the LDLt that each lower part follows a normal distribution. Note that the variance term is omitted because of the form of NIG.
===
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
This part derives the prior distribution for the lagged regressors in the transition equation. 

    * Input:
        * ρ: benchmark persistence of dependent variables. It should be different between growth variables and level variables
        * ψ0: sparsity on the P-intercept term
        * ψ: sparsity on the P-slope term
        * q: Minnesota shrikages with dimension 4 (own, cross, lag, intercept shrikages)
        * ν0(d.f.), Ω(scale): hyperparameters for prior on p-error covariance

    * Output: Normal prior distributions on the slope coefficient of lagged variables and intercepts in P-transition equation. Note that the variance term is omitted because of the form of NIG.
===
"""
function prior_ϕ0(ρ::Vector; ψ0, ψ, q, ν0, Ω0)

    dP, dPp = size(ψ) # dimension & #{regressors}
    p = Int(dPp / dP) # the number of lags
    ϕ0 = Matrix{Any}(undef, dP, dPp + 1)
    dQ = dimQ() # reduced dimension of yields

    prior_κQ_ = prior_κQ()
    κQ_candidate = support(prior_κQ_)
    κQ_prob = pdf(prior_κQ_)
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
It return a prior variance of the slope coefficients in the transition equation in a spirit of the Minneosta Prior. 

    * Input: 
        * lag l, dependent variable i, regressor j in the VAR(p)
        * q = [own, cross, lag, intercept] shrikages
        * ν0(d.f.), Ω0(scale): Inverse-Wishart prior for the error-covariance matrix of VAR(p)

    * Output: corresponding Minnesota variance
===
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
The function derive the maximizer decay parameter κQ that maximize the curvature factor loading at each candidate medium-term maturity.

    * Input:
        * medium_τ: several candidate bond maturities that represent medium term maturity bonds.

    * Output: uniform prior distribution that has a support of the maximizer κQ
===
"""
function prior_κQ(medium_τ=12 * [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]) # Default candidates are one to five years

    N = length(medium_τ) # the number of candidates
    κQ_candidate = Vector{Float64}(undef, N)

    for i in 1:N # calculate the maximizer κQ
        obj(κQ) = 1200 * dcurvature_dτ(medium_τ[i]; κQ)
        κQ_candidate[i] = fzero(obj, 0.001, 0.2)
    end

    return DiscreteNonParametric(κQ_candidate, ones(N) / N)

end

"""
This function calculate the first derivative of the curvature factor loading w.r.t. the maturity, ``\frac{d\text{curvature}}{d\tau} ``.

    * Input:
        * κQ: The decay parameter
        * τ: The maturity that the derivative is calculated

    * Output: the first derivative of the curvature factor loading w.r.t. the maturity
===
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
There is a hierarchcal structure in the Measurement equation. The prior means of the measurement errors are γᵢ and each γᵢ follows Gamma(1,γ_bar) distribution. This function decides γ_bar empirically. OLS is used to estimate the measurement equation and then the residuals are averaged across all maturities. An inverse of the average value is set to γ_bar.

    * Input: yield data in which each column shows time-series of a specific bond yield. Here, yields does not contain initial conditions. 

    * Output: hyperparameter γ_bar
===
"""
function prior_γ(yields)
    dQ = dimQ() # #(latent factors in the bond market)

    std_yields = yields .- mean(yields, dims=1)
    std_yields ./= std(yields, dims=1)
    V = reverse(eigen(cov(std_yields)).vectors, dims=2)
    Wₚ = V[:, 1:dQ]'
    Wₒ = V[:, (dQ+1):end]'

    PCs = (Wₚ * yields')'
    OCs = (Wₒ * yields')'

    return 1 / var(OCs - PCs * ((PCs'PCs) \ (PCs'OCs)))
end
