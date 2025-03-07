"""
    prior_varFF(; nu0, Omega0::Vector)
We translate the Inverse-Wishart prior to a series of the Normal-Inverse-Gamma (NIG) prior distributions. If the dimension is dₚ, there are dₚ NIG prior distributions. This function generates Inverse-Gamma priors.  
# Output: 
- prior of `varFF` in the LDLt decomposition,` OmegaFF = inv(C)*diagm(varFF)*inv(C)'`
- Each element in the output follows Inverse-Gamma priors.
"""
function prior_varFF(; nu0, Omega0::Vector)

    dP = length(Omega0) # dimension
    varFF = Vector{Any}(undef, dP) # Diagonal matrix in the LDLt decomposition that follows a IG distribution

    for i in eachindex(varFF)
        varFF[i] = InverseGamma((nu0 + i - dP) / 2, Omega0[i] / 2)
    end

    return varFF
end

"""
    prior_C(; Omega0::Vector)
We translate the Inverse-Wishart prior to a series of the Normal-Inverse-Gamma (NIG) prior distributions. If the dimension is dₚ, there are dₚ NIG prior distributions. This function generates Normal priors.  
# Output: 
- unscaled prior of `C` in the LDLt decomposition, `OmegaFF = inv(C)*diagm(varFF)*inv(C)'`
# Important note
prior variance for `C[i,:] = varFF[i]*variance of output[i,:]`
"""
function prior_C(; Omega0::Vector)

    dP = length(Omega0) # dimension
    C = Matrix{Any}(undef, dP, dP) # A lower triangular Matrix in the LDLt that follows a Normal

    C[1, 1] = Dirac(1)
    for j in 2:dP
        C[1, j] = Dirac(0)
    end
    for i in 2:dP
        for j in 1:(i-1)
            C[i, j] = Normal(0, sqrt(1 / Omega0[j]))
        end
        C[i, i] = Dirac(1)
        for j in (i+1):dP
            C[i, j] = Dirac(0)
        end
    end

    return C
end

"""
    prior_phi0(mean_phi_const, rho::Vector, prior_kappaQ_, tau_n, Wₚ; psi_const, psi, q, nu0, Omega0, fix_const_PC1)
This part derives the prior distribution for coefficients of the lagged regressors in the orthogonalized VAR. 
# Input 
- `prior_kappaQ_` is a output of function `prior_kappaQ`.
- When `fix_const_PC1==true`, the first element in a constant term in our orthogonalized VAR is fixed to its prior mean during the posterior sampling.
# Output
- Normal prior distributions on the slope coefficient of lagged variables and intercepts in the orthogonalized equation. 
- `Output[:,1]` for intercepts, `Output[:,1+1:1+dP]` for the first lag, `Output[:,1+dP+1:1+2*dP]` for the second lag, and so on.
# Important note
prior variance for `phi[i,:]` = `varFF[i]*var(output[i,:])`
"""
function prior_phi0(mean_phi_const, rho::Vector, prior_kappaQ_, tau_n, Wₚ; psi_const, psi, q, nu0, Omega0, fix_const_PC1)

    dP, dPp = size(psi) # dimension & #{regressors}
    p = Int(dPp / dP) # the number of lags
    phi0 = Matrix{Any}(undef, dP, dPp + 1)

    if length(prior_kappaQ_) == 1
        dQ = dimQ() # reduced dimension of yields
        kappaQ_candidate = support(prior_kappaQ_)
        kappaQ_prob = probs(prior_kappaQ_)
        GQ_XX_mean = zeros(dQ, dQ)

        for i in eachindex(kappaQ_candidate)
            kappaQ = kappaQ_candidate[i]
            bτ_ = bτ(tau_n[end]; kappaQ, dQ)
            Bₓ_ = Bₓ(bτ_, tau_n)
            T1X_ = T1X(Bₓ_, Wₚ)
            GQ_XX_mean += (T1X_ * GQ_XX(; kappaQ) / T1X_) .* kappaQ_prob[i]
        end
    else
        dQ = length(prior_kappaQ_)
        GQ_XX_mean = prior_kappaQ_ |> x -> mean.(x) |> diagm
    end

    for i in 1:dQ
        if i == 1 && fix_const_PC1
            phi0[i, 1] = Normal(mean_phi_const[i], sqrt(psi_const[i] * 1e-10))
        else
            phi0[i, 1] = Normal(mean_phi_const[i], sqrt(psi_const[i] * q[4, 1]))
        end
        for l = 1:1
            for j in 1:dQ
                phi0[i, 1+dP*(l-1)+j] = Normal(GQ_XX_mean[i, j], sqrt(psi[i, dP*(l-1)+j] * Minnesota(l, i, j; q, nu0, Omega0, dQ)))
            end
            for j in (dQ+1):dP
                phi0[i, 1+dP*(l-1)+j] = Normal(0, sqrt(psi[i, dP*(l-1)+j] * Minnesota(l, i, j; q, nu0, Omega0, dQ)))
            end
        end
        for l = 2:p
            for j in 1:dP
                phi0[i, 1+dP*(l-1)+j] = Normal(0, sqrt(psi[i, dP*(l-1)+j] * Minnesota(l, i, j; q, nu0, Omega0, dQ)))
            end
        end
    end
    for i in (dQ+1):dP
        phi0[i, 1] = Normal(mean_phi_const[i], sqrt(psi_const[i] * q[4, 2]))
        for l = 1:p
            for j in 1:dP
                if i == j && l == 1
                    phi0[i, 1+dP*(l-1)+j] = Normal(rho[i-dQ], sqrt(psi[i, dP*(l-1)+j] * Minnesota(l, i, j; q, nu0, Omega0, dQ)))
                else
                    phi0[i, 1+dP*(l-1)+j] = Normal(0, sqrt(psi[i, dP*(l-1)+j] * Minnesota(l, i, j; q, nu0, Omega0, dQ)))
                end
            end
        end
    end

    return phi0
end

"""
    Minnesota(l, i, j; q, nu0, Omega0, dQ=[])
It return unscaled prior variance of the Minnesota prior.
# Input 
- lag `l`, dependent variable `i`, regressor `j` in the VAR(`p`)
- `q[:,1]` and `q[:,2]` are [own, cross, lag, intercept] shrikages for the first `dQ` and remaining `dP-dQ` equations, respectively.
- `nu0`(d.f.), `Omega0`(scale): Inverse-Wishart prior for the error-covariance matrix of VAR(`p`).
# Output
- Minnesota part in the prior variance
"""
function Minnesota(l, i, j; q, nu0, Omega0, dQ=[])

    dP = length(Omega0) # dimension
    if isempty(dQ)
        dQ = dimQ() # reduced dimension of yields
    end

    if i < dQ + 1
        if i == j
            Minn_var = q[1, 1]
        else
            Minn_var = q[2, 1]
        end
        Minn_var /= l^q[3, 1]
        Minn_var *= nu0 - dP - 1
        Minn_var /= Omega0[j]
    else
        if i == j
            Minn_var = q[1, 2]
        else
            Minn_var = q[2, 2]
        end
        Minn_var /= l^q[3, 2]
        Minn_var *= nu0 - dP - 1
        Minn_var /= Omega0[j]
    end

    return Minn_var
end

"""
    prior_kappaQ(medium_tau, pr)
The function derive the maximizer decay parameter `kappaQ` that maximize the curvature factor loading at each candidate medium-term maturity. And then, it impose a discrete prior distribution on the maximizers with a prior probability vector `pr`.
# Input 
- `medium_tau::Vector`(candidate medium maturities, # of candidates)
- `pr::Vector`(probability, # of candidates)
# Output
- discrete prior distribution that has a support of the maximizers `kappaQ`
"""
function prior_kappaQ(medium_tau, pr) # Default candidates are one to five years

    if typeof(pr[1]) <: Real
        medium_tauN = length(medium_tau) # the number of candidates
        kappaQ_candidate = Vector{Float64}(undef, medium_tauN) # support of the prior

        for i in eachindex(medium_tau) # calculate the maximizer kappaQ
            obj(kappaQ) = dcurvature_dτ(medium_tau[i]; kappaQ)
            kappaQ_candidate[i] = fzero(obj, 0.001, 2)
        end

        return DiscreteNonParametric(kappaQ_candidate, pr)
    else
        return pr
    end

end

"""
    dcurvature_dτ(τ; kappaQ)
This function calculate the first derivative of the curvature factor loading w.r.t. the maturity.
# Input
- `kappaQ`: The decay parameter
- `τ`: The maturity that the derivative is calculated
# Output
- the first derivative of the curvature factor loading w.r.t. the maturity
"""
function dcurvature_dτ(τ; kappaQ)
    derivative = (kappaQ * τ + 1) * exp(-kappaQ * τ)
    derivative -= 1
    derivative /= kappaQ * (τ^2)
    derivative += exp(-kappaQ * τ) * kappaQ
    return derivative
end

"""
    prior_gamma(yields, p)
There is a hierarchcal structure in the measurement equation. The prior means of the measurement errors are `gamma[i]` and each `gamma[i]` follows Gamma(1,`gamma_bar`) distribution. This function decides `gamma_bar` empirically. OLS is used to estimate the measurement equation and then a variance of residuals is calculated for each maturities. An inverse of the average residual variances is set to `gamma_bar`.
# Output
- hyperparameter `gamma_bar`
"""
function prior_gamma(yields, p)
    yields = yields[p+1:end, :]

    PCs, OCs = PCA(yields, 0)
    T = size(OCs, 1)

    res_var = Vector{Float64}(undef, size(OCs, 2))
    for i in axes(OCs, 2)
        y = OCs[:, i]
        X = [ones(T) PCs]
        res_var[i] = var(y - X * ((X'X) \ (X'y)))
    end

    return 1 / mean(res_var), res_var
end
