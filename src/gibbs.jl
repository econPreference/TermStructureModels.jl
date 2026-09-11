
"""
    post_kQ_infty(mean_kQ_infty, std_kQ_infty, yields, tau_n; kappaQ, phi, varFF, SigmaO, data_scale, pca_loadings)
# Output
- Full conditional posterior distribution
"""
function post_kQ_infty(mean_kQ_infty, std_kQ_infty, yields, tau_n; kappaQ, phi, varFF, SigmaO, data_scale, pca_loadings)

    dP = length(varFF)
    dQ = dimQ() + size(yields, 2) - length(tau_n)
    p = Int(((size(phi, 2) - 1) / dP) - 1)
    yields = yields[p+1:end, :]

    N = length(tau_n) # of maturities
    T = size(yields, 1) # length of dependent variables
    PCs, OCs, Wₚ, Wₒ, mean_PCs = PCA(yields, 0; pca_loadings)

    bτ_ = bτ(tau_n[end]; kappaQ, dQ)
    Bₓ_ = Bₓ(bτ_, tau_n)
    T1X_ = T1X(Bₓ_, Wₚ)
    Bₚ_ = Bₚ(Bₓ_, T1X_, Wₒ)
    ΩPP = phi_varFF_2_ΩPP(; phi, varFF, dQ)

    a0 = zeros(tau_n[end])
    a1 = zeros(tau_n[end])
    for τ in 2:tau_n[end]
        a0[τ] = a0[τ-1] - jensens_inequality(τ, bτ_, T1X_; ΩPP, data_scale)
        if length(kappaQ) > 1
            a1[τ] = a1[τ-1] + (1 - (kappaQ[1]^(τ - 1))) / (1 - kappaQ[1])
        else
            a1[τ] = a1[τ-1] + (τ - 1)
        end
    end
    A0_kQ_infty = a0[tau_n] ./ tau_n
    A1_kQ_infty = a1[tau_n] ./ tau_n

    # Dependent variable
    y = vec(OCs')
    y -= kron(ones(T), Wₒ * (I(N) - Bₓ_ / T1X_ * Wₚ) * A0_kQ_infty + Wₒ * Bₓ_ / T1X_ * mean_PCs)
    y -= vec(Bₚ_ * PCs')
    y ./= kron(ones(T), sqrt.(SigmaO))

    # regressor
    X = Wₒ * (I(N) - Bₓ_ / T1X_ * Wₚ) * A1_kQ_infty
    X ./= sqrt.(SigmaO)
    X = kron(ones(T), X)

    kQ_infty_var = inv(X'X + (1 / (std_kQ_infty^2)))
    return Normal(kQ_infty_var * ((mean_kQ_infty / (std_kQ_infty^2)) + X'y), sqrt(kQ_infty_var))

end

"""
    post_kappaQ(yields, prior_kappaQ_, tau_n; kQ_infty, phi, varFF, SigmaO, data_scale, pca_loadings)
# Input
- `prior_kappaQ_` is an output of function `prior_kappaQ`.
# Output
- Full conditional posterior distribution
"""
function post_kappaQ(yields, prior_kappaQ_, tau_n; kQ_infty, phi, varFF, SigmaO, data_scale, pca_loadings)

    kappaQ_candidate = support(prior_kappaQ_)

    kern = Vector{Float64}(undef, length(kappaQ_candidate)) # Posterior kernel

    for i in eachindex(kappaQ_candidate)
        # likelihood of the measurement eq
        kern[i] = loglik_mea(yields, tau_n; kappaQ=kappaQ_candidate[i], kQ_infty, phi, varFF, SigmaO, data_scale, pca_loadings) + logpdf(prior_kappaQ_, kappaQ_candidate[i])
    end

    kern .-= maximum(kern)
    Pr = exp.(kern)
    Pr ./= sum(Pr)

    return DiscreteNonParametric(kappaQ_candidate, Pr)
end

"""
    proposal_kappaQ2(yields, macros, mean_phi_const, rho, prior_kappaQ_, tau_n; kappaQ, kQ_infty, phi, varFF, SigmaO, psi, psi_const, q, nu0, Omega0, gamma_bar, mean_kQ_infty, std_kQ_infty, fix_const_PC1, data_scale, pca_loadings, proposal_NM_maxiter=0)
This function prepares the tailored independent MH proposal for `kappaQ` under the JSZ model. It uses LBFGS to find an initial joint mode and computes the corresponding Hessian after integrating out the VAR intercept and lag coefficients and `gamma`.
If LBFGS performs poorly, set `proposal_NM_maxiter` to a value greater than 0 to use Nelder-Mead to improve its starting point. Its value sets the maximum number of Nelder-Mead iterations.
# Output(2)
- `proposal_dist(kQ_infty, phi, varFF, SigmaO)`: conditional Student t proposal for `[kappaQ[1]; diff(kappaQ)]`.
- `param_mode::Parameter`: initial parameters based on the optimized mode.
"""
function proposal_kappaQ2(yields, macros, mean_phi_const, rho, prior_kappaQ_, tau_n; kappaQ, kQ_infty, phi, varFF, SigmaO, psi, psi_const, q, nu0, Omega0, gamma_bar, mean_kQ_infty, std_kQ_infty, fix_const_PC1, data_scale, pca_loadings, proposal_NM_maxiter=0)

    dQ, dP = length(kappaQ), length(varFF)
    p = Int(size(psi, 2) / dP)
    PCs, ~, Wₚ = PCA(yields, p; pca_loadings)
    yphi, Xphi = yphi_Xphi(PCs, macros, p)
    prior_phi_ = [prior_phi0(mean_phi_const, rho, prior_kappaQ_, tau_n, Wₚ; psi_const, psi, q, nu0, Omega0, fix_const_PC1) prior_C(; Omega0)]
    prior_varFF_ = prior_varFF(; nu0, Omega0)
    m = mean.(prior_phi_)
    V = var.(prior_phi_)
    mCQ, VCQ, post_varFFQ = [], [], []

    # Integrate out the intercept and lag coefficients in the transition equation.
    for i in 1:dQ
        Kphiᵢ = Kphi(i, V, Xphi, dP)
        phiᵢ_hat = phi_hat(i, m, V, yphi, Xphi, dP)
        Sᵢ_hat = S_hat(i, m, V, yphi, Xphi, dP; Omega0)
        push!(mCQ, phiᵢ_hat[(1+p*dP+1):end])
        push!(VCQ, Symmetric(inv(Kphiᵢ))[(1+p*dP+1):end, (1+p*dP+1):end])
        push!(post_varFFQ, InverseGamma(shape(prior_varFF_[i]) + size(yphi, 1) / 2, Sᵢ_hat))
    end

    idxCQ = [CartesianIndex(i, j) for i in 2:dQ for j in 1:(i-1)]
    idx_varFFQ = (dQ+2+length(idxCQ)):(2dQ+1+length(idxCQ))
    idx_SigmaO = (last(idx_varFFQ)+1):(last(idx_varFFQ)+length(SigmaO))
    other_params(kQ_infty_, phi_, varFF_, SigmaO_) = [kQ_infty_; phi_[1:dQ, (1+p*dP+1):(1+p*dP+dQ)][idxCQ]; log.(varFF_[1:dQ]); log.(SigmaO_)]

    # Joint marginal log posterior of kappaQ, kQ_infty, CQ, varFFQ and SigmaO in z coordinates, integrating out VAR intercepts, lag coefficients and gamma.
    function logpost(z)
        if !(z[1] < 1 && all(z[2:dQ] .< 0))
            return -Inf
        end
        kappaQ_ = cumsum(z[1:dQ])
        logprior = sum(logpdf.(prior_kappaQ_, kappaQ_))
        !isfinite(logprior) && return -Inf
        CQ = Matrix{eltype(z)}(I, dQ, dQ)
        CQ[idxCQ] = z[dQ+2:last(idx_varFFQ)-dQ]
        varFFQ = exp.(z[idx_varFFQ])
        SigmaO_ = exp.(z[idx_SigmaO])
        ΩPP = (CQ \ diagm(varFFQ)) / CQ'
        logpost_ = logprior + logpdf(Normal(mean_kQ_infty, std_kQ_infty), z[dQ+1])
        logpost_ += loglik_mea2(yields, tau_n, p; kappaQ=kappaQ_, kQ_infty=z[dQ+1], ΩPP, SigmaO=SigmaO_, data_scale, pca_loadings)
        for i in 1:dQ
            logpost_ += logpdf(post_varFFQ[i], varFFQ[i]) + z[idx_varFFQ[i]]
            if i > 1
                logpost_ += logpdf(MvNormal(mCQ[i], varFFQ[i] * VCQ[i]), CQ[i, 1:i-1])
            end
        end
        # Integrate out gamma; include the Jacobians for log variances.
        logpost_ += sum(log(2gamma_bar) .- 3log.(1 .+ gamma_bar .* SigmaO_) .+ z[idx_SigmaO])
        return logpost_
    end

    # Optimize ordered roots below 1.
    function transform(u)
        return [1 - exp(u[1]); -exp.(u[2:dQ]); u[dQ+1:end]]
    end
    u = [log(1 - kappaQ[1]); log.(-diff(kappaQ)); other_params(kQ_infty, phi, varFF, SigmaO)]
    println("Optimizing posterior mode...")
    flush(stdout)
    if proposal_NM_maxiter >= 1
        opt = optimize(u -> -logpost(transform(u)), u, NelderMead(), Optim.Options(iterations=proposal_NM_maxiter, show_trace=true))
        u = Optim.minimizer(opt)
    end
    opt = optimize(u -> -logpost(transform(u)), u, LBFGS(), Optim.Options(show_trace=true); autodiff=AutoForwardDiff())
    z_mode = transform(Optim.minimizer(opt))
    println("Computing proposal Hessian...")
    flush(stdout)
    objective = TwiceDifferentiable(z -> -logpost(z), z_mode; autodiff=AutoForwardDiff())
    inv_hess = Symmetric(inv(Optim.hessian!(objective, z_mode)))
    inv_V_other = inv(inv_hess[dQ+1:end, dQ+1:end])
    B = inv_hess[1:dQ, dQ+1:end] * inv_V_other
    V_cond = Matrix(Symmetric(inv_hess[1:dQ, 1:dQ] - B * inv_hess[dQ+1:end, 1:dQ]))

    # Conditional distribution of a joint Student t approximation (15 d.f.).
    function proposal_dist(kQ_infty_, phi_, varFF_, SigmaO_)
        delta = other_params(kQ_infty_, phi_, varFF_, SigmaO_) - z_mode[dQ+1:end]
        df = 15 + length(delta)
        scale = (15 + delta' * inv_V_other * delta) / df
        return MvTDist(df, z_mode[1:dQ] + B * delta, scale * V_cond)
    end

    phi_mode, varFF_mode = copy(phi), copy(varFF)
    for (i, idx) in enumerate(idxCQ)
        phi_mode[idx[1], 1+p*dP+idx[2]] = z_mode[dQ+1+i]
    end
    varFF_mode[1:dQ] = exp.(z_mode[idx_varFFQ])
    SigmaO_mode = exp.(z_mode[idx_SigmaO])
    param_mode = Parameter(kappaQ=cumsum(z_mode[1:dQ]), kQ_infty=z_mode[dQ+1], phi=phi_mode, varFF=varFF_mode, SigmaO=SigmaO_mode, gamma=mean.(post_gamma(; gamma_bar, SigmaO=SigmaO_mode)))
    return proposal_dist, param_mode
end

"""
    post_kappaQ2(yields, prior_kappaQ_, tau_n; kappaQ, kQ_infty, phi, varFF, SigmaO, data_scale, pca_loadings, proposal_dist)
This function conducts the tailored independent Metropolis-Hastings algorithm for the reparameterized `kappaQ` under the unrestricted JSZ form. The proposal conditions the initial joint Student t approximation on the current values of the other parameters.
- Reparameterization:
    kappaQ = cumsum(x)
    x = [kappaQ[1]; diff(kappaQ)]
- Jacobian: a lower triangular matrix of ones.
- The determinant = 1
"""
function post_kappaQ2(yields, prior_kappaQ_, tau_n; kappaQ, kQ_infty, phi, varFF, SigmaO, data_scale, pca_loadings, proposal_dist)

    function logpost(x)
        kappaQ_logpost = cumsum(x)
        logprior = 0.0
        for i in eachindex(prior_kappaQ_)
            logprior += logpdf(prior_kappaQ_[i], kappaQ_logpost[i])
        end
        logprior == -Inf && return -Inf
        loglik = loglik_mea(yields, tau_n; kappaQ=kappaQ_logpost, kQ_infty, phi, varFF, SigmaO, data_scale, pca_loadings)
        return loglik + logprior
    end

    # Independent MH step
    x = [kappaQ[1]; diff(kappaQ)]
    proposal_dist_ = proposal_dist(kQ_infty, phi, varFF, SigmaO)
    x_prop = rand(proposal_dist_)
    kappaQ_prop = cumsum(x_prop)
    if !(sort(kappaQ_prop, rev=true) == kappaQ_prop && kappaQ_prop[1] < 1.0)
        return kappaQ, false
    end
    logpost_prop = logpost(x_prop)
    logpost_prop == -Inf && return kappaQ, false
    log_MHPr = min(0.0, logpost_prop + logpdf(proposal_dist_, x) - logpost(x) - logpdf(proposal_dist_, x_prop))
    if log(rand()) < log_MHPr
        return kappaQ_prop, true
    else
        return kappaQ, false
    end
end

"""
    post_phi_varFF(yields, macros, mean_phi_const, rho, prior_kappaQ_, tau_n; phi, psi, psi_const, varFF, q, nu0, Omega0, kappaQ, kQ_infty, SigmaO, fix_const_PC1, data_scale, pca_loadings)
Full-conditional posterior sampler for `phi` and `varFF`
# Input
- `prior_kappaQ_` is an output of function `prior_kappaQ`.
- When `fix_const_PC1==true`, the first element in a constant term in the orthogonalized VAR is fixed to its prior mean during the posterior sampling.
# Output(3)
`phi`, `varFF`, `isaccept=Vector{Bool}(undef, dQ)`
- Returns a posterior sample.
"""
function post_phi_varFF(yields, macros, mean_phi_const, rho, prior_kappaQ_, tau_n; phi, psi, psi_const, varFF, q, nu0, Omega0, kappaQ, kQ_infty, SigmaO, fix_const_PC1, data_scale, pca_loadings)

    dQ = dimQ() + size(yields, 2) - length(tau_n)
    dP = size(psi, 1)
    p = Int(size(psi)[2] / dP)
    PCs, ~, Wₚ = PCA(yields, p; pca_loadings)

    yphi, Xphi = yphi_Xphi(PCs, macros, p)
    prior_phi0_ = prior_phi0(mean_phi_const, rho, prior_kappaQ_, tau_n, Wₚ; psi_const, psi, q, nu0, Omega0, fix_const_PC1)
    prior_phi_ = [prior_phi0_ prior_C(; Omega0)]
    prior_varFF_ = prior_varFF(; nu0, Omega0)

    isaccept = fill(false, dQ)
    for i in 1:dP
        if i <= dQ
            prop_phi = copy(phi) # proposal for C
            prop_varFF = copy(varFF) # proposal for varFF

            mᵢ = mean.(prior_phi_[i, 1:(1+p*dP+i-1)])
            Vᵢ = var.(prior_phi_[i, 1:(1+p*dP+i-1)])
            prop_phi[i, 1:(1+p*dP+i-1)], prop_varFF[i] = NIG_NIG(yphi[:, i], Xphi[:, 1:(end-dP+i-1)], mᵢ, diagm(Vᵢ), shape(prior_varFF_[i]), scale(prior_varFF_[i]))

            prob = loglik_mea(yields, tau_n; kappaQ, kQ_infty, phi=prop_phi, varFF=prop_varFF, SigmaO, data_scale, pca_loadings)
            prob -= loglik_mea(yields, tau_n; kappaQ, kQ_infty, phi, varFF, SigmaO, data_scale, pca_loadings)

            if rand() < min(1.0, exp(prob))
                phi = copy(prop_phi)
                varFF = copy(prop_varFF)
                isaccept[i] = true
            end
        else
            mᵢ = mean.(prior_phi_[i, 1:(1+p*dP+i-1)])
            Vᵢ = var.(prior_phi_[i, 1:(1+p*dP+i-1)])
            phi[i, 1:(1+p*dP+i-1)], varFF[i] = NIG_NIG(yphi[:, i], Xphi[:, 1:(end-dP+i-1)], mᵢ, diagm(Vᵢ), shape(prior_varFF_[i]), scale(prior_varFF_[i]))
        end
    end

    return phi, varFF, isaccept

end

"""
    post_kappaQ_phi_varFF_q_nu0(yields, macros, tau_n, mean_phi_const, rho, prior_q, prior_nu0, prior_diff_kappaQ; phi, psi, psi_const, varFF, q, nu0, kappaQ, kQ_infty, SigmaO, fix_const_PC1, data_scale, pca_loadings, sampler, chain, is_warmup)
Full-conditional posterior sampler for `kappaQ`, `phi` and `varFF`
# Input
- `prior_q`: The 4 by 2 matrix that contains the prior distribution for q. All entries should be objects in `Distributions.jl`.
- `prior_nu0`: The prior distribution for nu0 - (dP + 1). It should be an object in `Distributions.jl`.
- `prior_diff_kappaQ` is a vector of the truncated normals(`Distributions.truncated(Distributions.Normal(), lower, upper)`). It has a prior for `[kappaQ[1]; diff(kappaQ)]`.
- When `fix_const_PC1==true`, the first element in a constant term in the orthogonalized VAR is fixed to its prior mean during the posterior sampling.
- `sampler` and `chain` are the objects in `Turing.jl`.
- If the current step is in the warmup phase, set `is_warmup=true`.
# Output(6)
chain, q, nu0, kappaQ, phi, varFF
"""
function post_kappaQ_phi_varFF_q_nu0(yields, macros, tau_n, mean_phi_const, rho, prior_q, prior_nu0, prior_diff_kappaQ; phi, psi, psi_const, varFF, q, nu0, kappaQ, kQ_infty, SigmaO, fix_const_PC1, data_scale, pca_loadings, sampler, chain, is_warmup)

    dQ = dimQ() + size(yields, 2) - length(tau_n)
    dP = size(psi, 1)
    p = Int(size(psi)[2] / dP)
    PCs, ~, Wₚ = PCA(yields, p; pca_loadings)
    dims_phi = [1 + p * dP + i - 1 for i in 1:dQ] |> cumsum
    net_nu0 = nu0 - (dP + 1)

    if isempty(macros)
        factors = copy(PCs)
    else
        factors = [PCs macros]
    end

    Omega0 = Vector{Float64}(undef, dP)
    for i in eachindex(Omega0)
        Omega0[i] = (AR_res_var(factors[:, i], p)[1]) * net_nu0
    end

    yphi, Xphi = yphi_Xphi(PCs, macros, p)
    prior_kappaQ_ = mean.([prior_diff_kappaQ[i].untruncated for i in eachindex(prior_diff_kappaQ)]) |> cumsum |> x -> [Dirac(x[i]) for i in eachindex(x)]
    prior_phi0_ = prior_phi0(mean_phi_const, rho, prior_kappaQ_, tau_n, Wₚ; psi_const, psi, q, nu0, Omega0, fix_const_PC1)
    prior_phi_ = [prior_phi0_ prior_C(; Omega0)]
    prior_varFF_ = prior_varFF(; nu0, Omega0)
    GQ_XX_mean = prior_kappaQ_ |> x -> mean.(x) |> diagm
    updated_q_idx = findall(x -> !(x isa Dirac), prior_q)

    if isempty(chain)
        chain = Vector{MCMCChains.Chains}(undef, length(sampler))
    end
    for j in eachindex(sampler)
        if j == 1
            NUTS_model_ = diff_kappaQ_NUTS_model(yields, PCs, tau_n, macros, p, dims_phi, prior_diff_kappaQ; kQ_infty, phi, varFF, SigmaO, data_scale, pca_loadings)
            initial_params = (diff_kappaQ=[kappaQ[1]; diff(kappaQ)],)
        elseif j <= dQ + 1
            NUTS_model_ = VAR_NUTS_model(j - 1, yields, PCs, tau_n, macros, dP, p, dims_phi, prior_phi_, prior_varFF_; kappaQ, kQ_infty, phi, varFF, SigmaO, data_scale, pca_loadings)
            initial_params = (phiQ=phi[j-1, 1:(1+p*dP+j-2)], varFFQ=varFF[j-1])
        else
            for i in (dQ+1):dP
                mᵢ = mean.(prior_phi_[i, 1:(1+p*dP+i-1)])
                Vᵢ = var.(prior_phi_[i, 1:(1+p*dP+i-1)])
                phi[i, 1:(1+p*dP+i-1)], varFF[i] = NIG_NIG(yphi[:, i], Xphi[:, 1:(end-dP+i-1)], mᵢ, diagm(Vᵢ), shape(prior_varFF_[i]), scale(prior_varFF_[i]))
            end
            NUTS_model_ = q_nu0_NUTS_model(factors, prior_q, prior_nu0, p, dQ, dP, GQ_XX_mean, rho; phi0=phi[:, 1:end-dP], C=phi[:, end-dP+1:end], varFF, psi_const, psi, mean_phi_const, fix_const_PC1)
            initial_params = (updated_q=q[updated_q_idx], net_nu0=net_nu0)
        end

        if !isassigned(chain, j)
            chain[j] = Turing.sample(NUTS_model_, sampler[j], 2; initial_params=Turing.InitFromParams(initial_params), chain_type=MCMCChains.Chains, save_state=true, progress=false)
        else
            state = Turing.Inference.loadstate(chain[j])
            state = Turing.Inference.gibbs_update_state!!(sampler[j], state, NUTS_model_, Turing.Inference.gibbs_get_parameter_values(state))
            chain[j] = Turing.AbstractMCMC.mcmcsample(
                Random.default_rng(),
                NUTS_model_,
                sampler[j],
                1;
                chain_type=MCMCChains.Chains,
                initial_state=state,
                progress=false,#Turing.PROGRESS[],
                nadapts=is_warmup ? state.i + 1 : 0,
                discard_adapt=false,
                discard_initial=0,
                save_state=true,
                verbose=false
            )
        end

        if j == 1
            kappaQ = group(chain[j], :diff_kappaQ).value |> x -> x[end, :, 1] |> cumsum
        elseif j <= dQ + 1
            phi[j-1, 1:(1+p*dP+j-2)] = group(chain[j], :phiQ).value |> x -> x[end, :, 1]
            varFF[j-1] = group(chain[j], :varFFQ).value[end, 1, 1]
        else
            q[updated_q_idx] = group(chain[j], :updated_q).value |> x -> x[end, :, 1]
            nu0 = group(chain[j], :net_nu0).value[end, 1, 1] + (dP + 1)
        end
    end

    return chain, q, nu0, kappaQ, phi, varFF

end

"""
    function diff_kappaQ_NUTS_model(yields, PCs, tau_n, macros, p, dims_phi, prior_diff_kappaQ_; kQ_infty, phi, varFF, SigmaO, data_scale, pca_loadings)
This function creates a model for `diff_kappaQ` in the syntax of `Turing.jl`.
"""

@model function diff_kappaQ_NUTS_model(yields, PCs, tau_n, macros, p, dims_phi, prior_diff_kappaQ_; kQ_infty, phi, varFF, SigmaO, data_scale, pca_loadings)

    diff_kappaQ ~ product_distribution(prior_diff_kappaQ_)
    phiQ = zeros(1)
    varFFQ = zeros(1)

    log_lik = loglik_NUTS([], yields, PCs, tau_n, macros, dims_phi, p; phiQ, varFFQ, diff_kappaQ, kQ_infty, phi, varFF, SigmaO, data_scale, pca_loadings)
    Turing.@addlogprob! log_lik

    return diff_kappaQ
end

"""
    function VAR_NUTS_model(i, yields, PCs, tau_n, macros, dP, p, dims_phi, prior_phi_, prior_varFF_; kappaQ, kQ_infty, phi, varFF, SigmaO, data_scale, pca_loadings)
This function creates a model for `phi` and `varFF` in the syntax of `Turing.jl`.
"""

@model function VAR_NUTS_model(i, yields, PCs, tau_n, macros, dP, p, dims_phi, prior_phi_, prior_varFF_; kappaQ, kQ_infty, phi, varFF, SigmaO, data_scale, pca_loadings)

    mᵢ = mean.(prior_phi_[i, 1:(1+p*dP+i-1)])
    Vᵢ = var.(prior_phi_[i, 1:(1+p*dP+i-1)])

    varFFQ ~ prior_varFF_[i]
    phiQ ~ MvNormal(mᵢ, varFFQ * diagm(Vᵢ))

    log_lik = loglik_NUTS(i, yields, PCs, tau_n, macros, dims_phi, p; phiQ, varFFQ, diff_kappaQ=[kappaQ[1]; diff(kappaQ)], kQ_infty, phi, varFF, SigmaO, data_scale, pca_loadings)
    Turing.@addlogprob! log_lik

    return phiQ, varFFQ
end

"""
    function q_nu0_NUTS_model(factors, prior_q, prior_nu0, p, dQ, dP, GQ_XX_mean, rho; phi0, C, varFF, psi_const, psi, mean_phi_const, fix_const_PC1)
This function creates a model for `q` and `nu0` in the syntax of `Turing.jl`.
"""

@model function q_nu0_NUTS_model(factors, prior_q, prior_nu0, p, dQ, dP, GQ_XX_mean, rho; phi0, C, varFF, psi_const, psi, mean_phi_const, fix_const_PC1)

    updated_q_idx = findall(x -> !(x isa Dirac), prior_q)
    updated_q ~ product_distribution(prior_q[updated_q_idx])
    fix_q_idx = findall(x -> x isa Dirac, prior_q)
    fix_q = mean.(prior_q[fix_q_idx])

    q = Matrix{promote_type(eltype(fix_q), eltype(updated_q))}(undef, 5, 2)
    q[updated_q_idx] = updated_q
    q[fix_q_idx] = fix_q

    net_nu0 ~ prior_nu0

    Omega0 = Vector{promote_type(Float64, eltype(net_nu0))}(undef, dP)
    for i in eachindex(Omega0)
        Omega0[i] = (AR_res_var(factors[:, i], p)[1]) * net_nu0
    end

    Turing.@addlogprob! logprior_varFF(varFF; nu0=net_nu0 + (dP + 1), Omega0)
    Turing.@addlogprob! logprior_C(C; varFF, Omega0)
    Turing.@addlogprob! logprior_phi0(phi0, mean_phi_const, rho, GQ_XX_mean, p, dQ, dP; varFF, psi_const, psi, q, nu0=net_nu0 + (dP + 1), Omega0, fix_const_PC1)

    return updated_q, net_nu0
end

"""
    loglik_NUTS(i, yields, PCs, tau_n, macros, dims_phi, p; phiQ, varFFQ, diff_kappaQ, kQ_infty, phi, varFF, SigmaO, data_scale, pca_loadings)
This function calculates the likelihood of the NUTS block.
"""
function loglik_NUTS(i, yields, PCs, tau_n, macros, dims_phi, p; phiQ, varFFQ, diff_kappaQ, kQ_infty, phi, varFF, SigmaO, data_scale, pca_loadings)

    phi_full = similar(phi, promote_type(eltype(phi), eltype(phiQ))) |> x -> x .= 0.0
    varFF_full = similar(varFF, promote_type(eltype(varFF), eltype(varFFQ)))

    phi_full[:, :] = phi |> deepcopy
    varFF_full[:] = varFF |> deepcopy

    yphi, Xphi = yphi_Xphi(PCs, macros, p)

    if i == 1
        phi_full[i, 1:dims_phi[i]] = phiQ
        varFF_full[i] = varFFQ
    elseif !isempty(i)
        phi_full[i, 1:diff(dims_phi)[i-1]] = phiQ
        varFF_full[i] = varFFQ
    end

    T = size(yphi, 1)
    log_pdf = 0.0
    if !isempty(i)
        log_pdf += logpdf(MvNormal(Xphi * phi_full[i, :], varFFQ * I(T)), yphi[:, i])
    end
    log_pdf += loglik_mea_NUTS(yields, tau_n; kappaQ=cumsum(diff_kappaQ), kQ_infty, phi=phi_full, varFF=varFF_full, SigmaO, data_scale, pca_loadings)

    return log_pdf
end

"""
    NIG_NIG(y, X, β₀, B₀, α₀, δ₀)
Normal-InverseGamma-Normal-InverseGamma update
- prior: `β|σ² ~ MvNormal(β₀,σ²B₀)`, `σ² ~ InverseGamma(α₀,δ₀)`
- likelihood: `y|β,σ² = Xβ + MvNormal(zeros(T,1),σ²I(T))`
# Output(2)
`β`, `σ²`
- posterior sample
"""
function NIG_NIG(y, X, β₀, B₀, α₀, δ₀)

    T = length(y)

    inv_B₀ = inv(B₀)
    inv_B₁ = inv_B₀ + X'X
    B₁ = Symmetric(inv(inv_B₁))
    β₁ = B₁ * (inv_B₀ * β₀ + X'y)
    δ₁ = δ₀ + 0.5 * (y'y + β₀' * inv_B₀ * β₀ - β₁' * inv_B₁ * β₁)

    σ² = rand(InverseGamma(α₀ + 0.5T, δ₁))
    β = rand(MvNormal(β₁, σ² * B₁))

    return β, σ²
end

"""
    post_SigmaO(yields, tau_n; kappaQ, kQ_infty, ΩPP, gamma, p, data_scale, pca_loadings)
Posterior sampler for the measurement errors
# Output
- `Vector{Dist}(IG, N-dQ)`
"""
function post_SigmaO(yields, tau_n; kappaQ, kQ_infty, ΩPP, gamma, p, data_scale, pca_loadings)
    yields = yields[p+1:end, :]

    dQ = dimQ() + size(yields, 2) - length(tau_n)
    N = length(tau_n)
    T = size(yields, 1)
    PCs, OCs, Wₚ, Wₒ, mean_PCs = PCA(yields, 0; pca_loadings)

    bτ_ = bτ(tau_n[end]; kappaQ, dQ)
    Bₓ_ = Bₓ(bτ_, tau_n)
    T1X_ = T1X(Bₓ_, Wₚ)
    Bₚ_ = Bₚ(Bₓ_, T1X_, Wₒ)

    aτ_ = aτ(tau_n[end], bτ_, tau_n, Wₚ; kQ_infty, ΩPP, data_scale)
    Aₓ_ = Aₓ(aτ_, tau_n)
    T0P_ = T0P(T1X_, Aₓ_, Wₚ, mean_PCs)
    Aₚ_ = Aₚ(Aₓ_, Bₓ_, T0P_, Wₒ)

    post_SigmaO_ = Vector{Any}(undef, N - dQ)
    for i in 1:N-dQ
        residuals = OCs[:, i] - (Aₚ_[i] .+ (Bₚ_[i, :]' * PCs')')
        post_SigmaO_[i] = InverseGamma(2 + 0.5T, gamma[i] + 0.5residuals'residuals)
    end

    return post_SigmaO_
end

"""
    post_gamma(; gamma_bar, SigmaO)
Posterior sampler for the population measurement error
# Output
- `Vector{Dist}(Gamma,length(SigmaO))`
"""
function post_gamma(; gamma_bar, SigmaO)

    N = length(SigmaO) # of measurement errors

    post_gamma_ = Vector{Any}(undef, N)
    for i in 1:N
        post_gamma_[i] = Gamma(3, 1 / (gamma_bar + (1 / SigmaO[i])))
    end

    return post_gamma_
end