
"""
    tuning_hyperparameter(yields, macros, tau_n, rho; populationsize=50, maxiter=10_000, medium_tau=collect(24:3:48), upper_q=[1 1; 1 1; 10 10; 100 100], mean_kQ_infty=0, std_kQ_infty=0.1, upper_nu0=[], mean_phi_const=[], fix_const_PC1=false, upper_p=18, mean_phi_const_PC1=[], data_scale=1200, kappaQ_prior_pr=[], init_nu0=[], is_pure_EH=false, psi_common=[], psi_const=[])
It optimizes our hyperparameters by maximizing the marginal likelhood of the transition equation. Our optimizer is a differential evolutionary algorithm that utilizes bimodal movements in the eigen-space(Wang, Li, Huang, and Li, 2014) and the trivial geography(Spector and Klein, 2006).
# Input
- When we compare marginal likelihoods between models, the data for the dependent variable should be the same across the models. To achieve that, we set a period of dependent variable based on `upper_p`. For example, if `upper_p = 3`, `yields[4:end,:]` and `macros[4:end,:]` are the data for our dependent variable. `yields[1:3,:]` and `macros[1:3,:]` are used for setting initial observations for all lags.
- `populationsize` and `maxiter` are options for the optimizer.
    - `populationsize`: the number of candidate solutions in each generation
    - `maxtier`: the maximum number of iterations
- The lower bounds for `q` and `nu0` are `0` and `dP+2`. 
- The upper bounds for `q`, `nu0` and VAR lag can be set by `upper_q`, `upper_nu0`, `upper_p`.
    - Our default option for `upper_nu0` is the time-series length of the data.
- If you use our default option for `mean_phi_const`,
    1. `mean_phi_const[dQ+1:end]` is a zero vector.
    2. `mean_phi_const[1:dQ]` is calibrated to make a prior mean of `λₚ` a zero vector.
    3. After step 2, `mean_phi_const[1]` is replaced with `mean_phi_const_PC1` if it is not empty.
- `mean_phi_const = Matrix(your prior, dP, upper_p)` 
- `mean_phi_const[:,i]` is a prior mean for the VAR(`i`) constant. Therefore `mean_phi_const` is a matrix only in this function. In other functions, `mean_phi_const` is a vector for the orthogonalized VAR system with your selected lag.
- When `fix_const_PC1==true`, the first element in a constant term in our orthogonalized VAR is fixed to its prior mean during the posterior sampling.
- `data_scale::scalar`: In typical affine term structure model, theoretical yields are in decimal and not annualized. But, for convenience(public data usually contains annualized percentage yields) and numerical stability, we sometimes want to scale up yields, so want to use (`data_scale`*theoretical yields) as variable `yields`. In this case, you can use `data_scale` option. For example, we can set `data_scale = 1200` and use annualized percentage monthly yields as `yields`.
- `is_pure_EH::Bool`: When `mean_phi_const=[]`, `is_pure_EH=false` sets `mean_phi_const` to zero vectors. Otherwise, `mean_phi_const` is set to imply the pure expectation hypothesis under `mean_phi_const=[]`.
- `psi_const` and `psi = kron(ones(1, lag length), psi_common)` are multiplied with prior variances of coefficients of the intercept and lagged regressors in the orthogonalized transition equation. They are used for imposing zero prior variances. A empty default value means that you do not use this function. `[psi_const psi][i,j]` is corresponds to `phi[i,j]`. The entries of `psi_common` and `psi_const` should be nearly zero(e.g., `1e-10`), not exactly zero.
# Output(2)
optimized Hyperparameter, optimization result
- Be careful that we minimized the negative log marginal likelihood, so the second output is about the minimization problem.
"""
function tuning_hyperparameter(yields, macros, tau_n, rho; populationsize=50, maxiter=10_000, medium_tau=collect(24:3:48), upper_q=[1 1; 1 1; 10 10; 100 100], mean_kQ_infty=0, std_kQ_infty=0.1, upper_nu0=[], mean_phi_const=[], fix_const_PC1=false, upper_p=18, mean_phi_const_PC1=[], data_scale=1200, kappaQ_prior_pr=[], init_nu0=[], is_pure_EH=false, psi_common=[], psi_const=[])


    if isempty(upper_nu0) == true
        upper_nu0 = size(yields, 1)
    end

    dQ = dimQ() + size(yields, 2) - length(tau_n)
    if isempty(macros)
        dP = copy(dQ)
    else
        dP = dQ + size(macros, 2)
    end
    if isempty(kappaQ_prior_pr)
        kappaQ_prior_pr = length(medium_tau) |> x -> ones(x) / x
    end

    if isempty(psi_common)
        psi_common = ones(dP, dP)
    end
    if isempty(psi_const)
        psi_const = ones(dP)
    end

    lx = [0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 1; 1]
    ux = 0.0 .+ [vec(upper_q); upper_nu0 - (dP + 1); upper_p]
    if isempty(mean_phi_const) && is_pure_EH
        mean_phi_const = Matrix{Float64}(undef, dP, upper_p)
        for i in axes(mean_phi_const, 2)
            mean_phi_const_PCs = -calibrate_mean_phi_const(mean_kQ_infty, std_kQ_infty, init_nu0, yields[upper_p-i+1:end, :], macros[upper_p-i+1:end, :], tau_n, i; medium_tau, iteration=10_000, data_scale, kappaQ_prior_pr)[1] |> x -> mean(x, dims=1)[1, :]
            if !isempty(mean_phi_const_PC1)
                mean_phi_const_PCs = [mean_phi_const_PC1, mean_phi_const_PCs[2], mean_phi_const_PCs[3]]
            end
            if isempty(macros)
                mean_phi_const[:, i] = copy(mean_phi_const_PCs)
            else
                mean_phi_const[:, i] = [mean_phi_const_PCs; zeros(size(macros, 2))]
            end
            prior_const_TP = calibrate_mean_phi_const(mean_kQ_infty, std_kQ_infty, init_nu0, yields[upper_p-i+1:end, :], macros[upper_p-i+1:end, :], tau_n, i; medium_tau, mean_phi_const_PCs, iteration=10_000, data_scale, kappaQ_prior_pr, τ=120)[2]
            println("For lag $i, mean_phi_const[1:dQ] is $mean_phi_const_PCs ,")
            println("and prior mean of the constant part in the term premium is $(mean(prior_const_TP)),")
            println("and prior std of the constant part in the term premium is $(std(prior_const_TP)).")
            println(" ")
        end
    elseif isempty(mean_phi_const) && !is_pure_EH
        mean_phi_const = zeros(dP, upper_p)
    end
    starting = (lx + ux) ./ 2
    starting[end] = 1

    function negative_log_marginal(input)

        # parameters
        q = [input[1] input[5]
            input[2] input[6]
            input[3] input[7]
            input[4] input[8]]
        nu0 = input[9] + dP + 1
        p = Int(input[10])

        PCs, ~, Wₚ = PCA(yields[(upper_p-p)+1:end, :], p)
        if isempty(macros)
            factors = copy(PCs)
        else
            factors = [PCs macros[(upper_p-p)+1:end, :]]
        end
        Omega0 = Vector{Float64}(undef, dP)
        for i in eachindex(Omega0)
            Omega0[i] = (AR_res_var(factors[:, i], p)[1]) * input[9]
        end

        tuned = Hyperparameter(p=copy(p), q=copy(q), nu0=copy(nu0), Omega0=copy(Omega0), mean_phi_const=copy(mean_phi_const[:, p]))
        if isempty(macros)
            psi = kron(ones(1, p), psi_common)
            return -log_marginal(factors, macros, rho, tuned, tau_n, Wₚ; medium_tau, kappaQ_prior_pr, fix_const_PC1, psi, psi_const)
        else
            psi = kron(ones(1, p), psi_common)
            return -log_marginal(factors[:, 1:dQ], factors[:, dQ+1:end], rho, tuned, tau_n, Wₚ; medium_tau, kappaQ_prior_pr, fix_const_PC1, psi, psi_const)
        end

        # Although the input data should contains initial observations, the argument of the marginal likelihood should be the same across the candidate models. Therefore, we should align the length of the dependent variable across the models.

    end

    ss = MixedPrecisionRectSearchSpace(lx, ux, [-1ones(Int64, 9); 0])
    opt = bboptimize(negative_log_marginal, starting; SearchSpace=ss, MaxSteps=maxiter, PopulationSize=populationsize, CallbackInterval=10, CallbackFunction=x -> println("Current Best: p = $(Int(best_candidate(x)[10])), q[:,1] = $(best_candidate(x)[1:4]), q[:,2] = $(best_candidate(x)[5:8]), nu0 = $(best_candidate(x)[9] + dP + 1)"))

    q = [best_candidate(opt)[1] best_candidate(opt)[5]
        best_candidate(opt)[2] best_candidate(opt)[6]
        best_candidate(opt)[3] best_candidate(opt)[7]
        best_candidate(opt)[4] best_candidate(opt)[8]]
    nu0 = best_candidate(opt)[9] + dP + 1
    p = best_candidate(opt)[10] |> Int

    PCs = PCA(yields[(upper_p-p)+1:end, :], p)[1]
    if isempty(macros)
        factors = copy(PCs)
    else
        factors = [PCs macros[(upper_p-p)+1:end, :]]
    end
    Omega0 = Vector{Float64}(undef, dP)
    for i in eachindex(Omega0)
        Omega0[i] = (AR_res_var(factors[:, i], p)[1]) * best_candidate(opt)[9]
    end

    return Hyperparameter(p=copy(p), q=copy(q), nu0=copy(nu0), Omega0=copy(Omega0), mean_phi_const=copy(mean_phi_const[:, p])), opt

end

"""
    AR_res_var(TS::Vector, p)
It derives an MLE error variance estimate of an AR(`p`) model
# Input
- univariate time series `TS` and the lag `p`
# output(2)
residual variance estimate, AR(p) coefficients
"""
function AR_res_var(TS::Vector, p)
    Y = TS[(p+1):end]
    T = length(Y)
    X = ones(T)
    for i in 1:p
        X = hcat(X, TS[p+1-i:end-i])
    end

    β = (X'X) \ (X'Y)
    return var(Y - X * β), β
end

"""
    posterior_sampler(yields, macros, tau_n, rho, iteration, tuned::Hyperparameter; medium_tau=collect(24:3:48), init_param=[], psi=[], psi_const=[], gamma_bar=[], kappaQ_prior_pr=[], mean_kQ_infty=0, std_kQ_infty=0.1, fix_const_PC1=false, data_scale=1200)
This is a posterior distribution sampler.
# Input
- `iteration`: # of posterior samples
- `tuned`: optimized hyperparameters used during estimation
- `init_param`: starting point of the sampler. It should be a type of Parameter.
- `psi_const` and `psi` are multiplied with prior variances of coefficients of the intercept and lagged regressors in the orthogonalized transition equation. They are used for imposing zero prior variances. A empty default value means that you do not use this function. `[psi_const psi][i,j]` is corresponds to `phi[i,j]`. The entries of `psi` and `psi_const` should be nearly zero(e.g., `1e-10`), not exactly zero.
# Output(2)
`Vector{Parameter}(posterior, iteration)`, acceptance rate of the MH algorithm
"""
function posterior_sampler(yields, macros, tau_n, rho, iteration, tuned::Hyperparameter; medium_tau=collect(24:3:48), init_param=[], psi=[], psi_const=[], gamma_bar=[], kappaQ_prior_pr=[], mean_kQ_infty=0, std_kQ_infty=0.1, fix_const_PC1=false, data_scale=1200)

    p, q, nu0, Omega0, mean_phi_const = tuned.p, tuned.q, tuned.nu0, tuned.Omega0, tuned.mean_phi_const
    N = size(yields, 2) # of maturities
    dQ = dimQ() + size(yields, 2) - length(tau_n)
    if isempty(macros)
        dP = copy(dQ)
    else
        dP = dQ + size(macros, 2)
    end
    if isempty(kappaQ_prior_pr)
        kappaQ_prior_pr = length(medium_tau) |> x -> ones(x) / x
    end
    Wₚ = PCA(yields, p)[3]
    prior_kappaQ_ = prior_kappaQ(medium_tau, kappaQ_prior_pr)
    if isempty(gamma_bar)
        gamma_bar = prior_gamma(yields, p)[1]
    end

    if typeof(init_param) == Parameter
        kappaQ, kQ_infty, phi, varFF, SigmaO, gamma = init_param.kappaQ, init_param.kQ_infty, init_param.phi, init_param.varFF, init_param.SigmaO, init_param.gamma
    else
        ## initial parameters ##
        if typeof(kappaQ_prior_pr[1]) <: Real
            kappaQ = 0.0609
        else
            kappaQ = [0.99, 0.95, 0.9]
        end
        kQ_infty = 0.0
        phi = [zeros(dP) diagm(Float64.([0.9ones(dQ); rho])) zeros(dP, dP * (p - 1)) zeros(dP, dP)] # The last dP by dP block matrix in phi should always be a lower triangular matrix whose diagonals are also always zero.
        bτ_ = bτ(tau_n[end]; kappaQ, dQ)
        Bₓ_ = Bₓ(bτ_, tau_n)
        T1X_ = T1X(Bₓ_, Wₚ)
        phi[1:dQ, 2:(dQ+1)] = T1X_ * GQ_XX(; kappaQ) / T1X_
        varFF = [Omega0[i] / (nu0 + i - dP) for i in eachindex(Omega0)]
        SigmaO = 1 ./ fill(gamma_bar, N - dQ)
        gamma = 1 ./ fill(gamma_bar, N - dQ)
        ########################
    end
    if isempty(psi)
        psi = ones(dP, dP * p)
    end
    if isempty(psi_const)
        psi_const = ones(dP)
    end
    if !(typeof(kappaQ_prior_pr[1]) <: Real)

        ΩPP = mle_error_covariance(yields, [], tau_n, p)
        function logpost(x)
            kappaQ_logpost = cumsum(x[1:dQ])
            kQ_infty_logpost = x[dQ+1]
            SigmaO_logpost = x[dQ+1+1:dQ+1+length(tau_n)-dQ] |> x -> exp.(x)
            if maximum(abs.(kappaQ_logpost)) > 1 || !(sort(kappaQ_logpost, rev=true) == kappaQ_logpost) || !isposdef(diagm(SigmaO_logpost)) || !(minimum(kappaQ_logpost .∈ support.(prior_kappaQ_))) || !isempty(findall(abs.(diff(kappaQ_logpost)) .<= 0.001))
                return -Inf
            end

            logprior = 0.0
            for i in eachindex(prior_kappaQ_)
                logprior += logpdf(prior_kappaQ_[i], kappaQ_logpost[i])
            end
            return logprior + loglik_mea2(yields, tau_n, p; kappaQ=kappaQ_logpost, kQ_infty=kQ_infty_logpost, ΩPP, SigmaO=SigmaO_logpost, data_scale)
        end

        # Construct the proposal distribution
        #kappaQ = 0.2rand(3) .+ 0.8 |> x -> sort(x, rev=true)
        x = [kappaQ[1]; diff(kappaQ[1:end])]
        init = [x; kQ_infty; log.(SigmaO)]
        minimizers = optimize(x -> -logpost(x), [0; -1 * ones(length(kappaQ) - 1); -Inf; fill(-Inf, length(tau_n) - dQ)], [1; 0 * ones(length(kappaQ) - 1); Inf; fill(Inf, length(tau_n) - dQ)], init, ParticleSwarm(), Optim.Options(show_trace=true)) |>
                     Optim.minimizer |>
                     y -> optimize(x -> -logpost(x), [0; -1 * ones(length(kappaQ) - 1); -Inf; fill(-Inf, length(tau_n) - dQ)], [1; 0 * ones(length(kappaQ) - 1); Inf; fill(Inf, length(tau_n) - dQ)], y, Fminbox(LBFGS(; alphaguess=LineSearches.InitialPrevious())), Optim.Options(show_trace=true)) |>
                          Optim.minimizer

        x_mode = minimizers[1:dQ]
        x_hess = hessian(x -> -logpost(x), minimizers) |> x -> x[1:dQ, 1:dQ]
        inv_x_hess = inv(x_hess) |> x -> 0.5 * (x + x')
        if !isposdef(inv_x_hess)
            C, V = eigen(inv_x_hess)
            C = max.(eps(), C) |> diagm
            inv_x_hess = V * C / V |> x -> 0.5 * (x + x')
        end

    end

    isaccept_MH = zeros(dQ + 1)
    saved_params = Vector{Parameter}(undef, iteration)
    @showprogress 5 "posterior_sampler..." for iter in 1:iteration

        if typeof(kappaQ_prior_pr[1]) <: Real
            kappaQ = rand(post_kappaQ(yields, prior_kappaQ_, tau_n; kQ_infty, phi, varFF, SigmaO, data_scale))
        else
            kappaQ, isaccept = post_kappaQ2(yields, prior_kappaQ_, tau_n; kappaQ, kQ_infty, phi, varFF, SigmaO, data_scale, x_mode, inv_x_hess)
            isaccept_MH[end] += isaccept
        end

        kQ_infty = rand(post_kQ_infty(mean_kQ_infty, std_kQ_infty, yields, tau_n; kappaQ, phi, varFF, SigmaO, data_scale))

        phi, varFF, isaccept = post_phi_varFF(yields, macros, mean_phi_const, rho, prior_kappaQ_, tau_n; phi, psi, psi_const, varFF, q, nu0, Omega0, kappaQ, kQ_infty, SigmaO, fix_const_PC1, data_scale)
        isaccept_MH[1:dQ] += isaccept

        SigmaO = rand.(post_SigmaO(yields, tau_n; kappaQ, kQ_infty, ΩPP=phi_varFF_2_ΩPP(; phi, varFF, dQ), gamma, p, data_scale))

        gamma = rand.(post_gamma(; gamma_bar, SigmaO))

        saved_params[iter] = Parameter(kappaQ=copy(kappaQ), kQ_infty=copy(kQ_infty), phi=copy(phi), varFF=copy(varFF), SigmaO=copy(SigmaO), gamma=copy(gamma))

    end

    return saved_params, 100isaccept_MH / iteration
end

"""
    generative(T, dP, tau_n, p, noise::Float64; kappaQ, kQ_infty, KPXF, GPXFXF, OmegaXFXF, data_scale=1200)
This function generate a simulation data given parameters. Note that all parameters are the things in the latent factor state space (that is, parameters in struct LatentSpace). There is some differences in notations because it is hard to express mathcal letters in VScode. So, mathcal{F} in my paper is expressed in `F` in the VScode. And, "F" in my paper is expressed as `XF`.
# Input: 
- noise = variance of the measurement errors
# Output(3)
`yields`, `latents`, `macros`
- `yields = Matrix{Float64}(obs,T,length(tau_n))`
- `latents = Matrix{Float64}(obs,T,dimQ())`
- `macros = Matrix{Float64}(obs,T,dP - dimQ())`
"""
function generative(T, dP, tau_n, p, noise::Float64; kappaQ, kQ_infty, KPXF, GPXFXF, OmegaXFXF, data_scale=1200)
    N = length(tau_n) # of observed maturities
    dQ = dimQ() # of latent factors

    # Generating factors XF, where latents & macros ∈ XF
    XF = randn(p, dP)
    for horizon = 1:(round(Int, 1.5T))
        regressors = vec(XF[1:p, :]')
        samples = KPXF + GPXFXF * regressors + rand(MvNormal(zeros(dP), OmegaXFXF))
        XF = vcat(samples', XF)
    end
    XF = reverse(XF, dims=1)
    XF = XF[end-T+1:end, :]

    # Generating yields
    bτ_ = bτ(tau_n[end]; kappaQ, dQ)
    Bₓ_ = Bₓ(bτ_, tau_n)

    ΩXX = OmegaXFXF[1:dQ, 1:dQ]
    aτ_ = aτ(tau_n[end], bτ_; kQ_infty, ΩXX, data_scale)
    Aₓ_ = Aₓ(aτ_, tau_n)

    yields = Matrix{Float64}(undef, T, N)
    for t = 1:T
        yields[t, :] = (Aₓ_ + Bₓ_ * XF[t, 1:dQ])' + rand(Normal(0, sqrt(noise)), N)'
    end

    return yields, XF[:, 1:dQ], XF[:, (dQ+1):end]
end

"""
    ineff_factor(saved_params)
It returns inefficiency factors of each parameter.
# Input
- `Vector{Parameter}` from `posterior_sampler`
# Output
- Estimated inefficiency factors are in Tuple(`kappaQ`, `kQ_infty`, `gamma`, `SigmaO`, `varFF`, `phi`). For example, if you want to load an inefficiency factor of `phi`, you can use `Output.phi`.
- If `fix_const_PC1==true` in your optimized struct Hyperparameter, `Output.phi[1,1]` can be weird. So you should ignore it.
"""
function ineff_factor(saved_params)

    iteration = length(saved_params)

    init_kappaQ = saved_params[:kappaQ][1]
    init_kQ_infty = saved_params[:kQ_infty][1]
    init_phi = saved_params[:phi][1] |> vec
    init_varFF = saved_params[:varFF][1]
    init_SigmaO = saved_params[:SigmaO][1]
    init_gamma = saved_params[:gamma][1]

    initial_θ = [init_kappaQ; init_kQ_infty; init_gamma; init_SigmaO; init_varFF; init_phi]
    vec_saved_params = Matrix{Float64}(undef, iteration, length(initial_θ))
    vec_saved_params[1, :] = initial_θ
    prog = Progress(iteration - 1; dt=5, desc="ineff_factor...")
    Threads.@threads for iter in 2:iteration
        kappaQ = saved_params[:kappaQ][iter]
        kQ_infty = saved_params[:kQ_infty][iter]
        phi = saved_params[:phi][iter] |> vec
        varFF = saved_params[:varFF][iter]
        SigmaO = saved_params[:SigmaO][iter]
        gamma = saved_params[:gamma][iter]

        vec_saved_params[iter, :] = [kappaQ; kQ_infty; gamma; SigmaO; varFF; phi]
        next!(prog)
    end
    finish!(prog)

    ineff = Vector{Float64}(undef, size(vec_saved_params)[2])
    prog = Progress(size(vec_saved_params, 2); dt=5, desc="ineff_factor...")
    Threads.@threads for i in axes(vec_saved_params, 2)
        ineff[i] = longvar(vec_saved_params[:, i]) / var(vec_saved_params[:, i])
        next!(prog)
    end
    finish!(prog)

    phi_ineff = ineff[length(init_kappaQ)+1+length(init_gamma)+length(init_SigmaO)+length(init_varFF)+1:end] |> x -> reshape(x, size(saved_params[:phi][1], 1), size(saved_params[:phi][1], 2))
    dP = size(phi_ineff, 1)
    for i in 1:dP, j in i:dP
        phi_ineff[i, end-dP+j] = 0
    end

    if length(init_kappaQ) == 1
        kappaQ_ineff = ineff[1]
    else
        kappaQ_ineff = ineff[1:length(init_kappaQ)]
    end
    return (;
        kappaQ=kappaQ_ineff,
        kQ_infty=ineff[length(init_kappaQ)+1],
        gamma=ineff[length(init_kappaQ)+1+1:length(init_kappaQ)+1+length(init_gamma)],
        SigmaO=ineff[length(init_kappaQ)+1+length(init_gamma)+1:length(init_kappaQ)+1+length(init_gamma)+length(init_SigmaO)],
        varFF=ineff[length(init_kappaQ)+1+length(init_gamma)+length(init_SigmaO)+1:length(init_kappaQ)+1+length(init_gamma)+length(init_SigmaO)+length(init_varFF)],
        phi=copy(phi_ineff)
    )
end

"""
    longvar(v)
It calculates the long-run variance of `v` using the quadratic spectral window with selection of bandwidth of Andrews(1991). We use the AR(1) approximation.
# Input
- Time-series Vector `v`
# Output
- Estimated 2*π*h(0) of `v`, where h(x) is the spectral density of `v` at x.
"""
function longvar(v)

    v .-= mean(v)
    T = size(v)[1]

    gamma = zeros(T)
    for j = 0:T-1
        gamma[j+1] = (1 / T) * v[j+1:T]'v[1:T-j]
    end

    vh = v[2:T]
    vl = v[1:T-1]
    r = (vl'vh) / (vl'vl)
    rho = copy(r)
    e = vh - vl * r
    sig = (e'e) / T

    numerator = 4 * (rho^2) * (sig^2) / ((1 - rho)^8)
    denominator = (sig^2) / ((1 - rho)^4)

    alpha = numerator / denominator
    m = 1.3221 * (alpha * T)^(1 / 5)

    ## Applying QS window

    S = gamma[1]
    for ind = 1:T-1
        d = 6 * pi * (ind / m) / 5
        w = 3 * (sin(d) / d - cos(d)) / (d^2)
        S = S + w * gamma[ind+1]
    end
    for ind = 1:T-1
        d = 6 * pi * (-ind / m) / 5
        w = 3 * (sin(d) / d - cos(d)) / (d^2)
        S = S + w * gamma[ind+1]
    end

    return S * (T / (T - 1))

end

"""
    mle_error_covariance(yields, macros, tau_n, p)
It calculates the MLE estimates of the error covariance matrix of the VAR(p) model.
"""
function mle_error_covariance(yields, macros, tau_n, p)

    ## Extracting PCs
    PCs = PCA(yields, p)[1]
    N = length(tau_n)
    dQ = dimQ() + size(yields, 2) - N
    if isempty(macros)
        dP = copy(dQ)
        factors = [PCs yields[:, end-(dQ-dimQ()-1):end]]
    else
        dP = dQ + size(macros, 2)
        factors = [PCs yields[:, end-(dQ-dimQ()-1):end] macros]
    end

    ## VAR(p) estimation
    Y = factors[(p+1):end, :]
    X = ones(size(Y, 1))
    for i in 1:p
        X = hcat(X, factors[(p-i+1):(end-i), :])
    end
    PHI = (X'X) \ (X'Y)
    res = Y - X * PHI
    Omega = res'res / size(Y, 1)

    # ## Transform to the recursive VAR
    # phi = Matrix{Float64}(undef, dP, 1 + dP * (p + 1))
    # phi[:, 1] = PHI'[:, 1]
    # for i in 1:p
    #     phi[:, 1+dP*(i-1)+1:1+dP*i] = PHI'[:, 1+dP*(i-1)+1:1+dP*i]
    # end
    # phi[:, end-dP+1:end] = I(dP)
    # L, D = LDL(Omega)
    # varFF = diag(D)
    # phi = L \ phi
    # phi[:, end-dP+1:end] -= I(dP)
    # phi_vec = vec(phi[:, 1:end-dP])
    # for i in 1:dP
    #     phi_vec = vcat(phi_vec, phi[i+1:end, end-dP+i])
    # end

    return Omega
end
