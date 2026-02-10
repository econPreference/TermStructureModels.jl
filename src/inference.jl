
"""
    tuning_hyperparameter(yields, macros, tau_n, rho; populationsize=50, maxiter=10_000, medium_tau=collect(24:3:48), upper_q=[1 1; 1 1; 1 1; 4 4; 100 100], mean_kQ_infty=0, std_kQ_infty=0.1, upper_nu0=[], mean_phi_const=[], fix_const_PC1=false, upper_p=24, mean_phi_const_PC1=[], data_scale=1200, kappaQ_prior_pr=[], init_nu0=[], is_pure_EH=false, psi_common=[], psi_const=[], pca_loadings=[], prior_mean_diff_kappaQ=[], prior_std_diff_kappaQ=[], optimizer=:LBFGS, ml_tol=1.0, init_x=[])
This function optimizes the hyperparameters by maximizing the marginal likelihood of the transition equation.
# Input
- When comparing marginal likelihoods between models, the data for the dependent variable should be the same across models. To achieve this, we set the period of the dependent variable based on `upper_p`. For example, if `upper_p = 3`, `yields[4:end,:]` and `macros[4:end,:]` are the data for the dependent variable. `yields[1:3,:]` and `macros[1:3,:]` are used for setting initial observations for all lags.
- `optimizer`: The optimization algorithm to use.
    - `:LBFGS` (default): Uses unconstrained LBFGS from `Optim.jl` with hybrid parameter transformations (exp for non-negativity, sigmoid for bounded parameters). Alternates between optimizing hyperparameters (with fixed lag) and selecting the best lag (with fixed hyperparameters) until convergence.
    - `:BBO`: Uses a differential evolutionary algorithm (BlackBoxOptim.jl). The lag and hyperparameters are optimized simultaneously.
- `ml_tol`: Tolerance for parsimony in lag selection (only for `:LBFGS`). After finding the lag with the best marginal likelihood, the algorithm iteratively selects smaller lags if their marginal likelihood is within `ml_tol` of the best. This favors simpler models (smaller lags) when performance is comparable.
- `init_x`: Initial values for hyperparameters and lag (only for `:LBFGS`). Should be a vector of length 12 in the format `[vec(q); nu0-(dP+1); p]`. If empty (default), uses `[0.1, 0.1, 0.1, 2.0, 1.0, 0.1, 0.1, 0.1, 2.0, 1.0, 1.0, 1]`.
- `populationsize` and `maxiter` are options for the optimizer.
    - `populationsize`: the number of candidate solutions in each generation (only for `:BBO`)
    - `maxiter`: the maximum number of iterations
- The lower bounds for `q` and `nu0` are `0` and `dP+2`.
- The upper bounds for `q`, `nu0`, and VAR lag can be set by `upper_q`, `upper_nu0`, and `upper_p`.
    - The default option for `upper_nu0` is the time-series length of the data.
- If you use the default option for `mean_phi_const`,
    1. `mean_phi_const[dQ+1:end]` is a zero vector.
    2. `mean_phi_const[1:dQ]` is calibrated to make the prior mean of `λₚ` a zero vector.
    3. After step 2, `mean_phi_const[1]` is replaced with `mean_phi_const_PC1` if it is not empty.
- `mean_phi_const = Matrix(your prior, dP, upper_p)`
- `mean_phi_const[:,i]` is the prior mean for the VAR(`i`) constant. Therefore, `mean_phi_const` is a matrix only in this function. In other functions, `mean_phi_const` is a vector for the orthogonalized VAR system with the selected lag.
- When `fix_const_PC1==true`, the first element in the constant term in the orthogonalized VAR is fixed to its prior mean during posterior sampling.
- `data_scale::scalar`: In a typical affine term structure model, theoretical yields are in decimals and not annualized. However, for convenience (public data usually contains annualized percentage yields) and numerical stability, we sometimes want to scale up yields and use (`data_scale`*theoretical yields) as the variable `yields`. In this case, you can use the `data_scale` option. For example, we can set `data_scale = 1200` and use annualized percentage monthly yields as `yields`.
- `kappaQ_prior_pr` is a vector of prior distributions for `kappaQ` under the JSZ model: each element specifies the prior for `kappaQ[i]` and must be provided as a `Distributions.jl` object. Alternatively, you can supply `prior_mean_diff_kappaQ` and `prior_std_diff_kappaQ`, which define means and standard deviations for Normal priors on `[kappaQ[1]; diff(kappaQ)]`; the implied Normal prior for each `kappaQ[i]` is then truncated to (0, 1). These options are only needed when using the JSZ model.
- `is_pure_EH::Bool`: When `mean_phi_const=[]`, `is_pure_EH=false` sets `mean_phi_const` to zero vectors. Otherwise, `mean_phi_const` is set to imply the pure expectation hypothesis under `mean_phi_const=[]`.
- `psi_const` and `psi = kron(ones(1, lag length), psi_common)` are multiplied with prior variances of coefficients of the intercept and lagged regressors in the orthogonalized transition equation. They are used for imposing zero prior variances. An empty default value means that you do not use this function. `[psi_const psi][i,j]` corresponds to `phi[i,j]`. The entries of `psi_common` and `psi_const` should be nearly zero (e.g., `1e-10`), not exactly zero.
- `pca_loadings=Matrix{, dQ, size(yields, 2)}` stores the loadings for the first dQ principal components (so `principal_components = yields * pca_loadings'`), and you may optionally provide these loadings externally; if omitted, the package computes them internally via PCA.
# Output(2)
Optimized hyperparameter, optimization result
- Note that we minimize the negative log marginal likelihood, so the second output is for the minimization problem.
- When `optimizer=:LBFGS`, the second output is a NamedTuple with fields `minimizer`, `minimum`, `p`, `all_minimizer`, `all_minimum`.
"""
function tuning_hyperparameter(yields, macros, tau_n, rho; populationsize=50, maxiter=10_000, medium_tau=collect(24:3:48), upper_q=[1 1; 1 1; 1 1; 4 4; 100 100], mean_kQ_infty=0, std_kQ_infty=0.1, upper_nu0=[], mean_phi_const=[], fix_const_PC1=false, upper_p=24, mean_phi_const_PC1=[], data_scale=1200, kappaQ_prior_pr=[], init_nu0=[], is_pure_EH=false, psi_common=[], psi_const=[], pca_loadings=[], prior_mean_diff_kappaQ=[], prior_std_diff_kappaQ=[], optimizer=:LBFGS, ml_tol=1.0, init_x=[])


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
        if isempty(prior_mean_diff_kappaQ)
            kappaQ_prior_pr = length(medium_tau) |> x -> ones(x) / x
        else
            kappaQ_prior_pr = [truncated(Normal(prior_mean_diff_kappaQ[1], prior_std_diff_kappaQ[1]), eps(), 1 - eps())]
            for i in 2:length(prior_mean_diff_kappaQ)
                kappaQ_prior_pr = [kappaQ_prior_pr; truncated(convolve(Normal(prior_mean_diff_kappaQ[i], prior_std_diff_kappaQ[i]), deepcopy(kappaQ_prior_pr[i-1].untruncated)), eps(), 1 - eps())]
            end
        end
    end

    if isempty(psi_common)
        psi_common = ones(dP, dP)
    end
    if isempty(psi_const)
        psi_const = ones(dP)
    end

    lx = [0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 1; 1]
    ux = 0.0 .+ [vec(upper_q); upper_nu0 - (dP + 1); upper_p]
    if isempty(mean_phi_const) && is_pure_EH
        mean_phi_const = Matrix{Float64}(undef, dP, upper_p)
        for i in axes(mean_phi_const, 2)
            mean_phi_const_PCs = -calibrate_mean_phi_const(mean_kQ_infty, std_kQ_infty, init_nu0, yields[upper_p-i+1:end, :], macros[upper_p-i+1:end, :], tau_n, i; medium_tau, iteration=10_000, data_scale, kappaQ_prior_pr, pca_loadings)[1] |> x -> mean(x, dims=1)[1, :]
            if !isempty(mean_phi_const_PC1)
                mean_phi_const_PCs = [mean_phi_const_PC1, mean_phi_const_PCs[2], mean_phi_const_PCs[3]]
            end
            if isempty(macros)
                mean_phi_const[:, i] = copy(mean_phi_const_PCs)
            else
                mean_phi_const[:, i] = [mean_phi_const_PCs; zeros(size(macros, 2))]
            end
            prior_const_TP = calibrate_mean_phi_const(mean_kQ_infty, std_kQ_infty, init_nu0, yields[upper_p-i+1:end, :], macros[upper_p-i+1:end, :], tau_n, i; medium_tau, mean_phi_const_PCs, iteration=10_000, data_scale, kappaQ_prior_pr, τ=120, pca_loadings)[2]
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
        q = [input[1] input[6]
            input[2] input[7]
            input[3] input[8]
            input[4] input[9]
            input[5] input[10]]
        nu0 = input[11] + dP + 1
        p = Int(input[12])

        PCs, ~, Wₚ = PCA(yields[(upper_p-p)+1:end, :], p; pca_loadings)
        if isempty(macros)
            factors = copy(PCs)
        else
            factors = [PCs macros[(upper_p-p)+1:end, :]]
        end
        Omega0 = Vector{Float64}(undef, dP)
        for i in eachindex(Omega0)
            Omega0[i] = (AR_res_var(factors[:, i], p)[1]) * input[11]
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

    if optimizer == :BBO
        ss = MixedPrecisionRectSearchSpace(lx, ux, [-1ones(Int64, 11); 0])
        opt = bboptimize(negative_log_marginal, starting; SearchSpace=ss, MaxSteps=maxiter, PopulationSize=populationsize, CallbackInterval=10, CallbackFunction=x -> println("Current Best: p = $(Int(best_candidate(x)[12])), q[:,1] = $(best_candidate(x)[1:5]), q[:,2] = $(best_candidate(x)[6:10]), nu0 = $(best_candidate(x)[11] + dP + 1)"))

        q = [best_candidate(opt)[1] best_candidate(opt)[6]
            best_candidate(opt)[2] best_candidate(opt)[7]
            best_candidate(opt)[3] best_candidate(opt)[8]
            best_candidate(opt)[4] best_candidate(opt)[9]
            best_candidate(opt)[5] best_candidate(opt)[10]]
        nu0 = best_candidate(opt)[11] + dP + 1
        p = best_candidate(opt)[12] |> Int

    elseif optimizer == :LBFGS
        # Alternating optimization between hyperparameters and lag selection
        # all_x[p] stores optimized hyperparameters for lag p
        # all_fitness[p] stores the objective value for that optimization
        all_x = [fill(NaN, 11) for _ in 1:upper_p]
        all_fitness = fill(NaN, upper_p)

        # Set initial values: [vec(q); nu0; p]
        if isempty(init_x)
            init_hyperparameters = [0.1, 0.1, 0.1, 2.0, 1.0, 0.1, 0.1, 0.1, 2.0, 1.0, 1.0]
            init_p = 1
        else
            init_hyperparameters = init_x[1:11]
            init_p = Int(init_x[12])
        end

        # Helper functions for bounded transformation (sigmoid-based)
        function y_to_x(y)
            y_upper = copy(y)
            for i in [1, 2, 3, 5, 6, 7, 8, 10, 11]
                y_upper[i] = min(y[i], log(ux[i] - 1e-16))
            end

            x = exp.(y_upper) .+ 1e-16
            # Apply bounded transformation to indices 3 and 7
            x[4] = lx[4] + (ux[4] - lx[4]) / (1 + exp(-y[4]))
            x[9] = lx[9] + (ux[9] - lx[9]) / (1 + exp(-y[9]))
            return x
        end

        function x_to_y(x)
            y = log.(x .- 1e-16)
            # Inverse transformation for indices 3 and 7
            y[4] = -log((ux[4] - lx[4]) / (x[4] - lx[4]) - 1)
            y[9] = -log((ux[9] - lx[9]) / (x[9] - lx[9]) - 1)
            return y
        end

        init_y = x_to_y(init_hyperparameters)

        function neg_logmarg_fixedp(y, p_fixed)
            x = y_to_x(y)
            try
                val = negative_log_marginal([x; p_fixed])
                return isfinite(val) ? val : 1e10
            catch
                return 1e10
            end
        end

        # Step 1: Initial hyperparameter optimization with init_p
        println("Initial optimization with p=$init_p")
        sol = optimize(y -> neg_logmarg_fixedp(y, init_p), init_y, LBFGS(), Optim.Options(iterations=maxiter, f_abstol=1e-2, x_abstol=1e-3, g_abstol=1e-4, show_trace=true))
        all_x[init_p] = y_to_x(Optim.minimizer(sol))
        all_fitness[init_p] = Optim.minimum(sol)
        println("Initial x = $(all_x[init_p]), fitness = $(all_fitness[init_p])")

        current_x = all_x[init_p]
        prev_p = 0
        current_p = init_p
        iteration = 0

        # Alternating optimization loop
        while prev_p != current_p
            iteration += 1
            println("\n=== Alternating optimization iteration $iteration ===")

            # Step 2: Evaluate objective for all lags with current hyperparameters fixed
            println("Evaluating all lags with current hyperparameters...")
            all_fitness_temp = Vector{Float64}(undef, upper_p)
            for p_candidate in 1:upper_p
                try
                    all_fitness_temp[p_candidate] = negative_log_marginal([current_x; p_candidate])
                    if !isfinite(all_fitness_temp[p_candidate])
                        all_fitness_temp[p_candidate] = 1e10
                    end
                catch
                    all_fitness_temp[p_candidate] = 1e10
                end
                println("  p = $p_candidate: fitness = $(all_fitness_temp[p_candidate])")
            end

            # Step 3: Select best lag with parsimony principle
            prev_p = current_p
            best_p = argmin(all_fitness_temp)
            best_fitness = all_fitness_temp[best_p]

            valid_lags = [p_candidate for p_candidate in 1:upper_p if all_fitness_temp[p_candidate] - best_fitness <= ml_tol]
            current_p = isempty(valid_lags) ? best_p : minimum(valid_lags)

            current_fitness = all_fitness_temp[current_p]

            println("Selected p = $current_p with fitness = $current_fitness")

            if prev_p == current_p
                println("Converged: optimal lag unchanged at p = $current_p")
                println("Final minimizer: $current_x")
                break
            end

            # Step 4: Re-optimize hyperparameters with the newly selected lag
            println("Re-optimizing hyperparameters with p = $current_p")
            current_y = x_to_y(current_x)
            sol = optimize(y -> neg_logmarg_fixedp(y, current_p), current_y, LBFGS(), Optim.Options(iterations=maxiter, f_abstol=1e-2, x_abstol=1e-3, g_abstol=1e-4, show_trace=true))
            all_x[current_p] = y_to_x(Optim.minimizer(sol))
            all_fitness[current_p] = Optim.minimum(sol)
            current_x = all_x[current_p]
            println("Re-optimized x = $current_x, fitness = $(all_fitness[current_p])")
        end

        p = current_p
        q = [current_x[1] current_x[6]
            current_x[2] current_x[7]
            current_x[3] current_x[8]
            current_x[4] current_x[9]
            current_x[5] current_x[10]]
        nu0 = current_x[11] + dP + 1
        opt = (minimizer=current_x, minimum=all_fitness[current_p], p=current_p, all_minimizer=all_x, all_minimum=all_fitness)
    end

    PCs = PCA(yields[(upper_p-p)+1:end, :], p; pca_loadings)[1]
    if isempty(macros)
        factors = copy(PCs)
    else
        factors = [PCs macros[(upper_p-p)+1:end, :]]
    end
    Omega0 = Vector{Float64}(undef, dP)
    for i in eachindex(Omega0)
        Omega0[i] = (AR_res_var(factors[:, i], p)[1]) * (optimizer == :BBO ? best_candidate(opt)[11] : current_x[11])
    end

    return Hyperparameter(p=copy(p), q=copy(q), nu0=copy(nu0), Omega0=copy(Omega0), mean_phi_const=copy(mean_phi_const[:, p])), opt

end

"""
    tuning_hyperparameter_with_vs(yields, macros, tau_n, rho; populationsize=50, maxiter=10_000, medium_tau=collect(24:3:48), upper_q=[1 1; 1 1; 1 1; 4 4; 100 100], mean_kQ_infty=0, std_kQ_infty=0.1, upper_nu0=[], mean_phi_const=[], fix_const_PC1=false, upper_p=24, mean_phi_const_PC1=[], data_scale=1200, kappaQ_prior_pr=[], init_nu0=[], is_pure_EH=false, psi_const=[], pca_loadings=[], prior_mean_diff_kappaQ=[], prior_std_diff_kappaQ=[], optimizer=:LBFGS, ml_tol=1.0, init_x=[])
This function optimizes the hyperparameters with automatic variable selection: selects which macro variables affect latent factors (PCs).
# Input
- When comparing marginal likelihoods between models, the data for the dependent variable should be the same across models. To achieve this, we set the period of the dependent variable based on `upper_p`. For example, if `upper_p = 3`, `yields[4:end,:]` and `macros[4:end,:]` are the data for the dependent variable. `yields[1:3,:]` and `macros[1:3,:]` are used for setting initial observations for all lags.
- `optimizer`: The optimization algorithm to use.
    - `:LBFGS` (default): Alternates between lag selection, forward stepwise variable selection for coefficients of macro variables on latent factors, and hyperparameter optimization. Variable selection stops when log marginal likelihood improvement ≤ 1.0.
    - `:BBO`: Uses BlackBoxOptim.jl to optimize lag, hyperparameters, and variable selection simultaneously.
- `ml_tol`: Tolerance for parsimony in lag selection (only for `:LBFGS`). After finding the lag with the best marginal likelihood, the algorithm iteratively selects smaller lags if their marginal likelihood is within `ml_tol` of the best. This favors simpler models (smaller lags) when performance is comparable.
- `init_x`: Initial values for hyperparameters and lag (only for `:LBFGS`). Should be a vector of length 12 in the format `[vec(q); nu0-(dP+1); p]`. If empty (default), uses `[0.1, 0.1, 0.1, 2.0, 1.0, 0.1, 0.1, 0.1, 2.0, 1.0, 1.0, 1]`.
- `populationsize` and `maxiter` are options for the optimizer.
    - `populationsize`: the number of candidate solutions in each generation (only for `:BBO`)
    - `maxiter`: the maximum number of iterations
- The lower bounds for `q` and `nu0` are `0` and `dP+2`.
- The upper bounds for `q`, `nu0`, and VAR lag can be set by `upper_q`, `upper_nu0`, and `upper_p`.
    - The default option for `upper_nu0` is the time-series length of the data.
- If you use the default option for `mean_phi_const`,
    1. `mean_phi_const[dQ+1:end]` is a zero vector.
    2. `mean_phi_const[1:dQ]` is calibrated to make the prior mean of `λₚ` a zero vector.
    3. After step 2, `mean_phi_const[1]` is replaced with `mean_phi_const_PC1` if it is not empty.
- `mean_phi_const = Matrix(your prior, dP, upper_p)`
- `mean_phi_const[:,i]` is the prior mean for the VAR(`i`) constant. Therefore, `mean_phi_const` is a matrix only in this function. In other functions, `mean_phi_const` is a vector for the orthogonalized VAR system with the selected lag.
- When `fix_const_PC1==true`, the first element in the constant term in the orthogonalized VAR is fixed to its prior mean during posterior sampling.
- `data_scale::scalar`: In a typical affine term structure model, theoretical yields are in decimals and not annualized. However, for convenience (public data usually contains annualized percentage yields) and numerical stability, we sometimes want to scale up yields and use (`data_scale`*theoretical yields) as the variable `yields`. In this case, you can use the `data_scale` option. For example, we can set `data_scale = 1200` and use annualized percentage monthly yields as `yields`.
- `kappaQ_prior_pr` is a vector of prior distributions for `kappaQ` under the JSZ model: each element specifies the prior for `kappaQ[i]` and must be provided as a `Distributions.jl` object. Alternatively, you can supply `prior_mean_diff_kappaQ` and `prior_std_diff_kappaQ`, which define means and standard deviations for Normal priors on `[kappaQ[1]; diff(kappaQ)]`; the implied Normal prior for each `kappaQ[i]` is then truncated to (0, 1). These options are only needed when using the JSZ model.
- `is_pure_EH::Bool`: When `mean_phi_const=[]`, `is_pure_EH=false` sets `mean_phi_const` to zero vectors. Otherwise, `mean_phi_const` is set to imply the pure expectation hypothesis under `mean_phi_const=[]`.
- `psi_const` and `psi` (dP × dP*p) are multiplied with prior variances of coefficients of the intercept and lagged regressors in the orthogonalized transition equation. Variable selection operates on all columns of `psi`: columns 1:dQ (lag 1 PCs) are always included, and all other columns are candidates. Setting `psi[1:dQ, col] = 1e-16` excludes a variable's effect on latent factors. For lag k, variable j, the column index is (k-1)*dP+j.
- `pca_loadings=Matrix{, dQ, size(yields, 2)}` stores the loadings for the first dQ principal components (so `principal_components = yields * pca_loadings'`), and you may optionally provide these loadings externally; if omitted, the package computes them internally via PCA.
# Output(3)
Optimized hyperparameter, optimization result, psi matrix
- The second output contains optimization results: when `optimizer=:LBFGS`, a NamedTuple with `minimizer`, `minimum`, `p`, `all_minimizer`, `all_minimum`, `selected_vars`, `psi`; when `optimizer=:BBO`, a NamedTuple with `opt` (bboptimize result), `selected_vars`, `psi`. `selected_vars` is a sorted list of (lag, variable) tuples indicating which columns are included beyond the always-included columns 1:dQ.
- The third output is `psi` (dP × dP*p), the final prior variance scaling matrix for VAR coefficients.
"""
function tuning_hyperparameter_with_vs(yields, macros, tau_n, rho; populationsize=50, maxiter=10_000, medium_tau=collect(24:3:48), upper_q=[1 1; 1 1; 1 1; 4 4; 100 100], mean_kQ_infty=0, std_kQ_infty=0.1, upper_nu0=[], mean_phi_const=[], fix_const_PC1=false, upper_p=24, mean_phi_const_PC1=[], data_scale=1200, kappaQ_prior_pr=[], init_nu0=[], is_pure_EH=false, psi_common=[], psi_const=[], pca_loadings=[], prior_mean_diff_kappaQ=[], prior_std_diff_kappaQ=[], optimizer=:LBFGS, ml_tol=1.0, init_x=[])


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
        if isempty(prior_mean_diff_kappaQ)
            kappaQ_prior_pr = length(medium_tau) |> x -> ones(x) / x
        else
            kappaQ_prior_pr = [truncated(Normal(prior_mean_diff_kappaQ[1], prior_std_diff_kappaQ[1]), eps(), 1 - eps())]
            for i in 2:length(prior_mean_diff_kappaQ)
                kappaQ_prior_pr = [kappaQ_prior_pr; truncated(convolve(Normal(prior_mean_diff_kappaQ[i], prior_std_diff_kappaQ[i]), deepcopy(kappaQ_prior_pr[i-1].untruncated)), eps(), 1 - eps())]
            end
        end
    end

    # Variable selection: total candidate columns = dP*upper_p - dQ
    # Always included: columns 1:dQ (lag 1 PCs). All others are candidates.
    n_vs_params = dP * upper_p - dQ

    if isempty(psi_common)
        # Initialize psi: only (1,1)..(1,dQ) included, all others excluded
        psi = ones(dP, dP * upper_p)
        if n_vs_params > 0
            psi[1:dQ, (dQ+1):end] .= 1e-16
        end
    else
        psi = kron(ones(1, upper_p), psi_common)
    end
    if isempty(psi_const)
        psi_const = ones(dP)
    end

    # Add variable selection parameters for BBO (binary: 0 or 1 for each candidate column)
    lx = [0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 1; 1]
    ux = 0.0 .+ [vec(upper_q); upper_nu0 - (dP + 1); upper_p]
    if n_vs_params > 0
        lx = [lx; zeros(n_vs_params)]  # 0 = excluded
        ux = [ux; ones(n_vs_params)]   # 1 = included
    end

    if isempty(mean_phi_const) && is_pure_EH
        mean_phi_const = Matrix{Float64}(undef, dP, upper_p)
        for i in axes(mean_phi_const, 2)
            mean_phi_const_PCs = -calibrate_mean_phi_const(mean_kQ_infty, std_kQ_infty, init_nu0, yields[upper_p-i+1:end, :], macros[upper_p-i+1:end, :], tau_n, i; medium_tau, iteration=10_000, data_scale, kappaQ_prior_pr, pca_loadings)[1] |> x -> mean(x, dims=1)[1, :]
            if !isempty(mean_phi_const_PC1)
                mean_phi_const_PCs = [mean_phi_const_PC1, mean_phi_const_PCs[2], mean_phi_const_PCs[3]]
            end
            if isempty(macros)
                mean_phi_const[:, i] = copy(mean_phi_const_PCs)
            else
                mean_phi_const[:, i] = [mean_phi_const_PCs; zeros(size(macros, 2))]
            end
            prior_const_TP = calibrate_mean_phi_const(mean_kQ_infty, std_kQ_infty, init_nu0, yields[upper_p-i+1:end, :], macros[upper_p-i+1:end, :], tau_n, i; medium_tau, mean_phi_const_PCs, iteration=10_000, data_scale, kappaQ_prior_pr, τ=120, pca_loadings)[2]
            println("For lag $i, mean_phi_const[1:dQ] is $mean_phi_const_PCs ,")
            println("and prior mean of the constant part in the term premium is $(mean(prior_const_TP)),")
            println("and prior std of the constant part in the term premium is $(std(prior_const_TP)).")
            println(" ")
        end
    elseif isempty(mean_phi_const) && !is_pure_EH
        mean_phi_const = zeros(dP, upper_p)
    end
    starting = (lx + ux) ./ 2
    starting[12] = 1  # initial lag
    if n_vs_params > 0
        starting[13:end] .= 0  # start with all candidate variables excluded
    end

    function negative_log_marginal(input)

        # parameters
        q = [input[1] input[6]
            input[2] input[7]
            input[3] input[8]
            input[4] input[9]
            input[5] input[10]]
        nu0 = input[11] + dP + 1
        p = Int(input[12])

        # Variable selection: for BBO, read from input[13:end]; for LBFGS, use outer psi (closure)
        if length(input) > 12 && n_vs_params > 0
            # BBO: columns 1:dQ always included, params map to columns dQ+1:dP*p
            psi_local = ones(dP, dP * p)
            psi_local[1:dQ, (dQ+1):end] .= 1e-16
            n_vs_active = dP * p - dQ  # relevant params for this p
            for i in 1:n_vs_active
                if Int(input[12+i]) == 1  # column dQ+i is included
                    psi_local[1:dQ, dQ+i] .= 1
                end
            end
        else
            # LBFGS: use outer psi (closure), trim to current p
            psi_local = copy(psi[:, 1:dP*p])
        end

        PCs, ~, Wₚ = PCA(yields[(upper_p-p)+1:end, :], p; pca_loadings)
        if isempty(macros)
            factors = copy(PCs)
        else
            factors = [PCs macros[(upper_p-p)+1:end, :]]
        end
        Omega0 = Vector{Float64}(undef, dP)
        for i in eachindex(Omega0)
            Omega0[i] = (AR_res_var(factors[:, i], p)[1]) * input[11]
        end

        tuned = Hyperparameter(p=copy(p), q=copy(q), nu0=copy(nu0), Omega0=copy(Omega0), mean_phi_const=copy(mean_phi_const[:, p]))
        if isempty(macros)
            return -log_marginal(factors, macros, rho, tuned, tau_n, Wₚ; medium_tau, kappaQ_prior_pr, fix_const_PC1, psi=psi_local, psi_const)
        else
            return -log_marginal(factors[:, 1:dQ], factors[:, dQ+1:end], rho, tuned, tau_n, Wₚ; medium_tau, kappaQ_prior_pr, fix_const_PC1, psi=psi_local, psi_const)
        end

        # Although the input data should contains initial observations, the argument of the marginal likelihood should be the same across the candidate models. Therefore, we should align the length of the dependent variable across the models.

    end

    if optimizer == :BBO
        # -1 = continuous, 0 = integer
        # [hyperparameters (11 continuous); lag (1 integer); variable selection (n_vs_params integers)]
        search_space_types = [-1 * ones(Int64, 11); zeros(Int64, 1 + n_vs_params)]
        ss = MixedPrecisionRectSearchSpace(lx, ux, search_space_types)
        opt = bboptimize(negative_log_marginal, starting; SearchSpace=ss, MaxSteps=maxiter, PopulationSize=populationsize, CallbackInterval=10, CallbackFunction=x -> println("Current Best: p = $(Int(best_candidate(x)[12])), q[:,1] = $(best_candidate(x)[1:5]), q[:,2] = $(best_candidate(x)[6:10]), nu0 = $(best_candidate(x)[11] + dP + 1)"))

        q = [best_candidate(opt)[1] best_candidate(opt)[6]
            best_candidate(opt)[2] best_candidate(opt)[7]
            best_candidate(opt)[3] best_candidate(opt)[8]
            best_candidate(opt)[4] best_candidate(opt)[9]
            best_candidate(opt)[5] best_candidate(opt)[10]]
        nu0 = best_candidate(opt)[11] + dP + 1
        p = best_candidate(opt)[12] |> Int

        # Extract selected variables (lag, variable) pairs; ignore params beyond selected p
        selected_vars = Tuple{Int,Int}[]
        psi_result = ones(dP, dP * p)
        psi_result[1:dQ, (dQ+1):end] .= 1e-16
        n_vs_active = dP * p - dQ
        for i in 1:n_vs_active
            col = dQ + i
            k = (col - 1) ÷ dP + 1
            j = (col - 1) % dP + 1
            if Int(best_candidate(opt)[12+i]) == 1
                push!(selected_vars, (k, j))
                psi_result[1:dQ, col] .= 1
            end
        end

        # Extend opt with variable selection info
        opt = (opt=opt, selected_vars=selected_vars, psi=copy(psi_result))

    elseif optimizer == :LBFGS
        # Alternating optimization between hyperparameters, lag selection, and variable selection
        # all_x[p] stores optimized hyperparameters for lag p
        # all_fitness[p] stores the objective value for that optimization
        all_x = [fill(NaN, 11) for _ in 1:upper_p]
        all_fitness = fill(NaN, upper_p)

        # Set initial values: [vec(q); nu0; p]
        if isempty(init_x)
            init_hyperparameters = [0.1, 0.1, 0.1, 2.0, 1.0, 0.1, 0.1, 0.1, 2.0, 1.0, 1.0]
            init_p = 1
        else
            init_hyperparameters = init_x[1:11]
            init_p = Int(init_x[12])
        end

        # Helper functions for bounded transformation (sigmoid-based)
        function y_to_x(y)
            y_upper = copy(y)
            for i in [1, 2, 3, 5, 6, 7, 8, 10, 11]
                y_upper[i] = min(y[i], log(ux[i] - 1e-16))
            end

            x = exp.(y_upper) .+ 1e-16
            # Apply bounded transformation to indices 3 and 7
            x[4] = lx[4] + (ux[4] - lx[4]) / (1 + exp(-y[4]))
            x[9] = lx[9] + (ux[9] - lx[9]) / (1 + exp(-y[9]))
            return x
        end

        function x_to_y(x)
            y = log.(x .- 1e-16)
            # Inverse transformation for indices 3 and 7
            y[4] = -log((ux[4] - lx[4]) / (x[4] - lx[4]) - 1)
            y[9] = -log((ux[9] - lx[9]) / (x[9] - lx[9]) - 1)
            return y
        end

        init_y = x_to_y(init_hyperparameters)

        function neg_logmarg_fixedp(y, p_fixed)
            x = y_to_x(y)
            try
                val = negative_log_marginal([x; p_fixed])
                return isfinite(val) ? val : 1e10
            catch
                return 1e10
            end
        end

        # Step 1: Initial hyperparameter optimization with init_p
        println("Initial optimization with p=$init_p")
        sol = optimize(y -> neg_logmarg_fixedp(y, init_p), init_y, LBFGS(), Optim.Options(iterations=maxiter, f_abstol=1e-2, x_abstol=1e-3, g_abstol=1e-4, show_trace=true))
        all_x[init_p] = y_to_x(Optim.minimizer(sol))
        all_fitness[init_p] = Optim.minimum(sol)
        println("Initial x = $(all_x[init_p]), fitness = $(all_fitness[init_p])")

        current_x = all_x[init_p]
        prev_p = 0
        current_p = init_p
        iteration = 0

        # Initialize variable selection: start with all candidate variables excluded
        selected_vars = Set{Tuple{Int,Int}}()  # (lag, variable) pairs
        prev_selected_vars = Set{Tuple{Int,Int}}()

        # Alternating optimization loop
        while prev_p != current_p || prev_selected_vars != selected_vars || iteration == 0
            iteration += 1
            println("\n=== Alternating optimization iteration $iteration ===")

            # Step 2: Evaluate objective for all lags with current hyperparameters fixed
            println("Evaluating all lags with current hyperparameters...")
            all_fitness_temp = Vector{Float64}(undef, upper_p)
            for p_candidate in 1:upper_p
                try
                    all_fitness_temp[p_candidate] = negative_log_marginal([current_x; p_candidate])
                    if !isfinite(all_fitness_temp[p_candidate])
                        all_fitness_temp[p_candidate] = 1e10
                    end
                catch
                    all_fitness_temp[p_candidate] = 1e10
                end
                println("  p = $p_candidate: fitness = $(all_fitness_temp[p_candidate])")
            end

            # Step 3: Select best lag with parsimony principle
            prev_p = current_p
            prev_selected_vars = copy(selected_vars)
            best_p = argmin(all_fitness_temp)
            best_fitness = all_fitness_temp[best_p]

            valid_lags = [p_candidate for p_candidate in 1:upper_p if all_fitness_temp[p_candidate] - best_fitness <= ml_tol]
            current_p = isempty(valid_lags) ? best_p : minimum(valid_lags)

            current_fitness = all_fitness_temp[current_p]

            println("Selected p = $current_p with fitness = $current_fitness")

            # Step 4: Hierarchical forward stepwise variable selection
            # Lag 1: macro candidates only (PCs already included)
            # Lag k≥2: only variables included at lag k-1 are candidates (including PCs)
            n_vs_active = dP * current_p - dQ
            if n_vs_active > 0
                # Reset variable selection: start from empty set each iteration
                selected_vars = Set{Tuple{Int,Int}}()

                # Reset psi: only columns 1:dQ included, all others excluded
                psi[1:dQ, (dQ+1):end] .= 1e-16

                println("\n--- Hierarchical forward stepwise variable selection (p=$current_p) ---")
                current_logmarg = -current_fitness

                for lag in 1:current_p
                    # Determine candidates for this lag
                    if lag == 1
                        # Lag 1: only macro variables (PCs already included)
                        candidate_vars = collect((dQ+1):dP)
                    else
                        # Lag k≥2: only variables included at lag k-1
                        if lag == 2
                            # Included at lag 1 = PCs (always) + selected macros
                            prev_included = Set(1:dQ)
                            for (l, j) in selected_vars
                                if l == 1
                                    push!(prev_included, j)
                                end
                            end
                        else
                            # Included at lag k-1 = variables selected at lag k-1
                            prev_included = Set{Int}()
                            for (l, j) in selected_vars
                                if l == lag - 1
                                    push!(prev_included, j)
                                end
                            end
                        end
                        candidate_vars = sort(collect(prev_included))
                    end

                    if isempty(candidate_vars)
                        println("  Lag $lag: no candidates → stopping")
                        break
                    end

                    println("  Lag $lag: $(length(candidate_vars)) candidates = $candidate_vars")

                    # Forward stepwise for this lag
                    while true
                        best_candidate_var = (0, 0)
                        best_candidate_logmarg = current_logmarg

                        for j in candidate_vars
                            if (lag, j) ∈ selected_vars
                                continue
                            end

                            col = (lag - 1) * dP + j
                            psi[1:dQ, col] .= 1

                            try
                                temp_fitness = negative_log_marginal([current_x; current_p])
                                temp_logmarg = -temp_fitness

                                println("    Candidate (lag=$lag, var=$j): log_marginal = $temp_logmarg (improvement = $(temp_logmarg - current_logmarg))")

                                if temp_logmarg > best_candidate_logmarg
                                    best_candidate_logmarg = temp_logmarg
                                    best_candidate_var = (lag, j)
                                end
                            catch e
                                println("    Candidate (lag=$lag, var=$j): evaluation failed ($e)")
                            end

                            psi[1:dQ, col] .= 1e-16
                        end

                        improvement = best_candidate_logmarg - current_logmarg
                        if improvement <= ml_tol || best_candidate_var == (0, 0)
                            println("  Lag $lag: stopped (improvement = $improvement)")
                            break
                        end

                        push!(selected_vars, best_candidate_var)
                        current_logmarg = best_candidate_logmarg
                        k_best, j_best = best_candidate_var
                        psi[1:dQ, (k_best-1)*dP+j_best] .= 1

                        println("    Added (lag=$k_best, var=$j_best) to model (log_marginal = $current_logmarg)")
                    end
                end

                println("Selected variables: $(sort(collect(selected_vars)))")
            end

            if prev_p == current_p && prev_selected_vars == selected_vars && iteration > 1
                println("Converged: optimal lag unchanged at p = $current_p and variable selection stabilized")
                println("Selected variables: $(sort(collect(selected_vars)))")
                println("Final minimizer: $current_x")
                break
            end

            # Step 5: Re-optimize hyperparameters with the newly selected lag and variables
            println("Re-optimizing hyperparameters with p = $current_p")
            current_y = x_to_y(current_x)
            sol = optimize(y -> neg_logmarg_fixedp(y, current_p), current_y, LBFGS(), Optim.Options(iterations=maxiter, f_abstol=1e-2, x_abstol=1e-3, g_abstol=1e-4, show_trace=true))
            all_x[current_p] = y_to_x(Optim.minimizer(sol))
            all_fitness[current_p] = Optim.minimum(sol)
            current_x = all_x[current_p]
            println("Re-optimized x = $current_x, fitness = $(all_fitness[current_p])")
        end

        p = current_p
        q = [current_x[1] current_x[6]
            current_x[2] current_x[7]
            current_x[3] current_x[8]
            current_x[4] current_x[9]
            current_x[5] current_x[10]]
        nu0 = current_x[11] + dP + 1
        opt = (minimizer=current_x, minimum=all_fitness[current_p], p=current_p, all_minimizer=all_x, all_minimum=all_fitness, selected_vars=sort(collect(selected_vars)), psi=copy(psi[:, 1:dP*current_p]))
    end
    PCs = PCA(yields[(upper_p-p)+1:end, :], p; pca_loadings)[1]
    if isempty(macros)
        factors = copy(PCs)
    else
        factors = [PCs macros[(upper_p-p)+1:end, :]]
    end
    Omega0 = Vector{Float64}(undef, dP)
    for i in eachindex(Omega0)
        Omega0[i] = (AR_res_var(factors[:, i], p)[1]) * (optimizer == :BBO ? best_candidate(opt.opt)[11] : current_x[11])
    end

    # Final psi: trim to selected lag length
    psi_final = optimizer == :BBO ? opt.psi : copy(psi[:, 1:dP*p])

    return Hyperparameter(p=copy(p), q=copy(q), nu0=copy(nu0), Omega0=copy(Omega0), mean_phi_const=copy(mean_phi_const[:, p])), opt, psi_final

end

"""
    AR_res_var(TS::Vector, p)
This function derives the MLE error variance estimate of an AR(`p`) model.
# Input
- Univariate time series `TS` and lag `p`
# Output(2)
Residual variance estimate, AR(p) coefficients
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
    posterior_sampler(yields, macros, tau_n, rho, iteration, tuned::Hyperparameter; medium_tau=collect(24:3:48), init_param=[], psi=[], psi_const=[], gamma_bar=[], kappaQ_prior_pr=[], mean_kQ_infty=0, std_kQ_infty=0.1, fix_const_PC1=false, data_scale=1200, pca_loadings=[], kappaQ_proposal_mode=[])
This function samples from the posterior distribution.
# Input
- `iteration`: Number of posterior samples
- `tuned`: Optimized hyperparameters used during estimation
- `init_param`: Starting point of the sampler. It should be of type Parameter.
- `psi_const` and `psi` are multiplied with prior variances of coefficients of the intercept and lagged regressors in the orthogonalized transition equation. They are used for imposing zero prior variances. An empty default value means that you do not use this function. `[psi_const psi][i,j]` corresponds to `phi[i,j]`. The entries of `psi` and `psi_const` should be nearly zero (e.g., `1e-10`), not exactly zero.
- `kappaQ_prior_pr` is a vector of prior distributions for `kappaQ` under the JSZ model: each element specifies the prior for `kappaQ[i]` and must be provided as a `Distributions.jl` object. This option is only needed when using the JSZ model.
- `pca_loadings=Matrix{, dQ, size(yields, 2)}` stores the loadings for the first dQ principal components (so `principal_components = yields * pca_loadings'`), and you may optionally provide these loadings externally; if omitted, the package computes them internally via PCA.
- `kappaQ_proposal_mode=Vector{, dQ}` contains the center of the proposal distribution for `kappaQ`. If it is empty, it is optimized by MLE.
# Output(2)
`Vector{Parameter}(posterior, iteration)`, acceptance rate of the MH algorithm
"""
function posterior_sampler(yields, macros, tau_n, rho, iteration, tuned::Hyperparameter; medium_tau=collect(24:3:48), init_param=[], psi=[], psi_const=[], gamma_bar=[], kappaQ_prior_pr=[], mean_kQ_infty=0, std_kQ_infty=0.1, fix_const_PC1=false, data_scale=1200, pca_loadings=[], kappaQ_proposal_mode=[])

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
    Wₚ = PCA(yields, p; pca_loadings)[3]
    prior_kappaQ_ = prior_kappaQ(medium_tau, kappaQ_prior_pr)
    if isempty(gamma_bar)
        gamma_bar = prior_gamma(yields, p; pca_loadings)[1]
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

        ΩPP = mle_error_covariance(yields, [], tau_n, p; pca_loadings)
        function logpost(x)
            kappaQ_logpost = cumsum(x[1:dQ])
            kQ_infty_logpost = x[dQ+1]
            SigmaO_logpost = x[dQ+1+1:dQ+1+length(tau_n)-dQ] |> x -> exp.(x)
            if maximum(abs.(kappaQ_logpost)) > 1 || !(sort(kappaQ_logpost, rev=true) == kappaQ_logpost) || !isposdef(diagm(SigmaO_logpost)) || !(minimum(kappaQ_logpost .∈ support.(prior_kappaQ_))) || !isempty(findall(abs.(diff(kappaQ_logpost)) .<= eps()))
                return -Inf
            end

            logprior = 0.0
            for i in eachindex(prior_kappaQ_)
                logprior += logpdf(prior_kappaQ_[i], kappaQ_logpost[i])
            end

            return logprior + loglik_mea2(yields, tau_n, p; kappaQ=kappaQ_logpost, kQ_infty=kQ_infty_logpost, ΩPP, SigmaO=SigmaO_logpost, data_scale, pca_loadings)

        end
        if isempty(kappaQ_proposal_mode)
            # Construct the proposal distribution
            #kappaQ = 0.2rand(3) .+ 0.8 |> x -> sort(x, rev=true)
            x = [kappaQ[1]; diff(kappaQ[1:end])]
            init = [x; kQ_infty; log.(SigmaO)]
            minimizers = optimize(x -> -logpost(x), [0; -1 * ones(length(kappaQ) - 1); -Inf; fill(-Inf, length(tau_n) - dQ)], [1; 0.01 * ones(length(kappaQ) - 1); Inf; fill(Inf, length(tau_n) - dQ)], init, ParticleSwarm(), Optim.Options(show_trace=true)) |>
                         Optim.minimizer |>
                         y -> optimize(x -> -logpost(x), [0; -1 * ones(length(kappaQ) - 1); -Inf; fill(-Inf, length(tau_n) - dQ)], [1; eps() * ones(length(kappaQ) - 1); Inf; fill(Inf, length(tau_n) - dQ)], y, Fminbox(LBFGS(; alphaguess=LineSearches.InitialPrevious())), Optim.Options(show_trace=true)) |>
                              Optim.minimizer
        else
            diff_kappaQ_proposal_mode = [kappaQ_proposal_mode[1]; diff(kappaQ_proposal_mode[1:end])]
            init = [kQ_infty; log.(SigmaO)]
            minimizers = optimize(x -> -logpost([diff_kappaQ_proposal_mode; x]), [-Inf; fill(-Inf, length(tau_n) - dQ)], [Inf; fill(Inf, length(tau_n) - dQ)], init, ParticleSwarm(), Optim.Options(show_trace=true)) |>
                         Optim.minimizer |>
                         y -> optimize(x -> -logpost([diff_kappaQ_proposal_mode; x]), [-Inf; fill(-Inf, length(tau_n) - dQ)], [Inf; fill(Inf, length(tau_n) - dQ)], y, Fminbox(LBFGS(; alphaguess=LineSearches.InitialPrevious())), Optim.Options(show_trace=true)) |>
                              Optim.minimizer |> x -> [diff_kappaQ_proposal_mode; x]
        end
        x_mode = minimizers[1:dQ]
        x_hess = hessian(x -> -logpost([x; minimizers[dQ+1:end]]), x_mode)
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
            kappaQ = rand(post_kappaQ(yields, prior_kappaQ_, tau_n; kQ_infty, phi, varFF, SigmaO, data_scale, pca_loadings))
        else
            kappaQ, isaccept = post_kappaQ2(yields, prior_kappaQ_, tau_n; kappaQ, kQ_infty, phi, varFF, SigmaO, data_scale, x_mode, inv_x_hess, pca_loadings)
            isaccept_MH[end] += isaccept
        end

        kQ_infty = rand(post_kQ_infty(mean_kQ_infty, std_kQ_infty, yields, tau_n; kappaQ, phi, varFF, SigmaO, data_scale, pca_loadings))

        phi, varFF, isaccept = post_phi_varFF(yields, macros, mean_phi_const, rho, prior_kappaQ_, tau_n; phi, psi, psi_const, varFF, q, nu0, Omega0, kappaQ, kQ_infty, SigmaO, fix_const_PC1, data_scale, pca_loadings)
        isaccept_MH[1:dQ] += isaccept

        SigmaO = rand.(post_SigmaO(yields, tau_n; kappaQ, kQ_infty, ΩPP=phi_varFF_2_ΩPP(; phi, varFF, dQ), gamma, p, data_scale, pca_loadings))

        gamma = rand.(post_gamma(; gamma_bar, SigmaO))

        saved_params[iter] = Parameter(kappaQ=copy(kappaQ), kQ_infty=copy(kQ_infty), phi=copy(phi), varFF=copy(varFF), SigmaO=copy(SigmaO), gamma=copy(gamma))

    end

    return saved_params, 100isaccept_MH / iteration
end

"""
    posterior_NUTS(p, yields, macros, tau_n, rho, NUTS_nadapt, iteration; init_param=[], prior_q, prior_nu0, psi=[], psi_const=[], gamma_bar=[], prior_mean_diff_kappaQ, prior_std_diff_kappaQ, mean_kQ_infty=0, std_kQ_infty=0.1, fix_const_PC1=false, data_scale=1200, pca_loadings=[], NUTS_target_acceptance_rate=0.65, NUTS_max_depth=10)
This function implements the NUTS-within-Gibbs sampler. Gibbs blocks that cannot be updated with conjugate priors are sampled using the NUTS sampler.
# Input
- `p`: The lag length of the VAR system
- `NUTS_nadapt`: Number of iterations for tuning settings in the NUTS sampler. The warmup samples are included in the output, so you should discard them.
- `iteration`: Number of posterior samples
- `init_param`: Starting point of the sampler. It should be of type Parameter_NUTS.
- `prior_q`: A 4 by 2 matrix that contains the prior distribution for q. All entries should be objects in `Distributions.jl`. For hyperparameters that do not need to be optimized, assigning a `Dirac(::Float64)` prior to the corresponding entry fixes that hyperparameter and optimizes only the remaining hyperparameters.
- `prior_nu0`: The prior distribution for nu0 - (dP + 1). It should be an object in `Distributions.jl`.
- `psi_const` and `psi` are multiplied with prior variances of coefficients of the intercept and lagged regressors in the orthogonalized transition equation. They are used for imposing zero prior variances. An empty default value means that you do not use this function. `[psi_const psi][i,j]` corresponds to `phi[i,j]`. The entries of `psi` and `psi_const` should be nearly zero (e.g., `1e-10`), not exactly zero.
- `prior_mean_diff_kappaQ` and `prior_std_diff_kappaQ` are vectors that contain the means and standard deviations of the Normal distributions for `[kappaQ[1]; diff(kappaQ)]`. Once Normal priors are assigned to these parameters, the prior for `kappaQ[1]` is truncated to (0, 1), and the priors for `diff(kappaQ)` are truncated to (−1, 0).
- `pca_loadings=Matrix{, dQ, size(yields, 2)}` stores the loadings for the first dQ principal components (so `principal_components = yields * pca_loadings'`), and you may optionally provide these loadings externally; if omitted, the package computes them internally via PCA.  ￼
- `NUTS_target_acceptance_rate`, `NUTS_max_depth` are the arguments of the NUTS sampler in `AdvancedHMC.jl`.
# Output
`Vector{Parameter_NUTS}(posterior, iteration)`
"""
function posterior_NUTS(p, yields, macros, tau_n, rho, NUTS_nadapt, iteration; init_param=[], prior_q, prior_nu0, psi=[], psi_const=[], gamma_bar=[], prior_mean_diff_kappaQ, prior_std_diff_kappaQ, mean_kQ_infty=0, std_kQ_infty=0.1, fix_const_PC1=false, data_scale=1200, pca_loadings=[], NUTS_target_acceptance_rate=0.65, NUTS_max_depth=10)

    N = size(yields, 2) # of maturities
    dQ = dimQ() + size(yields, 2) - length(tau_n)
    PCs, _, Wₚ = PCA(yields, p; pca_loadings)
    if isempty(macros)
        dP = copy(dQ)
        factors = copy(PCs)
    else
        dP = dQ + size(macros, 2)
        factors = [PCs macros]
    end

    if isempty(gamma_bar)
        gamma_bar = prior_gamma(yields, p; pca_loadings)[1]
    end
    prior_diff_kappaQ = truncated(Normal(prior_mean_diff_kappaQ[1], prior_std_diff_kappaQ[1]), eps(), 1 - eps())
    for i in 2:length(prior_mean_diff_kappaQ)
        prior_diff_kappaQ = [prior_diff_kappaQ; truncated(Normal(prior_mean_diff_kappaQ[i], prior_std_diff_kappaQ[i]), -1 + eps(), -eps())]
    end

    if typeof(init_param) == Parameter_NUTS
        q, nu0, kappaQ, kQ_infty, phi, varFF, SigmaO, gamma = init_param.q, init_param.nu0, init_param.kappaQ, init_param.kQ_infty, init_param.phi, init_param.varFF, init_param.SigmaO, init_param.gamma
    else
        ## initial parameters ##
        q = mean.(prior_q)
        net_nu0 = mean.(prior_nu0)
        nu0 = net_nu0 + (dP + 1)
        Omega0 = Vector{Float64}(undef, dP)
        for i in eachindex(Omega0)
            Omega0[i] = (AR_res_var(factors[:, i], p)[1]) * net_nu0
        end

        kappaQ = prior_mean_diff_kappaQ |> cumsum
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

    chain = []
    sampler = fill(NUTS(ceil(Int, 0.1NUTS_nadapt), NUTS_target_acceptance_rate; metricT=AdvancedHMC.DenseEuclideanMetric, max_depth=NUTS_max_depth), dQ + 1 + 1)
    is_warmup = true
    saved_params = Vector{Parameter_NUTS}(undef, iteration)
    @showprogress 5 "posterior_sampler..." for iter in 1:iteration
        if iter > NUTS_nadapt
            is_warmup = false
        end
        kQ_infty = rand(post_kQ_infty(mean_kQ_infty, std_kQ_infty, yields, tau_n; kappaQ, phi, varFF, SigmaO, data_scale, pca_loadings))

        chain, q, nu0, kappaQ, phi, varFF = post_kappaQ_phi_varFF_q_nu0(yields, macros, tau_n, zeros(dP), rho, prior_q, prior_nu0, prior_diff_kappaQ; phi, psi, psi_const, varFF, q, nu0, kappaQ, kQ_infty, SigmaO, fix_const_PC1, data_scale, pca_loadings, sampler, chain, is_warmup)

        SigmaO = rand.(post_SigmaO(yields, tau_n; kappaQ, kQ_infty, ΩPP=phi_varFF_2_ΩPP(; phi, varFF, dQ), gamma, p, data_scale, pca_loadings))

        gamma = rand.(post_gamma(; gamma_bar, SigmaO))

        saved_params[iter] = Parameter_NUTS(q=copy(q), nu0=copy(nu0), kappaQ=copy(kappaQ), kQ_infty=copy(kQ_infty), phi=copy(phi), varFF=copy(varFF), SigmaO=copy(SigmaO), gamma=copy(gamma))

    end

    return saved_params
end

"""
    generative(T, dP, tau_n, p, noise::Float64; kappaQ, kQ_infty, KPXF, GPXFXF, OmegaXFXF, data_scale=1200)
This function generates simulation data given parameters. Note that all parameters are in the latent factor state space (i.e., parameters in struct LatentSpace). There are some differences in notation because it is difficult to express mathcal letters in VSCode. Therefore, mathcal{F} in the paper is expressed as `F` in VSCode, and "F" in the paper is expressed as `XF`.
# Input
- `noise`: Variance of the measurement errors
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
    ineff_factor(saved_params::Vector{Parameter}; is_parallel=false)
This function returns the inefficiency factors for each parameter.
# Input
- `Vector{Parameter}` from `posterior_sampler`
- `is_parallel` enables multi-threaded parallel computation when set to `true`.
# Output
- Estimated inefficiency factors are returned as a Tuple(`kappaQ`, `kQ_infty`, `gamma`, `SigmaO`, `varFF`, `phi`). For example, if you want to access the inefficiency factor of `phi`, you can use `Output.phi`.
- If `fix_const_PC1==true` in your optimized Hyperparameter struct, `Output.phi[1,1]` may be unreliable and should be ignored.
"""
function ineff_factor(saved_params::Vector{Parameter}; is_parallel=false)

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
    prog = Progress(iteration - 1; dt=5, desc="ineff_factor(1.vectorization)...")
    if is_parallel
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
    else
        for iter in 2:iteration
            kappaQ = saved_params[:kappaQ][iter]
            kQ_infty = saved_params[:kQ_infty][iter]
            phi = saved_params[:phi][iter] |> vec
            varFF = saved_params[:varFF][iter]
            SigmaO = saved_params[:SigmaO][iter]
            gamma = saved_params[:gamma][iter]

            vec_saved_params[iter, :] = [kappaQ; kQ_infty; gamma; SigmaO; varFF; phi]
            next!(prog)
        end
    end
    finish!(prog)

    ineff = Vector{Float64}(undef, size(vec_saved_params)[2])
    prog = Progress(size(vec_saved_params, 2); dt=5, desc="ineff_factor(2.calculation)...")
    if is_parallel
        Threads.@threads for i in axes(vec_saved_params, 2)
            ineff[i] = longvar(vec_saved_params[:, i]) / var(vec_saved_params[:, i])
            next!(prog)
        end
    else
        for i in axes(vec_saved_params, 2)
            ineff[i] = longvar(vec_saved_params[:, i]) / var(vec_saved_params[:, i])
            next!(prog)
        end
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
    ineff_factor(saved_params::Vector{Parameter_NUTS}; is_parallel=false)
This function returns the inefficiency factors for each parameter.
# Input
- `Vector{Parameter_NUTS}` from `posterior_NUTS`
- `is_parallel` enables multi-threaded parallel computation when set to `true`.
# Output
- Estimated inefficiency factors are returned as a Tuple(`q`, `nu0`, `kappaQ`, `kQ_infty`, `gamma`, `SigmaO`, `varFF`, `phi`). For example, if you want to access the inefficiency factor of `phi`, you can use `Output.phi`.
- If `fix_const_PC1==true` in your optimized Hyperparameter struct, `Output.phi[1,1]` may be unreliable and should be ignored.
"""
function ineff_factor(saved_params::Vector{Parameter_NUTS}; is_parallel=false)

    iteration = length(saved_params)

    init_q = saved_params[:q][1] |> vec
    init_nu0 = saved_params[:nu0][1]
    init_kappaQ = saved_params[:kappaQ][1]
    init_kQ_infty = saved_params[:kQ_infty][1]
    init_phi = saved_params[:phi][1] |> vec
    init_varFF = saved_params[:varFF][1]
    init_SigmaO = saved_params[:SigmaO][1]
    init_gamma = saved_params[:gamma][1]

    initial_θ = [init_q; init_nu0; init_kappaQ; init_kQ_infty; init_gamma; init_SigmaO; init_varFF; init_phi]
    vec_saved_params = Matrix{Float64}(undef, iteration, length(initial_θ))
    vec_saved_params[1, :] = initial_θ
    prog = Progress(iteration - 1; dt=5, desc="ineff_factor(1.vectorization)...")
    if is_parallel
        Threads.@threads for iter in 2:iteration
            q = saved_params[:q][iter] |> vec
            nu0 = saved_params[:nu0][iter]
            kappaQ = saved_params[:kappaQ][iter]
            kQ_infty = saved_params[:kQ_infty][iter]
            phi = saved_params[:phi][iter] |> vec
            varFF = saved_params[:varFF][iter]
            SigmaO = saved_params[:SigmaO][iter]
            gamma = saved_params[:gamma][iter]

            vec_saved_params[iter, :] = [q; nu0; kappaQ; kQ_infty; gamma; SigmaO; varFF; phi]
            next!(prog)
        end
    else
        for iter in 2:iteration
            q = saved_params[:q][iter] |> vec
            nu0 = saved_params[:nu0][iter]
            kappaQ = saved_params[:kappaQ][iter]
            kQ_infty = saved_params[:kQ_infty][iter]
            phi = saved_params[:phi][iter] |> vec
            varFF = saved_params[:varFF][iter]
            SigmaO = saved_params[:SigmaO][iter]
            gamma = saved_params[:gamma][iter]

            vec_saved_params[iter, :] = [q; nu0; kappaQ; kQ_infty; gamma; SigmaO; varFF; phi]
            next!(prog)
        end
    end
    finish!(prog)

    ineff = Vector{Float64}(undef, size(vec_saved_params)[2])
    prog = Progress(size(vec_saved_params, 2); dt=5, desc="ineff_factor(2.calculation)...")
    if is_parallel
        Threads.@threads for i in axes(vec_saved_params, 2)
            ineff[i] = longvar(vec_saved_params[:, i]) / var(vec_saved_params[:, i])
            next!(prog)
        end
    else
        for i in axes(vec_saved_params, 2)
            ineff[i] = longvar(vec_saved_params[:, i]) / var(vec_saved_params[:, i])
            next!(prog)
        end
    end
    finish!(prog)

    phi_ineff = ineff[length(init_q)+1+length(init_kappaQ)+1+length(init_gamma)+length(init_SigmaO)+length(init_varFF)+1:end] |> x -> reshape(x, size(saved_params[:phi][1], 1), size(saved_params[:phi][1], 2))
    dP = size(phi_ineff, 1)
    for i in 1:dP, j in i:dP
        phi_ineff[i, end-dP+j] = 0
    end

    if length(init_kappaQ) == 1
        kappaQ_ineff = ineff[length(init_q)+1+1]
    else
        kappaQ_ineff = ineff[length(init_q)+1+1:length(init_q)+1+length(init_kappaQ)]
    end
    return (;
        q=ineff[1:length(init_q)],
        nu0=ineff[length(init_q)+1],
        kappaQ=kappaQ_ineff,
        kQ_infty=ineff[length(init_q)+1+length(init_kappaQ)+1],
        gamma=ineff[length(init_q)+1+length(init_kappaQ)+1+1:length(init_q)+1+length(init_kappaQ)+1+length(init_gamma)],
        SigmaO=ineff[length(init_q)+1+length(init_kappaQ)+1+length(init_gamma)+1:length(init_q)+1+length(init_kappaQ)+1+length(init_gamma)+length(init_SigmaO)],
        varFF=ineff[length(init_q)+1+length(init_kappaQ)+1+length(init_gamma)+length(init_SigmaO)+1:length(init_q)+1+length(init_kappaQ)+1+length(init_gamma)+length(init_SigmaO)+length(init_varFF)],
        phi=copy(phi_ineff)
    )
end

"""
    longvar(v)
This function calculates the long-run variance of `v` using the quadratic spectral window with bandwidth selection of Andrews (1991). The AR(1) approximation is used.
# Input
- Time-series vector `v`
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
    mle_error_covariance(yields, macros, tau_n, p; pca_loadings=[])
This function calculates the MLE estimates of the error covariance matrix of the VAR(p) model.
- `pca_loadings=Matrix{, dQ, size(yields, 2)}` stores the loadings for the first dQ principal components (so `principal_components = yields * pca_loadings'`), and you may optionally provide these loadings externally; if omitted, the package computes them internally via PCA.  ￼
"""
function mle_error_covariance(yields, macros, tau_n, p; pca_loadings=[])

    ## Extracting PCs
    PCs = PCA(yields, p; pca_loadings)[1]
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
