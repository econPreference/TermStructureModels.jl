# Step 1. Tuning hyper-parameters

We have five hyper-parameters, `p`, `q`, `nu0`, `Omega0`, and `mean_phi_const`.

- `p::Float64`: lag length of the $\mathbb{P}$-VAR(p)
- `q::Matrix{Float64}( , 4, 2)`: Shrinkage degrees in the Minnesota prior
- `nu0::Float64`(d.f.) and `Omega0::Vector`(scale matrix): Prior distribution of the error covariance matrix in the $\mathbb{P}$-VAR(p)
- `mean_phi_const`: Prior mean of the intercept term in the $\mathbb{P}$-VAR(p)

We recommend [`tuning_hyperparameter`](https://econpreference.github.io/TermStructureModels.jl/dev/api/#TermStructureModels.tuning_hyperparameter-NTuple{4,%20Any}) for deciding the hyperparameters.

```julia
tuned, results = tuning_hyperparameter(yields, macros, tau_n, rho; populationsize=50, maxiter=10_000, medium_tau=collect(24:3:48), upper_q=[1 1; 1 1; 10 10; 100 100], mean_kQ_infty=0, std_kQ_infty=0.1, upper_nu0=[], mean_phi_const=[], fix_const_PC1=false, upper_p=18, mean_phi_const_PC1=[], data_scale=1200, medium_tau_pr=[], init_nu0=[])
```

!!! note
Since we adopt the Differential Evolutionary algorithm, it is hard to set the terminal condition. Our strategy was "Run the algorithm with sufficient `maxiter`(our defaults), and verify that it is an global optimum by plotting the objective function". It is appropriate for academic projects.

    However, it is not good for practical projects. small `populationsize` or `maxiter` may not lead to the best model, but it will find a good model. The prior distribution

If users accept our default values, the function simplifies, that is

```juila
tuned, results = tuning_hyperparameter(yields, macros, tau_n, rho)
```

When using the function, T by N matrix "yields" and T by M matrix "macros" should contain initial observations ($t$ = 0, -1, -2, $\cdots$).

You can maximize the model selection criterion (marginal likelihood) directly if you want to. The objective function is

```juila
log_marginal(PCs, macros, rho, tuned::HyperParameter; medium_tau = 12 * [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
```

Here, the objective is maximized over "tuned", and initial observations also should be included. "PCs" are the first, second, and third principal components of yields. We have a function for the principal component analysis.

```juila
PCs, OCs, Wₚ, Wₒ = PCA(yields, p; rescaling=true)
```

The function uses eigenVectors of cov(yields[p+1:end,:]) to transform yields[1:end, :] to PCs. When rescaling = true, standard deviations of all PCs are normalized to an average of standard deviations of yields. Here, PCs and OCs are the first three and remaining principal components, respectively. Also, PCs[t, :] = Wₚ $\times$ yields[t, :] and OCs[t, :] = Wₒ $\times$ yields[t, :] hold.

# Step 2. sampling the posterior distribution of GDTSM

```juila
saved_params, acceptPr_C_varFF, acceptPr_ηψ = posterior_sampler(yields, macros, tau_n, rho, iteration, tuned::HyperParameter; sparsity=false, medium_tau=12 * [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5], init_param=[])
```

When using the function, T by N matrix "yields" and T by M matrix "macros" should contain initial observations ($t$ = 0, -1, -2, $\cdots$). tau*n is a vector that contains observed maturities of "yields". "Iteration" is the number of Gibbs sampling samples. Function "posterior*sampler" generate a vector of struct "Parameter"s that contains posterior samples. The second and third outputs say an MH acceptance probability of { $\phi*{i}$, $σ²*{FF,i}$: $i = 1$, $\cdots$, $d_\mathbb{Q}$ } and ηψ, respectively.

When "sparsity = true", we introduce additional Normal-Gamma(NG) priors on the intercepts and slopes while maintaining the Minnesota prior (Chan, 2021). The NG prior leads to the Generalized Inverse Gaussian posterior distribution. To sample this posterior, we use R package "GIGrvg" (Hörmann and Leydold, 2014).

We provide a default starting point for the sampler. However, if you want to set it, use keyward "init_param" that should be struct "Parameter".
