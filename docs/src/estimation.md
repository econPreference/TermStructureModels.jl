### Step1. Tuning hyper-parameters

We have four hyper-parameters, $p$, $q$, $\nu_0$, and $\Omega_0$.

- $p$(Float64): lag of the VAR(p) in $\mathbb{P}$ -measure
- $q$(Vector) $=$ [ $q_1$; $q_2$; $q_3$; $q_4$]: Minnesota prior
  - Prior variances of slopes $\propto$ $q_1$/ ${lag}^{q_3}$ (for own lagged variables) or $q_2$/ ${lag}^{q_3}$ (for cross lagged variables)
  - Prior variances of intercepts $\propto$ $q_4$
- $\nu_0$(Float64), $\Omega_0$(Vector): Error covariance of VAR(p) $\backsim$ InverseWishart($\nu_0$, $\Omega_0$)

We have additional two hyper-parameters that can be decided more objectively.

- $\rho$(Vector): rho is a vector that has a size of size(macros,2). If $i$'th variable in macros is a growth(or level) variable, $\rho$[i] = 0(or $\approx$ 1) should be set.
- medium_tau(Vector): Candidate maturities where the curvature factor loading is maximized. The default value is [12, 18, 24, 30, 36, 42, 48, 54, 60] (months). When you estimate quarterly or annual data, this value should be modified.

Struct "HyperParameter($p$, $q$, $\nu_0$, $\Omega_0$)" contains hyper-parameter values. We have a function "tuning_hyperparameter" that generates struct "HyperParameter" in a data-driven way (Chan, 2022).

```juila
tuned = tuning_hyperparameter(yields, macros, rho)
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

### Step 2. sampling the posterior distribution of GDTSM

```juila
saved_params, acceptPr_C_varFF, acceptPr_ηψ = posterior_sampler(yields, macros, tau_n, rho, iteration, tuned::HyperParameter; sparsity=false, medium_tau=12 * [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5], init_param=[])
```

When using the function, T by N matrix "yields" and T by M matrix "macros" should contain initial observations ($t$ = 0, -1, -2, $\cdots$). tau*n is a vector that contains observed maturities of "yields". "Iteration" is the number of Gibbs sampling samples. Function "posterior*sampler" generate a vector of struct "Parameter"s that contains posterior samples. The second and third outputs say an MH acceptance probability of { $\phi*{i}$, $σ²*{FF,i}$: $i = 1$, $\cdots$, $d_\mathbb{Q}$ } and ηψ, respectively.

When "sparsity = true", we introduce additional Normal-Gamma(NG) priors on the intercepts and slopes while maintaining the Minnesota prior (Chan, 2021). The NG prior leads to the Generalized Inverse Gaussian posterior distribution. To sample this posterior, we use R package "GIGrvg" (Hörmann and Leydold, 2014).

We provide a default starting point for the sampler. However, if you want to set it, use keyward "init_param" that should be struct "Parameter".
