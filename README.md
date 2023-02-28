# GDTSM.jl

[![Build Status](https://github.com/econPreference/GDTSM.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/econPreference/GDTSM.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/econPreference/GDTSM.jl/branch/main/graph/badge.svg?token=j1mtPiDTgF)](https://codecov.io/gh/econPreference/GDTSM.jl)

Gaussian Dynamic Term Structure Model (GDTSM) is a theoretical government bond model where the bond price satisfies the no-arbitrage condition. It is a Gaussian model because all shocks follow Normal distributions. **GDTSM.jl** is a package for estimating the GDTSM. I follow the three-factor GDTSM of Joslin, Singleton, and Zhu(JSZ, 2011).

The major features of the package are

* Bayesian Estimation with automatically tuned hyper-parameters in a data-driven way (including VAR(p) lag selection)
* Yield curve interpolation and fitting
* Decomposition of a bond yield into the expectation hypothesis component and the term premium component
* The capability of accommodating unspanned macro risks
* Scenario Analyses and unconditional forecasts under the large-scale VAR framework to inspect interactions between bond yields and the macroeconomy

## Model

We follow the JSZ form, but with the restriction that Q-eigenvalue is [1, exp(-$\kappa^\mathbb{Q}$), exp(-$\kappa^\mathbb{Q}$)]. In this case, $\kappa^\mathbb{Q}$ is statistically equivalent to the decay parameter of the Dynamic Nelson-Siegel model(Diebold and Li, 2006). That is, our restricted JSZ model is statistically equivalent to the AFNS model (Christensen, Diebold, and Rudebusch, 2011).

Despite the AFNS restriction, our theoretical model sustains the JSZ form. The latent factors in our JSZ model are transformed into the principal components. And then, we estimate the model with the transformed state space as the JSZ did. One major difference between JSZ and ours is that we use the Bayesian methodology. For details, see our paper.

## Estimation

Note that all yield data should be annual percentage data (i.e. yield data = 1200$\times$theoretical yield).

### Step1. Tuning hyper-parameters

We have four hyper-parameters, $p$, $q$, $\nu_0$, and $\Omega_0$.

* $p$(Float64): lag of the VAR(p) in $\mathbb{P}$ -measure
* $q$(Vector) $=$ [$q_1$; $q_2$; $q_3$; $q_4$]: Minnesota prior
  * Prior variances of slopes $\propto$ $q_1$/lag$^{q_3}$ (for own lagged variables) or $q_2$/lag$^{q_3}$ (for cross lagged variables)
  * Prior variances of intercepts $\propto$ $q_4$
* $\nu_0$(Float64), $\Omega_0$(Vector): Error covariance of VAR(p) $\backsim$ InverseWishart($\nu_0$, $\Omega_0$)

We have additional two hyper-parameters that can be decided more objectively.

* $\rho$(Vector): ρ is a vector that has a size of size(macros,2). If $i$'th variable in macros is a growth(or level) variable, $\rho$[i] = 0(or $\approx$ 1) should be set.
* medium_τ(Vector): Candidate maturities where the curvature factor loading is maximized. The default value is [12, 18, 24, 30, 36, 42, 48, 54, 60] (months). When you estimate quarterly or annual data, this value should be modified.

struct "HyperParameter($p$, $q$, $\nu_0$, $\Omega_0$)" contains hyper-parameter values. We have a function "tuning_hyperparameter" that generates struct "HyperParameter" in a data-driven way.

```juila
tuned = tuning_hyperparameter(yields, macros, ρ)
```

When using the function, T by N matrix "yields" and T by M matrix "macros" should contain initial observations ($t$ = 0, -1, -2, $\cdots$).

You can maximize the model selection criterion (marginal likelihood) directly if you want to. The objective function is

```juila
log_marginal(PCs, macros, ρ, tuned::HyperParameter; medium_τ = 12 * [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]) 
```

Here, the objective is maximized over "tuned". "PCs" are the first, second, and third principal components of yields. We have a function for the principal component analysis.

```juila
PCs, OCs, Wₚ, Wₒ = PCA(yields, p; rescaling=true)
```

EigenVectors of cov(yields[p+1:end,:]) are used to transform yields[1:end,:] to PCs. When rescaling = true, standard deviations of all PCs are normalized to an average of standard deviations of yields.

### Step 2. sampling the posterior distribution of GDTSM

```juila
saved_θ = posterior_sampler(yields, macros, τₙ, ρ, iteration, tuned::HyperParameter; sparsity=false, medium_τ=12 * [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
```

When using the function, T by N matrix "yields" and T by M matrix "macros" should contain initial observations ($t$ = 0, -1, -2, $\cdots$). τₙ is a vector that contains observed maturities of "yields". "Iteration" is the number of Gibbs sampling samples.
When "sparsity = true", we introduce additional Normal-Gamma priors on the intercepts and slopes while maintaining the Minnesota prior (Chan, 2021).

## Inference

To call posterior samples of parameters, use [:name]. For example,

```juila
samples = saved_saved_θ[:κQ]
samples[i] # i'th posterior sample of κQ
```

The variable names in structs "Parameter" and "ReducedForm", and "LatentSpace" represent

* κQ: $\kappa^{\mathbb{Q}}$,
* kQ_infty: $k^{\mathbb{Q}}_{\infty}$,
* ϕ: {$\phi_{i}$; $i$ $=$ $1$, $\cdots$, $d_\mathbb{P}$},
* σ²FF: {$\sigma^2_{\mathcal{FF},i}$; $i$ $=$ $1$, $\cdots$, $d_\mathbb{P}$},
* ηψ: $\eta_{\psi}$,
* ψ: $d_\mathbb{P}$ by $p\cdot$$d_\mathbb{P}$ Matrix, [$[\psi_{1,i,j}]_{i,j}$ $\cdots$ $[\psi_{p,i,j}]_{i,j}$]
* ψ0: {$\psi_{0,i}$:$i=1$,$\cdots$, $d_\mathbb{P}$}
* Σₒ: $\Sigma_{\mathcal{O}}$
* γ: {$\gamma_i$:$i=1$,$\cdots$, N - $d_\mathbb{Q}$}
* KₚF: $K^\mathbb{P}_\mathcal{F}$
* GₚFF: [$G^\mathbb{P}_\mathcal{FF,1}$ $\cdots$ $G^\mathbb{P}_\mathcal{FF,p}$]
* ΩFF: $\Omega_\mathcal{FF}$
* λP: $\lambda_\mathcal{P}$
* ΛPF: [[$\Lambda_\mathcal{PP,1}$, $\Lambda_{\mathcal{P}M,1}$] $\cdots$ [$\Lambda_\mathcal{PP,p}$, $\Lambda_{\mathcal{P}M,p}$]]
* KₚXF: $K^\mathbb{P}_F$
* GₚXFXF: [$G^\mathbb{P}_{FF,1}$ $\cdots$ $G^\mathbb{P}_{FF,p}$]
* ΩXFXF: $\Omega_{FF}$

in our paper. Parameters in "ReducedForm" and "LatentSpace" can be deduced by using functions reducedform and latentspace, respectively.

We also support mean(), var(), std(), median(), quantile(), so for example

```juila
mean(saved_θ)[:kQ_infty]
```

gives the corresponding posterior mean. All functions, [:name], $\cdots$, quantile(), can be run on five structs, that are "Parameter", "ReducedForm" "LatentSpace", "TermPremium", and "Scenario".

## Introducing a sparsity the error covariance matrix
```juila
sparse_θ, trace_λ, trace_sparsity = sparse_precision(saved_θ, yields, macros, τₙ)
```

## Yield curve interpolation

## Term premium

## Scenario Analysis

## Citation

* Joslin, S., Singleton, K. J., and Zhu, H. (2011), “A new perspective on Gaussian dynamic term structure models,” The Review of Financial Studies, Oxford University Press, 24, 926–970.
* Diebold, F. X., and Li, C. (2006), “Forecasting the term structure of government bond yields,” Journal of econometrics, Elsevier, 130, 337–364.
* Christensen, J. H. E., Diebold, F. X., and Rudebusch, G. D. (2011), “The affine arbitrage-free class of Nelson – Siegel term structure models,” Journal of Econometrics, Elsevier B.V., 164, 4–20. <https://doi.org/10.1016/j.jeconom.2011.02.011>.
* Chan, J. C. C. (2021), “Minnesota-type adaptive hierarchical priors for large Bayesian VARs,” International Journal of Forecasting, Elsevier, 37, 1212–1226. <https://doi.org/10.1016/J.IJFORECAST.2021.01.002>.
