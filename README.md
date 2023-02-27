# GDTSM.jl

[![Build Status](https://github.com/econPreference/GDTSM.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/econPreference/GDTSM.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/econPreference/GDTSM.jl/branch/main/graph/badge.svg?token=j1mtPiDTgF)](https://codecov.io/gh/econPreference/GDTSM.jl)

Gaussian Dynamic Term Structure Model (GDTSM) is a theoretical government bond model where the bond price satisfies the no-arbitrage condition. It is a Gaussian model because all shocks follow Normal distributions. **GDTSM.jl** is a package for estimating the GDTSM. I follow three-factor GDTSM of Joslin, Singleton, and Zhu(JSZ, 2011).

The major features of the package are

* Bayesian Estimation with automatically tuned hyper-parameters in a data-driven way (including VAR(p) lag selection)
* Yield curve interpolation and fitting
* Decomposition of a bond yield into the expectation hypothesis component and the term premium component
* The capability of accommodating unspanned macro risks
* Scenario Analyses and unconditional forecasts under the large-scale VAR framework to inspect interactions between bond yields and the macroeconomy

## Model

We basically follow the JSZ form, but with the restriction that Q-eigenvalue is [1, exp(-$\kappa^\mathbb{Q}$), exp(-$\kappa^\mathbb{Q}$)]. In this case, $\kappa^\mathbb{Q}$ is statistically equivalent to the decay parameter of the Dynamic Nelson-Siegel model(Diebold and Li, 2006). That is, our restricted JSZ model is statistically equivalent to the AFNS model (Christensen, Diebold, and Rudebusch, 2011).

Despite the AFNS restriction, our theoretical model sustains the JSZ form. The latent factors in our JSZ model are transformed into the principal components. And then, we estimate the model with the transformed state space as the JSZ did. One major difference between JSZ and ours is that we use the Bayesian methodology. For details, see our paper.

## Estimation

### Step1. Tuning hyper-parameters

We have four hyper-parameters, $p$, $q$, $\nu_0$, and $\Omega_0$.

* $p$(Float64): lag of the VAR(p) in $\mathbb{P}$ -measure
* $q$(Vector) $=$ [$q_1$; $q_2$; $q_3$; $q_4$]: Minnesota prior
  * Prior variances of slopes $\propto$ $q_1$/lag$^{q_3}$ (for own lagged variables) or $q_2$/lag$^{q_3}$ (for cross lagged variables)
  * Prior variances of intercepts $\propto$ $q_4$
* $\nu_0$(Float64), $\Omega_0$(Vector): Error covariance of VAR(p) $\backsim$ InverseWishart($\nu_0$, $\Omega_0$)

We have additional two hyper-parameters that can be decided in a more objective way.

* $\rho$(Vector): ρ is a vector that has a size of size(macros,2). If $i$'th variable in macros is a growth(or level) variable, $\rho$[i] = 0(or $\approx$ 1) should be set.
* medium_$\tau$(Vector): Candidate maturities where the curvature factor loading is maximized. The default value is [12, 18, 24, 30, 36, 42, 48, 54, 60]. When you estimate quarterly or annual data, this value should be modified.

struct "HyperParameter($p$, $q$, $\nu_0$, $\Omega_0$)" contains hyper-parameter values. We have a function "tuning_hyperparameter" that generates struct "HyperParameter" in a data-driven way.

```juila
tuned = tuning_hyperparameter(yields, macros, ρ)
```

When using the function, variables "yields" and "macros" should contain initial observations ($t$ = 0, -1, -2, $\cdots$).

You can maximize the model selection criterion (marginal likelihood) directly if you want to. The objective function is 

```juila
log_marginal(PCs, macros, ρ, tuned::HyperParameter; medium_τ = [12, 18, 24, 30, 36, 42, 48, 54, 60]) 
```
Here, the objective is maximized over "tuned". "PCs" are first, second, and third principal components of yields. 
## Citation

* Joslin, S., Singleton, K. J., and Zhu, H. (2011), “A new perspective on Gaussian dynamic term structure models,” The Review of Financial Studies, Oxford University Press, 24, 926–970.
* Diebold, F. X., and Li, C. (2006), “Forecasting the term structure of government bond yields,” Journal of econometrics, Elsevier, 130, 337–364.
* Christensen, J. H. E., Diebold, F. X., and Rudebusch, G. D. (2011), “The affine arbitrage-free class of Nelson – Siegel term structure models,” Journal of Econometrics, Elsevier B.V., 164, 4–20. <https://doi.org/10.1016/j.jeconom.2011.02.011>.
