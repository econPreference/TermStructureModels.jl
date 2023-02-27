# GDTSM.jl

[![Build Status](https://github.com/econPreference/GDTSM.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/econPreference/GDTSM.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/econPreference/GDTSM.jl/branch/main/graph/badge.svg?token=j1mtPiDTgF)](https://codecov.io/gh/econPreference/GDTSM.jl)

Gaussian Dynamic Term Structure Model (GDTSM) is a theoretical government bond model where the bond price satisfies the no-arbitrage condition. It is a Gaussian model because all shocks follow Normal distributions.

**GDTSM.jl** is a package for estimating the GDTSM. I follow three-factor GDTSM of Joslin, Singleton, and Zhu(JSZ, 2011). The major features of the package are

* Bayesian Estimation with automatically tuned hyper-parameters in a data-driven way (including VAR(p) lag selection)
* Yield curve interpolation and fitting
* Decomposition of a bond yield into the expectation hypothesis component and the term premium component
* The capability of accommodating unspanned macro risks
* Scenario Analyses and unconditional forecasts under the large-scale VAR framework to inspect interactions between bond yields and the macroeconomy

## Model

* One-month yield
  * $r_t = \iota'X_t$
  * $\iota$: vector of ones
  * $X_t$: bond market latent factor vector
* Risk Neutral measure (Q-measure)
  * $X_t = K^{\mathbb{Q}}_X + G^{\mathbb{Q}}_{XX}X_{t-1}+\epsilon^{\mathbb{Q}}_{X,t}$
  * $K^{\mathbb{Q}}_X$: Vector, the first element is $k^{\mathbb{Q}}_{\infty}$, and the other elements are zeros.
  * $G^{\mathbb{Q}}_{XX}$ = $\begin{pmatrix}1 & 0 & 0\\0 & \exp[-\kappa^{\mathbb{Q}}] & 1\\0 & 0 & \exp[-\kappa^{\mathbb{Q}}]\end{pmatrix},$
  * $\epsilon^{\mathbb{Q}}_{X,t}$: Normal(O, $\Omega_{XX}$)
* Physical(Actual) measure (P-measure)
  * $F_t = K^{\mathbb{P}}_{F} + \sum_{l=1}^pG^{\mathbb{P}}_{FF,l}F_{t-l}+\epsilon^{\mathbb{P}}_{F,t}$
  * $\epsilon^{\mathbb{P}}_{F,t}$: Normal(O,$\Omega_{FF}$)
  * $K^{\mathbb{P}}_{F}, G^{\mathbb{P}}_{FF,l},\Omega_{FF}$ : Full vector and matrices
* As JSZ did, we rotate the latent factor space into the principal component space. And then, 

## Citation

Joslin, S., Singleton, K. J., and Zhu, H. (2011), “A new perspective on Gaussian dynamic term structure models,” The Review of Financial Studies, Oxford University Press, 24, 926–970.
