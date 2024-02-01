# TermStructureModels.jl

**NOTE. As I am currently in the job market, the completion of the documentation has been delayed. I will finish the work in February. Thanks.**

#### Documentation: [![](https://img.shields.io/badge/docs-latest-blue.svg)](https://econpreference.github.io/TermStructureModels.jl/dev/)

[![Build Status](https://github.com/econPreference/GDTSM.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/econPreference/GDTSM.jl/actions/workflows/CI.yml?query=branch%3Amain)

Gaussian Dynamic Term Structure Model (GDTSM) is a theoretical government bond model where the bond price satisfies the no-arbitrage condition. It is a Gaussian model because all shocks follow Normal distributions. **GDTSM.jl** is a package for estimating the GDTSM. I follow the three-factor GDTSM of Joslin, Singleton, and Zhu (JSZ, 2011).

The **main features** of the package are

- Bayesian Estimation with automatically tuned hyper-parameters in a data-driven way (including VAR(p) lag selection)
- Yield curve interpolation and fitting
- Decomposition of a bond yield into the expectation hypothesis component and the term premium component
- The capability of accommodating unspanned macro risks
- Scenario Analyses and unconditional forecasts under the large-scale VAR framework to inspect interactions between bond yields and the macroeconomy

If you have any suggestions, please feel free to ask me by raising issues.

## Prerequisites

Since we use two R packages (GIGrvg, glasso), users have to install R language. If you already have experience using RCall.jl, you just need to install the two R Packages. If it is the first time using RCall.jl, follow the below steps.

1. Install R on your computer from the internet.
2. In R, run the below command and copy the home address.

```R
R.home()
```

3. In R, run the below code to install the packages.

```R
install.packages("GIGrvg")
install.packages("glasso")
```

4. In Juila, run

```juila
ENV["R_HOME"]=""
```

5. In Juila, run

```juila
ENV["PATH"]="...the address in step 2..."
```

6. In Juila, run

```juila
using Pkg
Pkg.add("RCall")
```

## Model

We follow the JSZ form, but with the restriction that Q-eigenvalues are [1, exp( $- \kappa^\mathbb{Q}$), exp( $- \kappa^\mathbb{Q}$)]. In this case, $\kappa^\mathbb{Q}$ is statistically equivalent to the decay parameter of the Dynamic Nelson-Siegel model (Diebold and Li, 2006). That is, our restricted JSZ model is statistically equivalent to the AFNS model (Christensen, Diebold, and Rudebusch, 2011).

Despite the AFNS restriction, our theoretical model sustains the JSZ form. The latent factors in our JSZ model are transformed into the principal components. And then, we estimate the model with the transformed state space as the JSZ did. One major difference between JSZ and ours is that we use the Bayesian methodology. For details, see our paper.

## Estimation

**Note that all yield data should be annual percentage data (i.e. yield data = 1200 $\times$ theoretical yield in the model).**

### Step1. Tuning hyper-parameters

We have four hyper-parameters, $p$, $q$, $\nu_0$, and $\Omega_0$.

- $p$(Float64): lag of the VAR(p) in $\mathbb{P}$ -measure
- $q$(Vector) $=$ [ $q_1$; $q_2$; $q_3$; $q_4$]: Minnesota prior
  - Prior variances of slopes $\propto$ $q_1$/ ${lag}^{q_3}$ (for own lagged variables) or $q_2$/ ${lag}^{q_3}$ (for cross lagged variables)
  - Prior variances of intercepts $\propto$ $q_4$
- $\nu_0$(Float64), $\Omega_0$(Vector): Error covariance of VAR(p) $\backsim$ InverseWishart($\nu_0$, $\Omega_0$)

We have additional two hyper-parameters that can be decided more objectively.

- $\rho$(Vector): ρ is a vector that has a size of size(macros,2). If $i$'th variable in macros is a growth(or level) variable, $\rho$[i] = 0(or $\approx$ 1) should be set.
- medium_τ(Vector): Candidate maturities where the curvature factor loading is maximized. The default value is [12, 18, 24, 30, 36, 42, 48, 54, 60] (months). When you estimate quarterly or annual data, this value should be modified.

Struct "HyperParameter($p$, $q$, $\nu_0$, $\Omega_0$)" contains hyper-parameter values. We have a function "tuning_hyperparameter" that generates struct "HyperParameter" in a data-driven way (Chan, 2022).

```juila
tuned = tuning_hyperparameter(yields, macros, ρ)
```

When using the function, T by N matrix "yields" and T by M matrix "macros" should contain initial observations ($t$ = 0, -1, -2, $\cdots$).

You can maximize the model selection criterion (marginal likelihood) directly if you want to. The objective function is

```juila
log_marginal(PCs, macros, ρ, tuned::HyperParameter; medium_τ = 12 * [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
```

Here, the objective is maximized over "tuned", and initial observations also should be included. "PCs" are the first, second, and third principal components of yields. We have a function for the principal component analysis.

```juila
PCs, OCs, Wₚ, Wₒ = PCA(yields, p; rescaling=true)
```

The function uses eigenVectors of cov(yields[p+1:end,:]) to transform yields[1:end, :] to PCs. When rescaling = true, standard deviations of all PCs are normalized to an average of standard deviations of yields. Here, PCs and OCs are the first three and remaining principal components, respectively. Also, PCs[t, :] = Wₚ $\times$ yields[t, :] and OCs[t, :] = Wₒ $\times$ yields[t, :] hold.

### Step 2. sampling the posterior distribution of GDTSM

```juila
saved_θ, acceptPr_C_σ²FF, acceptPr_ηψ = posterior_sampler(yields, macros, τₙ, ρ, iteration, tuned::HyperParameter; sparsity=false, medium_τ=12 * [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5], init_param=[])
```

When using the function, T by N matrix "yields" and T by M matrix "macros" should contain initial observations ($t$ = 0, -1, -2, $\cdots$). τₙ is a vector that contains observed maturities of "yields". "Iteration" is the number of Gibbs sampling samples. Function "posterior*sampler" generate a vector of struct "Parameter"s that contains posterior samples. The second and third outputs say an MH acceptance probability of { $\phi*{i}$, $σ²_{FF,i}$: $i = 1$, $\cdots$, $d_\mathbb{Q}$ } and ηψ, respectively.

When "sparsity = true", we introduce additional Normal-Gamma(NG) priors on the intercepts and slopes while maintaining the Minnesota prior (Chan, 2021). The NG prior leads to the Generalized Inverse Gaussian posterior distribution. To sample this posterior, we use R package "GIGrvg" (Hörmann and Leydold, 2014).

We provide a default starting point for the sampler. However, if you want to set it, use keyward "init_param" that should be struct "Parameter".

## Inference

To call posterior samples of objects in structs ("Parameter", "ReducedForm" "LatentSpace", "YieldCurve", "TermPremium", and "Forecast"), use [:name]. For example, for output "saved_θ" of function "posterior sampler",

```juila
samples = saved_θ[:κQ]
samples[i] # i'th posterior sample of κQ
```

The variable names in structs "Parameter", "ReducedForm", and "LatentSpace" represent

- κQ: $\kappa^{\mathbb{Q}}$,
- kQ*infty: $k^{\mathbb{Q}}*{\infty}$,
- ϕ: { $\phi_{i}$; $i$ $=$ $1$, $\cdots$, ${d}_{\mathbb{P}}$ },
- σ²FF: { $\sigma^2_{\mathcal{FF},i}$ ; $i$ $=$ $1$, $\cdots$, $d_\mathbb{P}$ },
- ηψ: $\eta_{\psi}$,
- ψ: $d_\mathbb{P}$ by ${p}{\cdot}$ $d_{\mathbb{P}}$ Matrix, [[ $\psi_{1,i,j}$ ] $\cdots$ [ $\psi_{p,i,j}$ ] ]
- ψ0: { $\psi_{0,i}$ : $i=1$, $\cdots$, $d_\mathbb{P}$ }
- Σₒ: $\Sigma_{\mathcal{O}}$
- γ: { $\gamma_i$ : $i=1$, $\cdots$, N - $d_\mathbb{Q}$ }
- KₚF: $K^\mathbb{P}_\mathcal{F}$
- GₚFF: [ $G^P_{\mathcal{FF},1}$ $\cdots$ $G^P_{\mathcal{FF},p}$ ]
- ΩFF: $\Omega_\mathcal{FF}$
- λP: $\lambda_\mathcal{P}$
- ΛPF: [[$\Lambda_{\mathcal{PP},1}$, $\Lambda_{\mathcal{P}M,1}$] $\cdots$ [ $\Lambda_{\mathcal{PP},p}$, $\Lambda_{\mathcal{P}M,p}$]]
- KₚXF: $K^\mathbb{P}_F$
- GₚXFXF: [ $G^P_{FF,1}$ $\cdots$ $G^P_{FF,p}$ ]
- ΩXFXF: $\Omega_{FF}$

in our paper. Parameters in "ReducedForm" and "LatentSpace" can be deduced by using functions "reducedform" and "latentspace", respectively. "ReducedForm" contains the reduced form VAR(p) parameters. "LatentSpace" contains parameters when our model is expressed in terms of latent factor $X_t$

We support mean(), var(), std(), median(), quantile() in Statistics.jl. So, for example, when we need a posterior mean,

```juila
mean(saved_θ)[:kQ_infty]
```

gives the corresponding posterior mean of kQ_infty. All functions, [:name], $\cdots$, quantile(), can be run on six structs, which are "Parameter", "ReducedForm" "LatentSpace", "YieldCurve", "TermPremium", and "Forecast".

## Structs in the packages

To see names of objects in the structs, run, for example,

```juila
help?>YieldCurve
```

We have eight structs, which are **HyperParameter**, **Parameter**, **ReducedForm**, **LatentSpace**, **YieldCurve**, **TermPremium**, **Scenario**, and **Forecast**. It also provides details of the structs.

## Introducing a sparsity on error precision matrix

```juila
sparse_θ, trace_λ, trace_sparsity = sparse_prec(saved_θ::Vector{Parameter}, yields, macros, τₙ)
```

It introduces a sparsity on the error precision matrix of VAR(p) P-dynamics using Freidman, Hastie, and Tibshirani (2008) and Hauzenberger, Huber, and Onorante (2021). We use R-package "glasso" to implement it. Specifically, the additionally introduced lasso penalty makes some small elements in the precision to zero.

Here, the data should contain initial observations. τₙ is a vector that contains observed maturities of "yields". "saved*θ" is an output of function "posterior sampler". For the outputs, "sparse*θ" is also a vector of struct "Parameter" but has sparse precision matrices. "trace_λ" and "trace_sparsity" contain the used optimal penalty parameter and the number of non-zero elements of the precision.

## Yield curve interpolation

```juila
fitted = fitted_YieldCurve(τ0, saved_Xθ::Vector{LatentSpace})
```

To derive the fitted yield curve, you first derive "saved_Xθ" from function "latentspace". τ0 is a vector that contains maturities of interest. The output is Vector{"YieldCurve"}.

## Term premium

```juila
saved_TP = term_premium(τ, τₙ, saved_θ::Vector{Parameter}, yields, macros)
```

The function calculates term premium estimates of maturity τ (months). Here, τ does not need to be the one in τₙ. "τₙ", "yields", and "macros" are the things that were inputs of function "posterior sampler".
"saved_θ" is the output of function "posterior sampler". Output "saved_TP" is Vector{TermPremium}.

## Scenario Analysis and unconditional forecasts

```juila
prediction = scenario_sampler(S::Scenario, τ, horizon, saved_θ, yields, macros, τₙ)
```

The function generates (un)conditional forecasts using our model. We use the Kalman filter to make conditional filtered forecasts (Bańbura, Giannone, and Lenza, 2015), and then we use Kim and Nelson (1999) to make smoothed posterior samples of the conditional forecasts. "S" is a conditioned scenario, and yields, risk factors, and a term premium of maturity "τ" are forecasted. "horizon" is a forecasting horizon. "τₙ", "yields", and "macros" are the things that were inputs of function "posterior sampler". "saved_θ" is an output of function "posterior sampler". The output is Vector{Forecast}.

Struct Scenario has two elements, "combinations" and "values". Meaning of the struct can be found by help? command. Examples of making struct "Scenario" are as follows.

```juila
# Case 1. Unconditional Forecasts
S = []

# Case 2. Scenario with one conditioned variable and time length 2
comb = zeros(1, size([yields macros], 2))
comb[1, 1] = 1.0 # one month yield is selected as a conditioned variable
values = [3.0] # Scenario: one month yield at time T+1 is 3.0
S = Scenario(combinations=comb, values=values)

# Case 3. Scenario with two conditioned combinations and time length 3
comb = zeros(2, size([yields macros], 2), 3)
values = zeros(2, 3)
for t in 1:3 # for simplicity, we just assume the same scenario for time = T+1, T+2, T+3. Users can freely assume different scenarios for each time T+t.
  comb[1, 1, t] = 1.0 # one month yield is selected as a conditioned variable in the first combination
  comb[2, 20, t] = 0.5
  comb[2, 21, t] = 0.5 # the average of 20th and 21st observables is selected as a second conditioned combination
  values[1,t] = 3.0 # one month yield at time T+t is 3.0
  values[2,t] = 0.0 # the average value is zero.
end
S = Scenario(combinations=comb, values=values)
```

Here, **both "combinations" and "values" should be type Array{Float64}**. Also, "horizon" should not be smaller than size(values, 2).

## Citation

- Joslin, S., Singleton, K. J., and Zhu, H. (2011), “A new perspective on Gaussian dynamic term structure models,” The Review of Financial Studies, Oxford University Press, 24, 926–970.
- Diebold, F. X., and Li, C. (2006), “Forecasting the term structure of government bond yields,” Journal of econometrics, Elsevier, 130, 337–364.
- Christensen, J. H. E., Diebold, F. X., and Rudebusch, G. D. (2011), “The affine arbitrage-free class of Nelson – Siegel term structure models,” Journal of Econometrics, Elsevier B.V., 164, 4–20. <https://doi.org/10.1016/j.jeconom.2011.02.011>.
- Chan, J. C. (2022), “Asymmetric Conjugate Priors for Large Bayesian VARs,” Quantitative Economics. <https://doi.org/10.2139/ssrn.3424437>.
- Chan, J. C. C. (2021), “Minnesota-type adaptive hierarchical priors for large Bayesian VARs,” International Journal of Forecasting, Elsevier, 37, 1212–1226. <https://doi.org/10.1016/J.IJFORECAST.2021.01.002>.
- Hörmann, W., and Leydold, J. (2014), “Generating generalized inverse Gaussian random variates,” Statistics and Computing, 24, 547–557. <https://doi.org/10.1007/s11222-013-9387-3>.
- Friedman, J., Hastie, T., and Tibshirani, R. (2008), “Sparse inverse covariance estimation with the graphical lasso,” Biostatistics, 9, 432–441. <https://doi.org/10.1093/biostatistics/kxm045>.
- Hauzenberger, N., Huber, F., and Onorante, L. (2021), “Combining shrinkage and sparsity in conjugate vector autoregressive models,” Journal of Applied Econometrics, n/a. <https://doi.org/10.1002/jae.2807>.
- Bańbura, M., Giannone, D., and Lenza, M. (2015), “Conditional forecasts and scenario analysis with vector autoregressions for large cross-sections,” International Journal of Forecasting, 31, 739–756. <https://doi.org/10.1016/j.ijforecast.2014.08.013>.
- Kim, C.-J., and Nelson, C. R. (2017), State-space models with regime switching: Classical and gibbs-sampling approaches with applications, The MIT Press. <https://doi.org/10.7551/mitpress/6444.001.0001>.

## To do list

1. make CITATION.bib
