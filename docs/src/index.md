# TermStructureModels.jl

_TermStructureModels.jl_ is a `Julia` package to estimate the term structure of interest rates. We currently provide an Gaussian affine term structure model that satisfies the No-Arbitrage condition.

Our model is three-factor JSZ(Joslin, Singleton, and Zhu, 2011) model constrained by the AFNS(Christensen, Diebold, and Rudebusch, 2011) restriction. [Our paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4708628) provides theoretical descriptions of our model.

The **main features** of the package is that our package allows for the lag length of the VAR system in the Physical measure to extend beyond one. Additionally, it permits the inclusion of numerous macroeconomic variables within the Physical measure in the form of unspanned risk. For instance, in our study, within the term-structure model that incorporates 28 macroeconomic variables, we set the lag length to 17.

**Other features** of the package include

- Estimation of the model in the Bayesian framework
- All hyperparameters, including the lag length of the VAR system, are automatically determined by the data
- Yield curve interpolation and fitting
- Decomposition of a bond yield into the expectation hypothesis component and the term premium component
- Conditional Forecasting,including Scenario Analyses, to inspect interactions between bond yields and the macroeconomy

## Installation

Type

```julia
using Pkg
Pkg.add("TermStructureModels")
```

in the julialang REPL.

## Usage

Type

```julia
using TermStructureModels
```

to load all functions of our package.
