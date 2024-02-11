# TermStructureModels.jl

#### Detailed explanations of how to use this package can be found in the documentation: Click [![](https://img.shields.io/badge/docs-latest-blue.svg)](https://econpreference.github.io/TermStructureModels.jl/dev/)

[![Build Status](https://github.com/econPreference/GDTSM.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/econPreference/GDTSM.jl/actions/workflows/CI.yml?query=branch%3Amain)

_TermStructureModels.jl_ is a `Julia` package to estimate the term structure of interest rates. We currently provide an Gaussian affine term structure model that satisfies the No-Arbitrage condition.

Our model is the three-factor JSZ[(Joslin, Singleton, and Zhu, 2011)](https://academic.oup.com/rfs/article-abstract/24/3/926/1590594) model constrained by the AFNS[(Christensen, Diebold, and Rudebusch, 2011)](https://www.sciencedirect.com/science/article/pii/S0304407611000388) restriction. [Our paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4708628) provides theoretical descriptions of our model.

The **main features** of the package is that it allows for the lag length of the VAR system in the Physical measure to extend beyond one. Additionally, it permits the inclusion of numerous macroeconomic variables as unspanned risks within the Physical measure. For instance, in our study, we set the lag length to 17 in the term-structure model that incorporates 28 macroeconomic variables.

**Other features** of the package include

- Estimation of the model in the Bayesian framework
- All hyperparameters, including the lag length of the VAR system, are automatically determined by the data
- Yield curve interpolation and fitting
- Decomposition of a bond yield into the expectation hypothesis component and the term premium component
- Conditional Forecasting, including Scenario Analyses, to inspect interactions between bond yields and the macroeconomy

## Installation

Run

```julia
using Pkg
Pkg.add("TermStructureModels")
```

## Usage

Run

```julia
using TermStructureModels
```

to load all functions of our package.

## Example File

To better understand how to use our package, refer to [the script file](https://github.com/econPreference/TermStructureModels.jl/blob/main/examples/LargeVAR_Yields_Macros/LargeVAR_Yields_Macros.ipynb) used for [our paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4708628).

## Citation

If you want to cite this package for your works, cite [our paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4708628).
