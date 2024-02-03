# TermStructureModels.jl

**NOTE. As I am currently in the job market, the completion of the documentation has been delayed. I will finish the work in February. Thanks.**

No-Arbitrage Term Structure Models are theoretical government bond models where the bond price satisfies the no-arbitrage condition. It is a Gaussian model because all shocks follow Normal distributions. **GDTSM.jl** is a package for estimating the GDTSM. I follow the three-factor GDTSM of Joslin, Singleton, and Zhu (JSZ, 2011).

The **main features** of the package are

- Bayesian Estimation with automatically tuned hyper-parameters in a data-driven way (including VAR(p) lag selection)
- Yield curve interpolation and fitting
- Decomposition of a bond yield into the expectation hypothesis component and the term premium component
- The capability of accommodating unspanned macro risks
- Scenario Analyses and unconditional forecasts under the large-scale VAR framework to inspect interactions between bond yields and the macroeconomy

If you have any suggestions, please feel free to ask me by raising issues.
