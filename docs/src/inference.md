## Inference for Parameters

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
