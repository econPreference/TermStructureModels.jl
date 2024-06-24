## Setting
using Pkg, Revise
Pkg.activate(@__DIR__)
Pkg.instantiate()
Pkg.precompile()
using TermStructureModels, ProgressMeter, Distributions, LinearAlgebra, Distributions
using CSV, DataFrames, XLSX, StatsBase, Dates, JLD2


## Data setting
upper_p = 18
date_start = Date("1987-01-01", "yyyy-mm-dd") |> x -> x - Month(upper_p + 2)
date_end = Date("2022-12-01", "yyyy-mm-dd")
tau_n = [1; 3; 6; 9; collect(12:6:60); collect(72:12:120)]
medium_tau = collect(36:42)

function data_loading(; date_start, date_end, tau_n)

    ## Yield data
    raw_yield = XLSX.readdata("LW_monthly.xlsx", "Sheet1", "A293:DQ748") |> x -> [Date.(string.(x[:, 1]), DateFormat("yyyymm")) convert(Matrix{Float64}, x[:, tau_n.+1])] |> x -> DataFrame(x, ["date"; ["Y$i" for i in tau_n]])
    yields = raw_yield[findall(x -> x == yearmonth(date_start), yearmonth.(raw_yield[:, 1]))[1]:findall(x -> x == yearmonth(date_end), yearmonth.(raw_yield[:, 1]))[1], :]
    yields = yields[3:end, :]

    yields = [Date.(string.(yields[:, 1]), DateFormat("yyyy-mm-dd")) Float64.(yields[:, 2:end])]
    rename!(yields, Dict(:x1 => "date"))

    return yields

end
yields = data_loading(; date_start, date_end, tau_n)

## Setting
# optimization
upper_q =
    [1 1
        1 1
        10 10
        100 100] .+ 0.0
std_kQ_infty = 0.2

# estimation
iteration = 5_000
burnin = 1_000

function estimation(; upper_p, tau_n, medium_tau, iteration, burnin, yields, std_kQ_infty)

    tuned = JLD2.load("tuned.jld2")["tuned"]
    p = tuned.p

    saved_θ, acceptPrMH = posterior_sampler(Array(yields[upper_p-p+1:end, 2:end]), [], tau_n, [], iteration, tuned; medium_tau, std_kQ_infty)
    saved_θ = saved_θ[burnin+1:end]
    iteration = length(saved_θ)

    saved_θ, Pr_stationary = erase_nonstationary_param(saved_θ)
    iteration = length(saved_θ)
    JLD2.save("posterior.jld2", "samples", saved_θ, "acceptPrMH", acceptPrMH, "Pr_stationary", Pr_stationary)

    ineff = ineff_factor(saved_θ)
    JLD2.save("ineff.jld2", "ineff", ineff)

    iter_sub = 2
    saved_TP = term_premium(120, tau_n, saved_θ[1:iter_sub:end], Array(yields[upper_p-p+1:end, 2:end]), [])
    JLD2.save("TP.jld2", "TP", saved_TP)

    return []
end

## Do

tuned, opt = tuning_hyperparameter(Array(yields[:, 2:end]), [], tau_n, []; upper_p=1, upper_q, std_kQ_infty, medium_tau)
JLD2.save("tuned.jld2", "tuned", tuned, "opt", opt)

estimation(; upper_p, tau_n, medium_tau, iteration, burnin, yields, std_kQ_infty)
