using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
Pkg.precompile()
using GDTSM, ProgressMeter, StatsBase, Dates
using CSV, DataFrames, LinearAlgebra, Gadfly, XLSX
using Cairo, Fontconfig, Colors, LaTeXStrings, Distributions
import Plots, JLD2
import StatsPlots: @df

## Data setting
upper_p = 18
date_start = Date("1987-01-01", "yyyy-mm-dd") |> x -> x - Month(upper_p + 2)
date_end = Date("2022-12-01", "yyyy-mm-dd")
τₙ = [1; 3; 6; 9; collect(12:6:60); collect(72:12:120)]
medium_τ = collect(36:42)

function data_loading(; date_start, date_end, τₙ)

    ## Yield data
    raw_yield = XLSX.readdata("LW_monthly.xlsx", "Sheet1", "A293:DQ748") |> x -> [Date.(string.(x[:, 1]), DateFormat("yyyymm")) convert(Matrix{Float64}, x[:, τₙ.+1])] |> x -> DataFrame(x, ["date"; ["Y$i" for i in τₙ]])
    yields = raw_yield[findall(x -> x == yearmonth(date_start), yearmonth.(raw_yield[:, 1]))[1]:findall(x -> x == yearmonth(date_end), yearmonth.(raw_yield[:, 1]))[1], :]
    yields = yields[3:end, :]

    yields = [Date.(string.(yields[:, 1]), DateFormat("yyyy-mm-dd")) Float64.(yields[:, 2:end])]
    rename!(yields, Dict(:x1 => "date"))

    return yields

end
yields = data_loading(; date_start, date_end, τₙ)

## Setting
# optimization
upper_q =
    [1 1
        1 1
        10 10
        100 100] .+ 0.0
μkQ_infty = 0
σkQ_infty = 0.02
init_ν0 = 39

# estimation
iteration = 15_000
burnin = 5_000
TPτ_interest = 120

function estimation(; upper_p, τₙ, medium_τ, iteration, burnin, yields, μkQ_infty, σkQ_infty)

    tuned = JLD2.load("nomacro/tuned.jld2")["tuned"]
    p = tuned.p

    saved_θ, acceptPrMH = posterior_sampler(Array(yields[upper_p-p+1:end, 2:end]), [], τₙ, [], iteration, tuned; medium_τ, μkQ_infty, σkQ_infty)
    saved_θ = saved_θ[burnin+1:end]
    iteration = length(saved_θ)

    saved_θ, Pr_stationary = erase_nonstationary_param(saved_θ)
    iteration = length(saved_θ)
    JLD2.save("nomacro/posterior.jld2", "samples", saved_θ, "acceptPrMH", acceptPrMH, "Pr_stationary", Pr_stationary)

    saved_TP = term_premium(TPτ_interest, τₙ, saved_θ, Array(yields[upper_p-p+1:end, 2:end]), [])
    JLD2.save("nomacro/TP.jld2", "TP", saved_TP)

    return []
end

## Do

tuned, opt = tuning_hyperparameter(Array(yields[:, 2:end]), [], τₙ, []; upper_p, upper_q, μkQ_infty, σkQ_infty, medium_τ, init_ν0)
JLD2.save("nomacro/tuned.jld2", "tuned", tuned, "opt", opt)

estimation(; upper_p, τₙ, medium_τ, iteration, burnin, yields, μkQ_infty, σkQ_infty)
