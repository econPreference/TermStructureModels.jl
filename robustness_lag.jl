using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
Pkg.precompile()
using GDTSM, ProgressMeter, StatsBase, Dates
using CSV, DataFrames, LinearAlgebra, Gadfly, XLSX
using Cairo, Fontconfig, Colors, LaTeXStrings, Distributions, ColorSchemes
import Plots, JLD2
import StatsPlots: @df

## Data setting
upper_p = 18
date_start = Date("1987-01-01", "yyyy-mm-dd") |> x -> x - Month(upper_p + 2)
date_end = Date("2022-12-01", "yyyy-mm-dd")
τₙ = [1; 3; 6; 9; collect(12:6:60); collect(72:12:120)]
medium_τ = collect(36:42)

function data_loading(; date_start, date_end, τₙ)

    ## Macro data
    raw_fred = CSV.File("current.csv") |> DataFrame |> x -> x[314:769, :]
    raw_fred = [Date.(raw_fred[:, 1], DateFormat("mm/dd/yyyy")) raw_fred[:, 2:end]]
    raw_fred = raw_fred[findall(x -> x == yearmonth(date_start), yearmonth.(raw_fred[:, 1]))[1]:findall(x -> x == yearmonth(date_end), yearmonth.(raw_fred[:, 1]))[1], :]

    excluded = ["FEDFUNDS", "CP3Mx", "TB3MS", "TB6MS", "GS1", "GS5", "GS10", "TB3SMFFM", "TB6SMFFM", "T1YFFM", "T5YFFM", "T10YFFM", "COMPAPFFx", "AAAFFM", "BAAFFM"]
    macros = raw_fred[:, findall(x -> !(x ∈ excluded), names(raw_fred))]
    idx = ones(Int, 1)
    for i in axes(macros[:, 2:end], 2)
        if sum(ismissing.(macros[:, i+1])) == 0
            push!(idx, i + 1)
        end
    end
    macros = macros[:, idx]
    excluded = ["W875RX1", "IPFPNSS", "IPFINAL", "IPCONGD", "IPDCONGD", "IPNCONGD", "IPBUSEQ", "IPMAT", "IPDMAT", "IPNMAT", "IPMANSICS", "IPB51222S", "IPFUELS", "HWIURATIO", "CLF16OV", "CE16OV", "UEMPLT5", "UEMP5TO14", "UEMP15OV", "UEMP15T26", "UEMP27OV", "USGOOD", "CES1021000001", "USCONS", "MANEMP", "DMANEMP", "NDMANEMP", "SRVPRD", "USTPU", "USWTRADE", "USTRADE", "USFIRE", "USGOVT", "AWOTMAN", "AWHMAN", "CES2000000008", "CES3000000008", "HOUSTNE", "HOUSTMW", "HOUSTS", "HOUSTW", "PERMITNE", "PERMITMW", "PERMITS", "PERMITW", "NONBORRES", "DTCOLNVHFNM", "AAAFFM", "BAAFFM", "EXSZUSx", "EXJPUSx", "EXUSUKx", "EXCAUSx", "WPSFD49502", "WPSID61", "WPSID62", "CPIAPPSL", "CPITRNSL", "CPIMEDSL", "CUSR0000SAC", "CUSR0000SAS", "CPIULFSL", "CUSR0000SA0L2", "CUSR0000SA0L5", "DDURRG3M086SBEA", "DNDGRG3M086SBEA", "DSERRG3M086SBEA"]
    push!(excluded, "CMRMTSPLx", "RETAILx", "HWI", "UEMPMEAN", "CLAIMSx", "AMDMNOx", "ANDENOx", "AMDMUOx", "BUSINVx", "ISRATIOx", "BUSLOANS", "NONREVSL", "CONSPI", "S&P: indust", "S&P div yield", "S&P PE ratio", "M1SL", "BOGMBASE")
    macros = macros[:, findall(x -> !(x ∈ excluded), names(macros))]
    macros = [macros[:, 1] Float64.(macros[:, 2:end])]
    rename!(macros, Dict(:x1 => "date"))
    raw_macros = deepcopy(macros)

    ρ = Vector{Float64}(undef, size(macros[:, 2:end], 2))
    is_percent = fill(false, size(macros[:, 2:end], 2))
    idx_diff = Vector{Float64}(undef, size(macros[:, 2:end], 2))
    logmacros = similar(macros[:, 2:end] |> Array)
    for i in axes(macros[:, 2:end], 2) # i'th macro variable (excluding date)
        logmacros[:, i] = 100log.(macros[:, i+1])

        if names(macros[:, 2:end])[i] ∈ ["CUMFNS", "UNRATE", "AAA", "BAA"]
            is_percent[i] = true
        end

        if names(macros[:, 2:end])[i] ∈ ["AAA", "BAA"]
            macros[2:end, i+1] = macros[2:end, i+1] - macros[1:end-1, i+1]
            ρ[i] = 0.0
            idx_diff[i] = 1
        elseif names(macros[:, 2:end])[i] ∈ ["CUMFNS", "UNRATE"]
            ρ[i] = 1.0
            idx_diff[i] = 0
        elseif names(macros[:, 2:end])[i] ∈ ["CES0600000007", "VIXCLSx"]
            macros[:, i+1] = log.(macros[:, i+1]) |> x -> 100 * x
            ρ[i] = 1.0
            idx_diff[i] = 0
        else
            macros[2:end, i+1] = log.(macros[2:end, i+1]) - log.(macros[1:end-1, i+1]) |> x -> 1200 * x
            ρ[i] = 0.0
            idx_diff[i] = 1
        end
    end

    raw_macros = raw_macros[3:end, :]
    macros = macros[3:end, :]
    logmacros = logmacros[3:end, :]
    mean_macros = mean(macros[:, 2:end] |> Array, dims=1)[1, :]
    macros[:, 2:end] .-= mean_macros'

    ## Yield data
    raw_yield = XLSX.readdata("LW_monthly.xlsx", "Sheet1", "A293:DQ748") |> x -> [Date.(string.(x[:, 1]), DateFormat("yyyymm")) convert(Matrix{Float64}, x[:, τₙ.+1])] |> x -> DataFrame(x, ["date"; ["Y$i" for i in τₙ]])
    yields = raw_yield[findall(x -> x == yearmonth(date_start), yearmonth.(raw_yield[:, 1]))[1]:findall(x -> x == yearmonth(date_end), yearmonth.(raw_yield[:, 1]))[1], :]
    yields = yields[3:end, :]

    yields = [Date.(string.(yields[:, 1]), DateFormat("yyyy-mm-dd")) Float64.(yields[:, 2:end])]
    rename!(yields, Dict(:x1 => "date"))

    return ρ, is_percent, idx_diff, logmacros, raw_macros, macros, mean_macros, yields
end
ρ, is_percent, idx_diff, logmacros, raw_macros, macros, mean_macros, yields = data_loading(; date_start, date_end, τₙ)

## Tools
sdate(yy, mm) = findall(x -> x == Date(yy, mm), macros[:, 1])[1]
dQ = dimQ()
dP = size(macros, 2) - 1 + dQ

## Setting
# optimization
upper_q =
    [1 1
        1 1
        10 10
        100 100] .+ 0.0
μkQ_infty = 0
σkQ_infty = 0.2

# estimation
iteration = 6_000
burnin = 1_000
TPτ_interest = 120

##

sdate(yy, mm) = findall(x -> x == Date(yy, mm), macros[:, 1])[1]
function estimation(p_, tuned; upper_p, τₙ, medium_τ, iteration, burnin, ρ, macros, yields, μkQ_infty, σkQ_infty)

    tuned_ = Hyperparameter(p=p_, q=deepcopy(tuned.q), ν0=deepcopy(tuned.ν0), Ω0=deepcopy(tuned.Ω0), μϕ_const=deepcopy(tuned.μϕ_const))
    saved_θ = posterior_sampler(Array(yields[upper_p-p_+1:end, 2:end]), Array(macros[upper_p-p_+1:end, 2:end]), τₙ, ρ, iteration, tuned_; medium_τ, μkQ_infty, σkQ_infty)[1]
    saved_θ = saved_θ[burnin+1:end]
    return erase_nonstationary_param(saved_θ)[1]

end

## Do

is_load = true
if is_load
    saved_θ_vec = JLD2.load("standard/posterior_robustness_lag.jld2")["samples"]
else
    tuned = JLD2.load("standard/tuned.jld2")["tuned"]
    saved_θ_vec = Vector{Vector}(undef, 6)
    prog = Progress(iteration; dt=5, desc="robustness_lag.jl")
    Threads.@threads for p_ in 13:upper_p
        saved_θ_vec[p_-12] = estimation(p_, tuned; upper_p, τₙ, medium_τ, iteration, burnin, ρ, macros, yields, μkQ_infty, σkQ_infty)
        next!(prog)
    end
    finish!(prog)

    JLD2.save("standard/posterior_robustness_lag.jld2", "samples", saved_θ_vec)
end

## Comparisons
for i in ["κQ", "kQ_infty", "ϕ", "σ²FF", "Σₒ", "γ"]
    for j in [1, 2, 3, 4, 6]
        println(i, ", ", "p = $(j+12)")
        if i == "ϕ"
            println("mean: ", maximum(abs.(mean(saved_θ_vec[j][Symbol(i)])[:, 1:435] ./ mean(saved_θ_vec[5][Symbol(i)])[:, 1:435])))
            println("std: ", maximum(abs.(std(saved_θ_vec[j][Symbol(i)])[:, 1:435] ./ std(saved_θ_vec[5][Symbol(i)])[:, 1:435])))
        else
            println("mean: ", maximum(abs.(mean(saved_θ_vec[j][Symbol(i)]) ./ mean(saved_θ_vec[5][Symbol(i)]))))
            println("std: ", maximum(abs.(std(saved_θ_vec[j][Symbol(i)]) ./ std(saved_θ_vec[5][Symbol(i)]))))
        end
    end
end