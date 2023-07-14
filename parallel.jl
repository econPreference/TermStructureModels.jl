## Setting
using Distributed
# addprocs(2)
@everywhere begin
    using Pkg
    Pkg.activate(@__DIR__)
    # Pkg.instantiate()
    # Pkg.precompile()
end
@everywhere begin
    using GDTSM, ProgressMeter, StatsBase, Dates
end
using RCall, CSV, DataFrames, LinearAlgebra, Gadfly, XLSX
import Plots, JLD2

## Setting
upper_lag = 18
date_start = Date("1987-01-01", "yyyy-mm-dd") |> x -> x - Month(upper_lag + 2)
date_end = Date("2022-12-01", "yyyy-mm-dd")
τₙ = [1; 3; 6; 9; collect(12:6:60); collect(72:12:120)]
medium_τ = 12 * [2, 2.5, 3, 3.5, 4, 4.5, 5]

step = 0
μϕ_const_PC1 = 0.1065
upper_q =
    [1 1
        1 1
        10 10
        100 100] .+ 0.0
μkQ_infty = 0
σkQ_infty = 0.01

iteration = 35_000
burnin = 5_000
TPτ_interest = 120
is_TP = true
is_ineff = true

begin ## Data: macro data
    raw_fred = CSV.File("current.csv") |> DataFrame |> x -> x[302:774, :]
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
    macros_growth = similar(macros[:, 2:end] |> Array)
    for i in axes(macros[:, 2:end], 2) # i'th macro variable (excluding date)
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
        elseif names(macros[:, 2:end])[i] ∈ ["HOUST", "PERMIT", "REALLN", "S&P 500", "CPIAUCSL", "PCEPI", "CES0600000008", "DTCTHFNM"]
            macros_growth[2:end, i] = log.(macros[2:end, i+1]) - log.(macros[1:end-1, i+1]) |> x -> 1200 * x
            macros[2:end, i+1] = macros_growth[2:end, i]
            macros[2:end, i+1] = macros[2:end, i+1] - macros[1:end-1, i+1]
            ρ[i] = 0.0
            idx_diff[i] = 2
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
    macros_growth = macros_growth[3:end, :]
    mean_macros = mean(macros[:, 2:end] |> Array, dims=1)
    macros[:, 2:end] .-= mean_macros
end

begin ## Data: yield data
    raw_yield = XLSX.readdata("LW_monthly.xlsx", "Sheet1", "A132:DQ748") |> x -> [Date.(string.(x[:, 1]), DateFormat("yyyymm")) convert(Matrix{Float64}, x[:, τₙ.+1])] |> x -> DataFrame(x, ["date"; ["Y$i" for i in τₙ]])
    yields = raw_yield[findall(x -> x == yearmonth(date_start), yearmonth.(raw_yield[:, 1]))[1]:findall(x -> x == yearmonth(date_end), yearmonth.(raw_yield[:, 1]))[1], :]
    yields = yields[3:end, :]

    yields = [Date.(string.(yields[:, 1]), DateFormat("yyyy-mm-dd")) Float64.(yields[:, 2:end])]
    rename!(yields, Dict(:x1 => "date"))
end

# aux_lag = 7
# μϕ_const_PCs = -calibration_μϕ_const(μkQ_infty, σkQ_infty, 120, Array(yields[upper_lag-aux_lag+1:end, 2:end]), τₙ, aux_lag; medium_τ, iteration=10_000)[2] |> x -> mean(x, dims=1)[1, :]
# μϕ_const_PCs = [0.1065, μϕ_const_PCs[2], μϕ_const_PCs[3]]
# μϕ_const = [μϕ_const_PCs; zeros(size(macros, 2) - 1)]
# @show calibration_μϕ_const(μkQ_infty, σkQ_infty, 120, Array(yields[upper_lag-aux_lag+1:end, 2:end]), τₙ, aux_lag; medium_τ, μϕ_const_PCs, iteration=10_000)[1] |> mean

if step == 1 ## Drawing pareto frontier

    tuned, opt = tuning_hyperparameter(Array(yields[:, 2:end]), Array(macros[:, 2:end]), τₙ, ρ; upper_lag, upper_q, μkQ_infty, σkQ_infty, medium_τ, μϕ_const_PC1)
    JLD2.save("tuned.jld2", "tuned", tuned, "opt", opt)

elseif step == 2 ## Estimation

    opt = JLD2.load("results/tuned.jld2")["opt"]
    tuned = JLD2.load("results/tuned.jld2")["tuned"]
    lag = tuned.p

    saved_θ, acceptPr_C_σ²FF, acceptPr_ηψ = posterior_sampler(Array(yields[upper_lag-lag+1:end, 2:end]), Array(macros[upper_lag-lag+1:end, 2:end]), τₙ, ρ, iteration, tuned; medium_τ)
    saved_θ = saved_θ[burnin+1:end]
    iteration = length(saved_θ)

    par_stationary_θ = @showprogress 1 "Stationary filtering..." pmap(1:iteration) do i
        stationary_θ([saved_θ[i]])
    end
    saved_θ = Vector{Parameter}(undef, 0)
    for i in eachindex(par_stationary_θ)
        if !isempty(par_stationary_θ[i][1])
            push!(saved_θ, par_stationary_θ[i][1][1])
        end
    end
    accept_rate = [par_stationary_θ[i][2] / 100 for i in eachindex(par_stationary_θ)] |> sum |> x -> (100x / iteration)
    iteration = length(saved_θ)
    JLD2.save("posterior.jld2", "samples", saved_θ, "acceptPr", [acceptPr_C_σ²FF; acceptPr_ηψ], "accept_rate", accept_rate)

    if is_ineff
        ineff = ineff_factor(saved_θ)
        JLD2.save("ineff.jld2", "ineff", ineff)
    end

    if is_TP
        par_TP = @showprogress 1 "Term premium..." pmap(1:ceil(Int, maximum(ineff)):iteration) do i
            term_premium(TPτ_interest, τₙ, [saved_θ[i]], Array(yields[upper_lag-lag+1:end, 2:end]), Array(macros[upper_lag-lag+1:end, 2:end]))
        end
        saved_TP = [par_TP[i][1] for i in eachindex(par_TP)]
        JLD2.save("TP.jld2", "TP", saved_TP)
    end

    include("ex_scenario.jl")

else
    @everywhere begin
        sdate(yy, mm) = findall(x -> x == Date(yy, mm), macros[:, 1])[1]
    end


    # from step 1
    opt = JLD2.load("results/tuned.jld2")["opt"]
    tuned = JLD2.load("results/tuned.jld2")["tuned"]
    lag = tuned.p
    @show calibration_μϕ_const(tuned.μkQ_infty, tuned.σkQ_infty, 120, Array(yields[upper_lag-lag+1:end, 2:end]), τₙ, lag; medium_τ, μϕ_const_PCs=tuned.μϕ_const[1:dimQ()], iteration=10_000)[1] |> mean
    @show prior_const_TP(tuned, 120, Array(yields[upper_lag-lag+1:end, 2:end]), τₙ, ρ; iteration=1_000) |> std

    # from step 2
    saved_θ = JLD2.load("results/posterior.jld2")["samples"]
    acceptPr = JLD2.load("results/posterior.jld2")["acceptPr"]
    accept_rate = JLD2.load("results/posterior.jld2")["accept_rate"]
    iteration = length(saved_θ)
    saved_TP = JLD2.load("results/TP.jld2")["TP"]
    ineff = JLD2.load("results/ineff.jld2")["ineff"]

    saved_Xθ = latentspace(saved_θ, Array(yields[upper_lag-lag+1:end, 2:end]), τₙ)
    fitted = fitted_YieldCurve(collect(1:τₙ[end]), saved_Xθ)
    fitted_yield = mean(fitted)[:yields] / 1200
    log_price = -collect(1:τₙ[end])' .* fitted_yield[tuned.p:end, :]
    xr = log_price[2:end, 1:end-1] - log_price[1:end-1, 2:end] .- fitted_yield[tuned.p:end-1, 1]
    realized_SR = mean(xr, dims=1) ./ std(xr, dims=1) |> x -> x[1, :]
    reduced_θ = reducedform(saved_θ[1:ceil(Int, maximum(ineff)):iteration], Array(yields[upper_lag-lag+1:end, 2:end]), Array(macros[upper_lag-lag+1:end, 2:end]), τₙ)
    mSR = [reduced_θ[:mpr][i] |> x -> sqrt.(diag(x * x')) for i in eachindex(reduced_θ)] |> mean

    raw_prediction = JLD2.load("results/scenario.jld2")["forecasts"]
    saved_prediction = Vector{Forecast}(undef, length(raw_prediction))
    for i in eachindex(saved_prediction)
        predicted_factors = deepcopy(raw_prediction[i][:factors])
        for j in 1:dP-dQ
            if idx_diff[j] == 2
                predicted_factors[1, dQ+j] = macros_growth[sdate(2020, 3), j]
                predicted_factors[:, dQ+j] = predicted_factors[:, dQ+j] |> cumsum
            elseif (idx_diff[j] == 1) && is_percent[j]
                predicted_factors[:, dQ+j] = [raw_macros[sdate(2020, 2), 1+j]; predicted_factors[:, dQ+j]] |> cumsum |> x -> x[2:end]
            elseif (idx_diff[j] == 0) && !is_percent[j]
                predicted_factors[:, dQ+j] = 12 * [macros[sdate(2020, 2), 1+j]; predicted_factors[:, dQ+j]] |> diff
            end
        end
        saved_prediction[i] = Forecast(yields=raw_prediction[i][:yields], factors=predicted_factors, TP=raw_prediction[i][:TP])
    end
end