## Setting
using Distributed
# addprocs(8)
@everywhere begin
    using Pkg
    Pkg.activate(@__DIR__)
    # Pkg.instantiate()
    # Pkg.precompile()
end
@everywhere begin
    using GDTSM, ProgressMeter
end
using RCall, CSV, DataFrames, Dates, JLD2, LinearAlgebra
import Plots
date_start = Date("1986-12-01", "yyyy-mm-dd")
date_end = Date("2020-02-01", "yyyy-mm-dd")

begin ## Data: macro data
    R"library(fbi)"
    raw_fred = rcopy(rcall(:fredmd, file="current.csv", date_start=date_start, date_end=date_end, transform=false))
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
    ρ = Vector{Float64}(undef, size(macros[:, 2:end], 2))
    for i in axes(macros[:, 2:end], 2) # i'th macro variable (excluding date)
        if rcopy(rcall(:describe_md, names(macros[:, 2:end])))[:, :fred][i] ∈ ["CUMFNS", "UNRATE"]
            macros[:, i+1] = log.(macros[:, i+1])
            ρ[i] = 0.9
        else
            macros[2:end, i+1] = 1200(log.(macros[2:end, i+1]) - log.(macros[1:end-1, i+1]))
            ρ[i] = 0
        end
    end
    macros = macros[2:end, :]
    mean_macro = mean(Array(macros[:, 2:end]), dims=1)
    macros[:, 2:end] .-= mean_macro
    # macros[:, 2:end] ./= std(Array(macros[:, 2:end]), dims=1)
end

begin ## Data: yield data
    # yield(3 months) and yield(6 months)
    raw_yield = CSV.File("FRB_H15.csv", missingstring="ND", types=[Date; fill(Float64, 11)]) |> DataFrame |> (x -> [x[5137:end, 1] x[5137:end, 3:4]]) |> dropmissing
    idx = month.(raw_yield[:, 1]) |> x -> (x .!= [x[2:end]; x[end]])
    yield_month = raw_yield[idx, :]
    yield_month = yield_month[findall(x -> x == yearmonth(date_start), yearmonth.(yield_month[:, 1]))[1]:findall(x -> x == yearmonth(date_end), yearmonth.(yield_month[:, 1]))[1], :] |> x -> x[:, 2:end]
    # longer than one year
    raw_yield = CSV.File("feds200628.csv", missingstring="NA", types=[Date; fill(Float64, 99)]) |> DataFrame |> (x -> [x[8:end, 1] x[8:end, 69:78]]) |> dropmissing
    idx = month.(raw_yield[:, 1]) |> x -> (x .!= [x[2:end]; x[end]])
    yield_year = raw_yield[idx, :]
    yield_year = yield_year[findall(x -> x == yearmonth(date_start), yearmonth.(yield_year[:, 1]))[1]:findall(x -> x == yearmonth(date_end), yearmonth.(yield_year[:, 1]))[1], :]
    yields = DataFrame([Matrix(yield_month) Matrix(yield_year[:, 2:end])], [:M3, :M6, :Y1, :Y2, :Y3, :Y4, :Y5, :Y6, :Y7, :Y8, :Y9, :Y10])
    yields = [yield_year[:, 1] yields]
    rename!(yields, Dict(:x1 => "date"))
    yields = yields[2:end, :]
end

## Tuning hyper-parameters
τₙ = [3; 6; collect(12:12:120)]
# tuned = tuning_hyperparameter(Array(yields[:, 2:end]), Array(macros[:, 2:end]), τₙ, ρ; upper_lag=8, upper_q1=0.1, upper_q45=50, σ²kQ_infty=0.01^2, weight=1E+5, mSR_mean=1.0)
# save("tuned.jld2", "tuned", tuned)
tuned, EA_opt = tuning_hyperparameter_mSR(Array(yields[:, 2:end]), Array(macros[:, 2:end]), τₙ, ρ; maxstep=50_000, weight=1E+5, mSR_mean=0.75, upper_lag=4, upper_q1=0.01, upper_q45=40, σ²kQ_infty=0.01^2)
save("tuned_pf.jld2", "tuned", tuned, "EA_opt", EA_opt)
tuned = load("tuned_pf.jld2")["tuned"]
EA_opt = load("tuned_pf.jld2")["EA_opt"]
tuned, p = mSR_ML_frontier(EA_opt, size(macros, 2) - 1; mSR_mean=0.5, σ²kQ_infty=0.01^2)
# Plots.plot(p)
mSR_prior = maximum_SR(Array(yields[:, 2:end]), Array(macros[:, 2:end]), tuned, τₙ, ρ; iteration=1000)

## Estimation
iteration = 25_000
saved_θ, acceptPr_C_σ²FF, acceptPr_ηψ = posterior_sampler(Array(yields[:, 2:end]), Array(macros[:, 2:end]), τₙ, ρ, iteration, tuned; sparsity=false)
save("posterior.jld2", "samples", saved_θ, "acceptPr", [acceptPr_C_σ²FF; acceptPr_ηψ])
saved_θ = load("posterior.jld2")["samples"]
saved_θ = saved_θ[5001:end]
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

# par_sparse_θ = @showprogress 1 "Sparse precision..." pmap(1:iteration) do i
#     sparse_precision([saved_θ[i]], size(macros, 1) - tuned.p)
# end
# saved_θ = [par_sparse_θ[i][1][1] for i in eachindex(par_sparse_θ)]
# trace_sparsity = [par_sparse_θ[i][2][1] for i in eachindex(par_sparse_θ)]
# save("sparse.jld2", "samples", saved_θ, "sparsity", trace_sparsity)
# saved_θ = load("sparse.jld2")["samples"]
# trace_sparsity = load("sparse.jld2")["sparsity"]
reduced_θ = reducedform(saved_θ, Array(yields[:, 2:end]), Array(macros[:, 2:end]), τₙ)

τ_interest = 120
par_TP = @showprogress 1 "Term premium..." pmap(1:iteration) do i
    term_premium(τ_interest, τₙ, [saved_θ[i]], Array(yields[:, 2:end]), Array(macros[:, 2:end]))
end
saved_TP = [par_TP[i][1] for i in eachindex(par_TP)]
save("TP.jld2", "TP", saved_TP)
saved_TP = load("TP.jld2")["TP"]
saved_Xθ = latentspace(saved_θ, Array(yields[:, 2:end]), τₙ)
fitted = fitted_YieldCurve([1; τₙ], saved_Xθ)

fitted = fitted_YieldCurve(collect(1:120), saved_Xθ)
fitted_yield = mean(fitted)[:yields] / 1200
log_price = -collect(1:120)' .* fitted_yield[tuned.p+1:end, :]
xr = log_price[2:end, 1:end-1] - log_price[1:end-1, 2:end] .- fitted_yield[tuned.p+1:end-1, 1]
realized_SR = mean(xr, dims=1) ./ std(xr, dims=1) |> x -> x[1, :]
mSR = mean(reduced_θ)[:mpr] |> x -> diag(x * x')

## Scenario
begin ## Data: macro data
    R"library(fbi)"
    raw_fred = rcopy(rcall(:fredmd, file="current.csv", date_start=Date("1986-12-01", "yyyy-mm-dd"), date_end=Date("2020-12-01", "yyyy-mm-dd"), transform=false))
    excluded = ["FEDFUNDS", "CP3Mx", "TB3MS", "TB6MS", "GS1", "GS5", "GS10", "TB3SMFFM", "TB6SMFFM", "T1YFFM", "T5YFFM", "T10YFFM", "COMPAPFFx", "AAAFFM", "BAAFFM"]
    macros_extended = raw_fred[:, findall(x -> !(x ∈ excluded), names(raw_fred))]
    idx = ones(Int, 1)
    for i in axes(macros_extended[:, 2:end], 2)
        if sum(ismissing.(macros_extended[:, i+1])) == 0
            push!(idx, i + 1)
        end
    end
    macros_extended = macros_extended[:, idx]
    excluded = ["W875RX1", "IPFPNSS", "IPFINAL", "IPCONGD", "IPDCONGD", "IPNCONGD", "IPBUSEQ", "IPMAT", "IPDMAT", "IPNMAT", "IPMANSICS", "IPB51222S", "IPFUELS", "HWIURATIO", "CLF16OV", "CE16OV", "UEMPLT5", "UEMP5TO14", "UEMP15OV", "UEMP15T26", "UEMP27OV", "USGOOD", "CES1021000001", "USCONS", "MANEMP", "DMANEMP", "NDMANEMP", "SRVPRD", "USTPU", "USWTRADE", "USTRADE", "USFIRE", "USGOVT", "AWOTMAN", "AWHMAN", "CES2000000008", "CES3000000008", "HOUSTNE", "HOUSTMW", "HOUSTS", "HOUSTW", "PERMITNE", "PERMITMW", "PERMITS", "PERMITW", "NONBORRES", "DTCOLNVHFNM", "AAAFFM", "BAAFFM", "EXSZUSx", "EXJPUSx", "EXUSUKx", "EXCAUSx", "WPSFD49502", "WPSID61", "WPSID62", "CPIAPPSL", "CPITRNSL", "CPIMEDSL", "CUSR0000SAC", "CUSR0000SAS", "CPIULFSL", "CUSR0000SA0L2", "CUSR0000SA0L5", "DDURRG3M086SBEA", "DNDGRG3M086SBEA", "DSERRG3M086SBEA"]
    push!(excluded, "CMRMTSPLx", "RETAILx", "HWI", "UEMPMEAN", "CLAIMSx", "AMDMNOx", "ANDENOx", "AMDMUOx", "BUSINVx", "ISRATIOx", "BUSLOANS", "NONREVSL", "CONSPI", "S&P: indust", "S&P div yield", "S&P PE ratio", "M1SL", "BOGMBASE")
    macros_extended = macros_extended[:, findall(x -> !(x ∈ excluded), names(macros_extended))]
    ρ = Vector{Float64}(undef, size(macros_extended[:, 2:end], 2))
    for i in axes(macros_extended[:, 2:end], 2) # i'th macro variable (excluding date)
        if rcopy(rcall(:describe_md, names(macros_extended[:, 2:end])))[:, :fred][i] ∈ ["CUMFNS", "UNRATE"]
            macros_extended[:, i+1] = log.(macros_extended[:, i+1])
            ρ[i] = 0.9
        else
            macros_extended[2:end, i+1] = 1200(log.(macros_extended[2:end, i+1]) - log.(macros_extended[1:end-1, i+1]))
            ρ[i] = 0
        end
    end
    macros_extended = macros_extended[2:end, :]
    macros_extended[:, 2:end] .-= mean_macro
    # macros[:, 2:end] ./= std(Array(macros[:, 2:end]), dims=1)
end

dP = size(macros_extended, 2) - 1 + dimQ()
scene = Vector{Scenario}(undef, 0)
combs = zeros(dP - dimQ() + 3, dP - dimQ() + length(τₙ))
vals = Vector{Float64}(undef, size(combs, 1))
combs[1:3, 1:3] = I(3)
vals[1:3] = zeros(3)
combs[4:end, length(τₙ)+1:length(τₙ)+dP-dimQ()] = I(dP - dimQ())
vals[4:end] = macros_extended[end-9, 2:end] |> Array
push!(scene, Scenario(combinations=combs, values=vals))
for h = 2:10
    combs = zeros(3, dP - dimQ() + length(τₙ))
    combs[1:3, 1:3] = I(3)
    vals = zeros(3)
    push!(scene, Scenario(combinations=combs, values=vals))
end
prediction = scenario_sampler(scene, 24, 10, saved_θ, Array(yields[:, 2:end]), Array(macros[:, 2:end]), τₙ)
