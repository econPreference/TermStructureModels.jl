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

## Setting
τₙ = [3; 6; collect(12:12:120)]
date_start = Date("1985-12-01", "yyyy-mm-dd")
date_end = Date("2020-02-01", "yyyy-mm-dd")

p_max = 12
step = 0
lag = 1
iteration = 25_000
burnin = 5000
issparse_coef = false
issparse_prec = false
TPτ_interest = 120

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
        if rcopy(rcall(:describe_md, names(macros[:, 2:end])))[:, :fred][i] ∈ ["CUMFNS", "UNRATE", "AAA", "BAA"]
            macros[2:end, i+1] = macros[2:end, i+1] - macros[1:end-1, i+1]
            ρ[i] = 0.0
        else
            macros[2:end, i+1] = 100(log.(macros[2:end, i+1]) - log.(macros[1:end-1, i+1]))
            ρ[i] = 0.0
        end
    end
    macros = macros[2:end, :]
    # mean_macro = mean(Array(macros[:, 2:end]), dims=1)
    # macros[:, 2:end] .-= mean_macro
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

if step == 0 ## Drawing pareto frontier

    par_tuned = @showprogress 1 "Tuning..." pmap(1:p_max) do i
        tuning_hyperparameter_MOEA(Array(yields[p_max-i+1:end, 2:end]), Array(macros[p_max-i+1:end, 2:end]), τₙ, ρ; lag=i)
    end
    pf = [par_tuned[i][1] for i in eachindex(par_tuned)]
    pf_input = [par_tuned[i][2] for i in eachindex(par_tuned)]
    opt = [par_tuned[i][3] for i in eachindex(par_tuned)]
    save("tuned_pf.jld2", "pf", pf, "pf_input", pf_input, "opt", opt)

elseif step == 1 ## Tuning hyperparameter

    par_tuned = @showprogress 1 "Tuning..." pmap(1:p_max) do i
        tuning_hyperparameter(Array(yields[p_max-i+1:end, 2:end]), Array(macros[p_max-i+1:end, 2:end]), τₙ, ρ; lag=i)
    end
    tuned = [par_tuned[i][1] for i in eachindex(par_tuned)]
    opt = [par_tuned[i][2] for i in eachindex(par_tuned)]
    save("tuned.jld2", "tuned", tuned, "opt", opt)

elseif step == 2 ## Estimation

    tuned = load("tuned.jld2")["tuned"][lag]
    saved_θ, acceptPr_C_σ²FF, acceptPr_ηψ = posterior_sampler(Array(yields[p_max-lag+1:end, 2:end]), Array(macros[p_max-lag+1:end, 2:end]), τₙ, ρ, iteration, tuned; sparsity=issparse_coef)
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
    save("posterior.jld2", "samples", saved_θ, "acceptPr", [acceptPr_C_σ²FF; acceptPr_ηψ], "accept_rate", accept_rate)

    if issparse_prec == true
        par_sparse_θ = @showprogress 1 "Sparse precision..." pmap(1:iteration) do i
            sparse_precision([saved_θ[i]], size(macros, 1) - tuned.p)
        end
        saved_θ = [par_sparse_θ[i][1][1] for i in eachindex(par_sparse_θ)]
        trace_sparsity = [par_sparse_θ[i][2][1] for i in eachindex(par_sparse_θ)]
        save("sparse.jld2", "samples", saved_θ, "sparsity", trace_sparsity)
    end

    par_TP = @showprogress 1 "Term premium..." pmap(1:iteration) do i
        term_premium(TPτ_interest, τₙ, [saved_θ[i]], Array(yields[p_max-lag+1:end, 2:end]), Array(macros[p_max-lag+1:end, 2:end]))
    end
    saved_TP = [par_TP[i][1] for i in eachindex(par_TP)]
    save("TP.jld2", "TP", saved_TP)

elseif step == 3 ## Statistical inference

    # from step 0
    pf = load("tuned_pf.jld2")["pf"]
    pf_input = load("tuned_pf.jld2")["pf_input"]
    opt = load("tuned_pf.jld2")["opt"]
    pf_plot = Plots.scatter(pf[:, 2], pf[:, 1], ylabel="marginal likelhood", xlabel="E[maximum SR]", label="")

    # from step 1
    tuned_set = load("tuned.jld2")["tuned"]
    tuned = tuned_set[lag]
    opt = load("tuned.jld2")["opt"]
    mSR_prior = maximum_SR(Array(yields[p_max-lag+1:end, 2:end]), Array(macros[p_max-lag+1:end, 2:end]), tuned, τₙ, ρ; iteration=1000)

    # from step 2
    if issparse_prec == true
        saved_θ = load("sparse.jld2")["samples"]
        trace_sparsity = load("sparse.jld2")["sparsity"]
        iteration = length(saved_θ)
    else
        saved_θ = load("posterior.jld2")["samples"]
        acceptPr = load("posterior.jld2")["acceptPr"]
        accept_rate = load("posterior.jld2")["accept_rate"]
        iteration = length(saved_θ)
    end
    saved_TP = load("TP.jld2")["TP"]

    saved_Xθ = latentspace(saved_θ, Array(yields[p_max-lag+1:end, 2:end]), τₙ)
    fitted = fitted_YieldCurve(collect(1:τₙ[end]), saved_Xθ)
    fitted_yield = mean(fitted)[:yields] / 1200
    log_price = -collect(1:τₙ[end])' .* fitted_yield[tuned.p+1:end, :]
    xr = log_price[2:end, 1:end-1] - log_price[1:end-1, 2:end] .- fitted_yield[tuned.p+1:end-1, 1]
    realized_SR = mean(xr, dims=1) ./ std(xr, dims=1) |> x -> x[1, :]
    reduced_θ = reducedform(saved_θ, Array(yields[p_max-lag+1:end, 2:end]), Array(macros[p_max-lag+1:end, 2:end]), τₙ)
    mSR = mean(reduced_θ)[:mpr] |> x -> diag(x * x')

    ## Scenario Analysis
    begin ## Data: macro data
        R"library(fbi)"
        raw_fred = rcopy(rcall(:fredmd, file="current.csv", date_start=Date("1985-12-01", "yyyy-mm-dd"), date_end=Date("2020-12-01", "yyyy-mm-dd"), transform=false))
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
            if rcopy(rcall(:describe_md, names(macros_extended[:, 2:end])))[:, :fred][i] ∈ ["CUMFNS", "UNRATE", "AAA", "BAA"]
                macros_extended[2:end, i+1] = macros_extended[2:end, i+1] - macros_extended[1:end-1, i+1]
                ρ[i] = 0.0
            else
                macros_extended[2:end, i+1] = 100(log.(macros_extended[2:end, i+1]) - log.(macros_extended[1:end-1, i+1]))
                ρ[i] = 0.0
            end
        end
        macros_extended = macros_extended[2:end, :]
        # macros_extended[:, 2:end] .-= mean_macro
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
        local combs = zeros(3, dP - dimQ() + length(τₙ))
        local combs[1:3, 1:3] = I(3)
        local vals = zeros(3)
        push!(scene, Scenario(combinations=combs, values=vals))
    end
    prediction = scenario_sampler(scene, 24, 10, saved_θ, Array(yields[p_max-lag+1:end, 2:end]), Array(macros[p_max-lag+1:end, 2:end]), τₙ)
end