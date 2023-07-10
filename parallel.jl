## Setting
using Distributed
# addprocs(3)
@everywhere begin
    using Pkg
    Pkg.activate(@__DIR__)
    # Pkg.instantiate()
    # Pkg.precompile()
end
@everywhere begin
    using GDTSM, ProgressMeter, StatsBase
    function mSR_ftn(mSR, mSR_data)
        return [skewness(mSR[1]), mSR[2]]
    end
end
using RCall, CSV, DataFrames, Dates, JLD2, LinearAlgebra, Gadfly, XLSX
import Plots

## Setting
τₙ = [1; 3; 6; 9; collect(12:6:60); collect(72:12:120)]
date_start = Date("1985-08-01", "yyyy-mm-dd")
date_end = Date("2022-12-01", "yyyy-mm-dd")
medium_τ = 12 * [2, 2.5, 3, 3.5, 4, 4.5, 5]
upper_lag = 15

step = 0
upper_q =
    [1 1
        1 1
        10 10
        100 100]
μkQ_infty = 0
σkQ_infty = 0.01
mSR_upper = [Inf; Inf]
opt_size = 100

iteration = 35_000
burnin = 5_000
TPτ_interest = 120
is_TP = true
is_ineff = true

begin ## Data: macro data
    raw_fred = CSV.File("current.csv") |> DataFrame |> x -> x[314:774, :]
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

    ρ = Vector{Float64}(undef, size(macros[:, 2:end], 2))
    for i in axes(macros[:, 2:end], 2) # i'th macro variable (excluding date)
        if names(macros[:, 2:end])[i] ∈ ["AAA", "BAA"]
            macros[2:end, i+1] = macros[2:end, i+1] - macros[1:end-1, i+1]
            ρ[i] = 0.0
        elseif names(macros[:, 2:end])[i] ∈ ["CUMFNS", "UNRATE", "CES0600000007", "VIXCLSx"]
            ρ[i] = 1.0
        elseif names(macros[:, 2:end])[i] ∈ ["HOUST", "PERMIT", "REALLN", "S&P 500", "CPIAUCSL", "PCEPI", "CES0600000008", "DTCTHFNM"]
            macros[2:end, i+1] = log.(macros[2:end, i+1]) - log.(macros[1:end-1, i+1]) |> x -> 1200 * x
            macros[2:end, i+1] = macros[2:end, i+1] - macros[1:end-1, i+1]
            ρ[i] = 0.0
        else
            macros[2:end, i+1] = log.(macros[2:end, i+1]) - log.(macros[1:end-1, i+1]) |> x -> 1200 * x
            ρ[i] = 0.0
        end
    end
    macros = macros[3:end, :]
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
# μϕ_const_PCs = -calibration_μϕ_const(μkQ_infty, σkQ_infty, 120, Array(yields[upper_lag-aux_lag+1:end, 2:end]), τₙ, aux_lag; medium_τ, iteration=10000)[2] |> x -> mean(x, dims=1)[1, :]
# μϕ_const_PCs = [0.09, μϕ_const_PCs[2], μϕ_const_PCs[3]]
# μϕ_const = [μϕ_const_PCs; zeros(size(macros, 2) - 1)]
# @show calibration_μϕ_const(μkQ_infty, σkQ_infty, 120, Array(yields[upper_lag-aux_lag+1:end, 2:end]), τₙ, aux_lag; medium_τ, μϕ_const_PCs, iteration=10000)[1] |> mean

if step == 1 ## Drawing pareto frontier

    tuned, opt = tuning_hyperparameter(Array(yields[:, 2:end]), Array(macros[:, 2:end]), τₙ, ρ; upper_lag, upper_q, μkQ_infty, σkQ_infty, medium_τ)
    save("tuned.jld2", "tuned", tuned, "opt", opt)

elseif step == 2 ## Tuning hyperparameter

    par_tuned = @showprogress 1 "Tuning..." pmap(1:upper_lag) do i
        μϕ_const_PCs = -calibration_μϕ_const(μkQ_infty, σkQ_infty, 120, Array(yields[upper_lag-i+1:end, 2:end]), τₙ, i; medium_τ, iteration=10000)[2] |> x -> mean(x, dims=1)[1, :]
        μϕ_const_PCs = [0.09, μϕ_const_PCs[2], μϕ_const_PCs[3]]
        μϕ_const = [μϕ_const_PCs; zeros(size(macros, 2) - 1)]
        @show calibration_μϕ_const(μkQ_infty, σkQ_infty, 120, Array(yields[upper_lag-i+1:end, 2:end]), τₙ, i; medium_τ, μϕ_const_PCs, iteration=10000)[1] |> mean

        tuning_hyperparameter_MOEA(Array(yields[upper_lag-i+1:end, 2:end]), Array(macros[upper_lag-i+1:end, 2:end]), τₙ, ρ; lag=i, μkQ_infty, σkQ_infty, upper_q, medium_τ, μϕ_const, mSR_ftn, populationsize=opt_size)
    end
    pf = [par_tuned[i][1] for i in eachindex(par_tuned)]
    pf_input = [par_tuned[i][2] for i in eachindex(par_tuned)]
    opt = [par_tuned[i][3] for i in eachindex(par_tuned)]
    save("tuned_pf.jld2", "pf", pf, "pf_input", pf_input, "opt", opt)

elseif step == 3 ## Estimation

    if isinf.(mSR_upper) |> minimum
        opt = load("standard/tuned.jld2")["opt"]
        tuned = load("standard/tuned.jld2")["tuned"]
        lag = tuned.p
    else
        pf = load("mSR/tuned_pf.jld2")["pf"]
        pf_input = load("mSR/tuned_pf.jld2")["pf_input"]

        pf_vec = Matrix{Float64}(undef, upper_lag * opt_size, 3)
        pf_input_vec = Vector{Hyperparameter}(undef, upper_lag * opt_size)
        for i in 1:upper_lag
            pf_vec[opt_size*(i-1)+1:opt_size*i, :] = pf[i]
            pf_input_vec[opt_size*(i-1)+1:opt_size*i] = pf_input[i]
        end
        idx = (pf_vec[:, 2] .< mSR_upper[1]) .* (pf_vec[:, 3] .< mSR_upper[2])
        tuned_set = pf_input_vec[idx]
        log_ml = pf_vec[idx, 1]
        tuned = tuned_set[sortperm(log_ml, rev=true)][1]
        lag = tuned.p
    end

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

    # if isinf.(mSR_upper) |> minimum
    save("posterior.jld2", "samples", saved_θ, "acceptPr", [acceptPr_C_σ²FF; acceptPr_ηψ], "accept_rate", accept_rate)
    # else
    #     par_mSR_θ = @showprogress 1 "filtering with mSR_ftn..." pmap(1:iteration) do i
    #         mSR_ftn_filter([saved_θ[i]], Array(yields[p_max-lag+1:end, 2:end]), Array(macros[p_max-lag+1:end, 2:end]), τₙ; mSR_ftn, mSR_upper)
    #     end
    #     saved_θ = Vector{Parameter}(undef, 0)
    #     for i in eachindex(par_mSR_θ)
    #         if !isempty(par_mSR_θ[i][1])
    #             push!(saved_θ, par_mSR_θ[i][1][1])
    #         end
    #     end
    #     accept_mSR = 100length(saved_θ) / iteration
    #     iteration = length(saved_θ)
    #     save("posterior.jld2", "samples", saved_θ, "acceptPr", [acceptPr_C_σ²FF; acceptPr_ηψ], "accept_rate", accept_rate, "accept_rate_mSR", accept_mSR)
    # end

    if is_ineff
        ineff = ineff_factor(saved_θ)
        save("ineff.jld2", "ineff", ineff)
    end

    if is_TP
        par_TP = @showprogress 1 "Term premium..." pmap(1:ceil(Int, maximum(ineff)):iteration) do i
            term_premium(TPτ_interest, τₙ, [saved_θ[i]], Array(yields[upper_lag-lag+1:end, 2:end]), Array(macros[upper_lag-lag+1:end, 2:end]))
        end
        saved_TP = [par_TP[i][1] for i in eachindex(par_TP)]
        save("TP.jld2", "TP", saved_TP)
    end

else

    # from step 1&2
    if isinf.(mSR_upper) |> minimum
        opt = load("standard/tuned.jld2")["opt"]
        tuned = load("standard/tuned.jld2")["tuned"]
        lag = tuned.p
    else
        pf = load("mSR/tuned_pf.jld2")["pf"]
        pf_input = load("mSR/tuned_pf.jld2")["pf_input"]

        pf_vec = Matrix{Float64}(undef, upper_lag * opt_size, 3)
        pf_input_vec = Vector{Hyperparameter}(undef, upper_lag * opt_size)
        for i in 1:upper_lag
            pf_vec[opt_size*(i-1)+1:opt_size*i, :] = pf[i]
            pf_input_vec[opt_size*(i-1)+1:opt_size*i] = pf_input[i]
        end
        idx = (pf_vec[:, 2] .< mSR_upper[1]) .* (pf_vec[:, 3] .< mSR_upper[2])
        tuned_set = pf_input_vec[idx]
        log_ml = pf_vec[idx, 1]
        tuned = tuned_set[sortperm(log_ml, rev=true)][1]
        lag = tuned.p
    end

    # from step 3
    if isinf.(mSR_upper) |> minimum
        saved_θ = load("standard/posterior.jld2")["samples"]
        acceptPr = load("standard/posterior.jld2")["acceptPr"]
        accept_rate = load("standard/posterior.jld2")["accept_rate"]
        iteration = length(saved_θ)
        saved_TP = load("standard/TP.jld2")["TP"]
        ineff = load("standard/ineff.jld2")["ineff"]
    else
        saved_θ = load("mSR/posterior.jld2")["samples"]
        acceptPr = load("mSR/posterior.jld2")["acceptPr"]
        accept_rate = load("mSR/posterior.jld2")["accept_rate"]
        # accept_mSR = load("mSR/posterior.jld2")["accept_rate_mSR"]
        iteration = length(saved_θ)
        saved_TP = load("mSR/TP.jld2")["TP"]
        ineff = load("mSR/ineff.jld2")["ineff"]
    end

    saved_Xθ = latentspace(saved_θ, Array(yields[upper_lag-lag+1:end, 2:end]), τₙ)
    fitted = fitted_YieldCurve(collect(1:τₙ[end]), saved_Xθ)
    fitted_yield = mean(fitted)[:yields] / 1200
    log_price = -collect(1:τₙ[end])' .* fitted_yield[tuned.p:end, :]
    xr = log_price[2:end, 1:end-1] - log_price[1:end-1, 2:end] .- fitted_yield[tuned.p:end-1, 1]
    realized_SR = mean(xr, dims=1) ./ std(xr, dims=1) |> x -> x[1, :]
    reduced_θ = reducedform(saved_θ[1:ceil(Int, maximum(ineff)):iteration], Array(yields[upper_lag-lag+1:end, 2:end]), Array(macros[upper_lag-lag+1:end, 2:end]), τₙ)
    mSR = [reduced_θ[:mpr][i] |> x -> sqrt.(diag(x * x')) for i in eachindex(reduced_θ)] |> mean
    include("ex_scenario.jl")

end