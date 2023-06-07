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
using RCall, CSV, DataFrames, Dates, JLD2, LinearAlgebra, Gadfly, XLSX
import Plots

## Setting
τₙ = [3; 6; collect(12:12:120)]
date_start = Date("1985-11-01", "yyyy-mm-dd")
date_end = Date("2020-02-01", "yyyy-mm-dd")
medium_τ = 12 * [2, 2.5, 3, 3.5, 4, 4.5, 5]

p_max = 12
step = 4

upper_q =
    [1 1
        1 1
        10 10
        1 100]
upper_ν0 = 600
μkQ_infty = 0
σkQ_infty = 0.02
mSR_tail = 3

lag = 6
iteration = 25_000
burnin = 5_000
issparse_coef = false
issparse_prec = false
TPτ_interest = 120
is_TP = true

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
        if names(macros[:, 2:end])[i] ∈ ["CUMFNS", "AAA", "UNRATE", "BAA"]
            macros[2:end, i+1] = macros[2:end, i+1] - macros[1:end-1, i+1]
            ρ[i] = 0.0
        elseif names(macros[:, 2:end])[i] ∈ ["DPCERA3M086SBEA", "HOUST", "M2SL", "M2REAL", "REALLN", "WPSFD49207", "PCEPI", "DTCTHFNM", "INVEST"]
            macros[2:end, i+1] = log.(macros[2:end, i+1]) - log.(macros[1:end-1, i+1]) |> x -> 1200 * x
            macros[2:end, i+1] = macros[2:end, i+1] - macros[1:end-1, i+1]
            ρ[i] = 0.0
        else
            macros[2:end, i+1] = log.(macros[2:end, i+1]) - log.(macros[1:end-1, i+1]) |> x -> 1200 * x
            ρ[i] = 0.0
        end
    end
    macros = macros[3:end, :]
end

begin ## Data: yield data
    # yield(3 months) and yield(6 months)
    raw_yield = CSV.File("FRB_H15.csv", missingstring="ND", types=[Date; fill(Float64, 11)]) |> DataFrame |> (x -> [x[5137:end, 1] x[5137:end, 3:4]]) |> dropmissing
    idx = month.(raw_yield[:, 1]) |> x -> (x .!= [x[2:end]; x[end]]) |> findall
    for i in eachindex(idx)
        if i == 1
            raw_yield[idx[i], 2:end] = mean(raw_yield[1:idx[i], 2:end] |> Array, dims=1)
        else
            raw_yield[idx[i], 2:end] = mean(raw_yield[idx[i-1]+1:idx[i], 2:end] |> Array, dims=1)
        end
    end
    yield_month = raw_yield[idx, :]
    yield_month = yield_month[findall(x -> x == yearmonth(date_start), yearmonth.(yield_month[:, 1]))[1]:findall(x -> x == yearmonth(date_end), yearmonth.(yield_month[:, 1]))[1], :] |> x -> x[:, 2:end]
    # longer than one year
    raw_yield = CSV.File("feds200628.csv", missingstring="NA", types=[Date; fill(Float64, 99)]) |> DataFrame |> (x -> [x[8:end, 1] x[8:end, 69:78]]) |> dropmissing
    idx = month.(raw_yield[:, 1]) |> x -> (x .!= [x[2:end]; x[end]]) |> findall
    for i in eachindex(idx)
        if i == 1
            raw_yield[idx[i], 2:end] = mean(raw_yield[1:idx[i], 2:end] |> Array, dims=1)
        else
            raw_yield[idx[i], 2:end] = mean(raw_yield[idx[i-1]+1:idx[i], 2:end] |> Array, dims=1)
        end
    end
    yield_year = raw_yield[idx, :]
    yield_year = yield_year[findall(x -> x == yearmonth(date_start), yearmonth.(yield_year[:, 1]))[1]:findall(x -> x == yearmonth(date_end), yearmonth.(yield_year[:, 1]))[1], :]
    yields = DataFrame([Matrix(yield_month) Matrix(yield_year[:, 2:end])], [:M3, :M6, :Y1, :Y2, :Y3, :Y4, :Y5, :Y6, :Y7, :Y8, :Y9, :Y10])
    yields = [yield_year[:, 1] yields]
    rename!(yields, Dict(:x1 => "date"))
    yields = yields[3:end, :]
end

μϕ_const_PCs = -calibration_kQ_infty(μkQ_infty, σkQ_infty, 120, Array(yields[p_max-lag+1:end, 2:end]), τₙ, lag; medium_τ, iteration=10000)[2] |> x -> mean(x, dims=1)[1, :]
μϕ_const_PCs = [0.07, μϕ_const_PCs[2], μϕ_const_PCs[3]]
μϕ_const = [μϕ_const_PCs; zeros(size(macros, 2) - 1)]
@show calibration_kQ_infty(μkQ_infty, σkQ_infty, 120, Array(yields[p_max-lag+1:end, 2:end]), τₙ, lag; medium_τ, μϕ_const_PCs, iteration=10000)[1] |> mean

if step == 0 ## Drawing pareto frontier

    par_tuned = @showprogress 1 "Tuning..." pmap(1:p_max) do i
        tuning_hyperparameter_MOEA(Array(yields[p_max-i+1:end, 2:end]), Array(macros[p_max-i+1:end, 2:end]), τₙ, ρ; lag=i, μkQ_infty, σkQ_infty, upper_q, upper_ν0, medium_τ, μϕ_const)
    end
    pf = [par_tuned[i][1] for i in eachindex(par_tuned)]
    pf_input = [par_tuned[i][2] for i in eachindex(par_tuned)]
    opt = [par_tuned[i][3] for i in eachindex(par_tuned)]
    save("tuned_pf.jld2", "pf", pf, "pf_input", pf_input, "opt", opt)

elseif step == 1 ## Tuning hyperparameter

    if isfile("tuned_pf.jld2")
        pf = load("tuned_pf.jld2")["pf"]
        pf_input = load("tuned_pf.jld2")["pf_input"]
    end
    par_tuned = @showprogress 1 "Tuning..." pmap(1:p_max) do i
        x0 = []
        if isfile("tuned_pf.jld2")
            dP = size(macros, 2) - 1 + dimQ()
            tuned_ = pf_input[i][findall(x -> x < mSR_tail, pf[i][2])]
            x0 = Matrix{Float64}(undef, length(tuned_), 9)
            for j in eachindex(tuned_)
                x0[j, :] = [tuned_[j].q[1, 1] tuned_[j].q[2, 1] / tuned_[j].q[1, 1] tuned_[j].q[3, 1] tuned_[j].q[4, 1] tuned_[j].q[1, 2] tuned_[j].q[2, 2] / tuned_[j].q[1, 2] tuned_[j].q[3, 2] tuned_[j].q[4, 2] tuned_[j].ν0 - dP - 1]
            end
        end

        tuning_hyperparameter(Array(yields[p_max-i+1:end, 2:end]), Array(macros[p_max-i+1:end, 2:end]), τₙ, ρ; lag=i, upper_q, μkQ_infty, σkQ_infty, mSR_tail, initial=x0, upper_ν0, medium_τ, μϕ_const)
    end
    tuned = [par_tuned[i][1] for i in eachindex(par_tuned)]
    opt = [par_tuned[i][2] for i in eachindex(par_tuned)]
    save("tuned.jld2", "tuned", tuned, "opt", opt)

elseif step == 2 ## Estimation

    tuned = load("tuned.jld2")["tuned"][lag]
    if issparse_prec == false
        saved_θ, acceptPr_C_σ²FF, acceptPr_ηψ = posterior_sampler(Array(yields[p_max-lag+1:end, 2:end]), Array(macros[p_max-lag+1:end, 2:end]), τₙ, ρ, iteration, tuned; sparsity=issparse_coef, medium_τ)
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
    else
        saved_θ = load("posterior.jld2")["samples"]
        iteration = length(saved_θ)

        par_sparse_θ = @showprogress 1 "Sparse precision..." pmap(1:iteration) do i
            sparse_precision([saved_θ[i]], size(macros, 1) - tuned.p)
        end
        saved_θ = [par_sparse_θ[i][1][1] for i in eachindex(par_sparse_θ)]
        trace_sparsity = [par_sparse_θ[i][2][1] for i in eachindex(par_sparse_θ)]
        save("sparse.jld2", "samples", saved_θ, "sparsity", trace_sparsity)
    end

    if is_TP
        par_TP = @showprogress 1 "Term premium..." pmap(1:iteration) do i
            term_premium(TPτ_interest, τₙ, [saved_θ[i]], Array(yields[p_max-lag+1:end, 2:end]), Array(macros[p_max-lag+1:end, 2:end]))
        end
        saved_TP = [par_TP[i][1] for i in eachindex(par_TP)]
        save("TP.jld2", "TP", saved_TP)
    end

elseif step == 3

    #     if issparse_prec == false
    #         saved_θ = load("posterior.jld2")["samples"]
    #         iteration = length(saved_θ)
    #     else
    #         saved_θ = load("sparse.jld2")["samples"]
    #         iteration = length(saved_θ)
    #     end

    #     dt_length = length(yields[p_max-lag+1:end, 1]) - lag
    #     fitted_survey = Matrix{Float64}(undef, dt_length, 10)
    #     par_fitted = @showprogress 1 "Forecasting..." pmap(lag+1:lag+dt_length) do i
    #         ind_fitted = Vector{Float64}(undef, 10)
    #         PCAs = PCA(Array(yields[p_max-lag+1:end, 2:end]), lag) |> x -> (x[1][1:i, :], x[2][1:i, :], x[3], x[4], x[5])
    #         prediction = scenario_sampler([], [], 14, saved_θ, Array(yields[p_max-lag+1:p_max-lag+i, 2:end]), Array(macros[p_max-lag+1:p_max-lag+i, 2:end]), τₙ; PCAs) |> x -> mean(x)[:yields]
    #         for j in 1:5
    #             aux_box = [0, 2, 5, 8, 11, 14]
    #             ind_fitted[j] = prediction[aux_box[j]+1:aux_box[j+1], 1] |> mean
    #             ind_fitted[5+j] = prediction[aux_box[j]+1:aux_box[j+1], end] |> mean
    #         end
    #         ind_fitted
    #     end
    #     for i in axes(fitted_survey, 1)
    #         fitted_survey[i, :] = par_fitted[i]
    #     end
    #     save("survey.jld2", "fitted", fitted_survey)

elseif step == 4 ## Statistical inference

    # from step 0
    pf = load("tuned_pf.jld2")["pf"]
    pf_input = load("tuned_pf.jld2")["pf_input"]
    opt = load("tuned_pf.jld2")["opt"]
    #pf_plot = Plots.scatter(pf[:, 2], pf[:, 1], ylabel="marginal likelhood", xlabel="E[maximum SR]", label="")

    # from step 1
    if mSR_tail == Inf
        tuned_set = load("standard/tuned.jld2")["tuned"]
        tuned = tuned_set[lag]
        opt = load("standard/tuned.jld2")["opt"]
        mSR_prior = maximum_SR(Array(yields[p_max-lag+1:end, 2:end]), Array(macros[p_max-lag+1:end, 2:end]), tuned, τₙ, ρ; iteration=1000, medium_τ)
    else
        tuned_set = load("mSR/tuned.jld2")["tuned"]
        tuned = tuned_set[lag]
        opt = load("mSR/tuned.jld2")["opt"]
        mSR_prior = maximum_SR(Array(yields[p_max-lag+1:end, 2:end]), Array(macros[p_max-lag+1:end, 2:end]), tuned, τₙ, ρ; iteration=1000, medium_τ)
    end

    # from step 2 and 3
    if mSR_tail == Inf
        saved_θ = load("standard/posterior.jld2")["samples"]
        acceptPr = load("standard/posterior.jld2")["acceptPr"]
        accept_rate = load("standard/posterior.jld2")["accept_rate"]
        iteration = length(saved_θ)
        saved_TP = load("standard/TP.jld2")["TP"]
        # fitted_survey = load("standard/survey.jld2")["fitted"]
    elseif issparse_prec == true && issparse_coef == false
        saved_θ = load("mSR+prec/sparse.jld2")["samples"]
        trace_sparsity = load("mSR+prec/sparse.jld2")["sparsity"]
        acceptPr = load("mSR+prec/posterior.jld2")["acceptPr"]
        accept_rate = load("mSR+prec/posterior.jld2")["accept_rate"]
        iteration = length(saved_θ)
        saved_TP = load("mSR+prec/TP.jld2")["TP"]
        # fitted_survey = load("mSR+prec/survey.jld2")["fitted"]
    elseif issparse_prec == false && issparse_coef == true
        saved_θ = load("mSR+sparsity/posterior.jld2")["samples"]
        acceptPr = load("mSR+sparsity/posterior.jld2")["acceptPr"]
        accept_rate = load("mSR+sparsity/posterior.jld2")["accept_rate"]
        iteration = length(saved_θ)
        saved_TP = load("mSR+sparsity/TP.jld2")["TP"]
        # fitted_survey = load("mSR+sparsity/survey.jld2")["fitted"]
    elseif issparse_prec == true && issparse_coef == true
        saved_θ = load("mSR+sparsity+prec/sparse.jld2")["samples"]
        trace_sparsity = load("mSR+sparsity+prec/sparse.jld2")["sparsity"]
        acceptPr = load("mSR+sparsity+prec/posterior.jld2")["acceptPr"]
        accept_rate = load("mSR+sparsity+prec/posterior.jld2")["accept_rate"]
        iteration = length(saved_θ)
        saved_TP = load("mSR+sparsity+prec/TP.jld2")["TP"]
        # fitted_survey = load("mSR+sparsity+prec/survey.jld2")["fitted"]
    else
        saved_θ = load("mSR/posterior.jld2")["samples"]
        acceptPr = load("mSR/posterior.jld2")["acceptPr"]
        accept_rate = load("mSR/posterior.jld2")["accept_rate"]
        iteration = length(saved_θ)
        saved_TP = load("mSR/TP.jld2")["TP"]
        # fitted_survey = load("mSR/survey.jld2")["fitted"]
    end

    saved_Xθ = latentspace(saved_θ, Array(yields[p_max-lag+1:end, 2:end]), τₙ)
    fitted = fitted_YieldCurve(collect(1:τₙ[end]), saved_Xθ)
    fitted_yield = mean(fitted)[:yields] / 1200
    log_price = -collect(1:τₙ[end])' .* fitted_yield[tuned.p+1:end, :]
    xr = log_price[2:end, 1:end-1] - log_price[1:end-1, 2:end] .- fitted_yield[tuned.p+1:end-1, 1]
    realized_SR = mean(xr, dims=1) ./ std(xr, dims=1) |> x -> x[1, :]
    reduced_θ = reducedform(saved_θ, Array(yields[p_max-lag+1:end, 2:end]), Array(macros[p_max-lag+1:end, 2:end]), τₙ)
    mSR = mean(reduced_θ)[:mpr] |> x -> diag(x * x')

end