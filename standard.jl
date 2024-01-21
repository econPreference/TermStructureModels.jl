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
iteration = 25_000
burnin = 5_000
TPτ_interest = 120

##

function estimation(; upper_p, τₙ, medium_τ, iteration, burnin, ρ, macros, yields, μkQ_infty, σkQ_infty)

    sdate(yy, mm) = findall(x -> x == Date(yy, mm), macros[:, 1])[1]

    tuned = JLD2.load("standard/tuned.jld2")["tuned"]
    p = tuned.p

    saved_θ, acceptPrMH = posterior_sampler(Array(yields[upper_p-p+1:end, 2:end]), Array(macros[upper_p-p+1:end, 2:end]), τₙ, ρ, iteration, tuned; medium_τ, μkQ_infty, σkQ_infty)
    saved_θ = saved_θ[burnin+1:end]
    iteration = length(saved_θ)

    saved_θ, Pr_stationary = erase_nonstationary_param(saved_θ)
    iteration = length(saved_θ)
    JLD2.save("standard/posterior.jld2", "samples", saved_θ, "acceptPrMH", acceptPrMH, "Pr_stationary", Pr_stationary)

    ineff = ineff_factor(saved_θ)
    JLD2.save("standard/ineff.jld2", "ineff", ineff)

    iter_sub = JLD2.load("standard/ineff.jld2")["ineff"] |> x -> (x[1], x[2], x[3] |> maximum, x[4] |> maximum, x[5] |> maximum, x[6] |> maximum) |> maximum |> ceil |> Int
    saved_TP = term_premium(TPτ_interest, τₙ, saved_θ[1:iter_sub:end], Array(yields[upper_p-p+1:end, 2:end]), Array(macros[upper_p-p+1:end, 2:end]))
    JLD2.save("standard/TP.jld2", "TP", saved_TP)

    return saved_θ, p
end

function do_projection(saved_θ, p; upper_p, τₙ, macros, yields)

    sdate(yy, mm) = findall(x -> x == Date(yy, mm), macros[:, 1])[1]

    # Assumed scenario
    scenario_TP = [12, 24, 60, 120]
    scenario_horizon = 60
    scenario_start_date = Date("2022-12-01", "yyyy-mm-dd")
    function gen_scene(idx_case)
        idx_inflt = [21, 23]
        idx_ur = [5]

        if idx_case == 1
            scene = Vector{Scenario}(undef, 36)
            for h in 1:36
                combs = zeros(1, dP - dQ + length(τₙ))
                vals = [0.0]
                scene[h] = Scenario(combinations=deepcopy(combs), values=deepcopy(vals))
            end

            combs = [1 zeros(1, dP - dQ + length(τₙ) - 1)]
            vals = [5.1]
            scene[12] = Scenario(combinations=deepcopy(combs), values=deepcopy(vals))

            combs = [1 zeros(1, dP - dQ + length(τₙ) - 1)]
            vals = [4.1]
            scene[24] = Scenario(combinations=deepcopy(combs), values=deepcopy(vals))

            combs = [1 zeros(1, dP - dQ + length(τₙ) - 1)]
            vals = [3.1]
            scene[end] = Scenario(combinations=deepcopy(combs), values=deepcopy(vals))
            return scene
        elseif idx_case == 2
            scene = Vector{Scenario}(undef, 10)
            VIX_path = raw_macros[sdate(2008, 9):sdate(2009, 6), end]
            for h in 1:10
                combs = zeros(1, dP - dQ + length(τₙ))
                vals = zeros(size(combs, 1))

                combs[1, end] = 1.0
                vals[1] = 100log(VIX_path[h]) - mean_macros[end]
                scene[h] = Scenario(combinations=deepcopy(combs), values=deepcopy(vals))
            end
            return scene
        end
    end

    ## Do 
    iter_sub = JLD2.load("standard/ineff.jld2")["ineff"] |> x -> (x[1], x[2], x[3] |> maximum, x[4] |> maximum, x[5] |> maximum, x[6] |> maximum) |> maximum |> ceil |> Int

    # unconditional prediction
    projections = scenario_analysis([], scenario_TP, scenario_horizon, saved_θ[1:iter_sub:end], Array(yields[upper_p-p+1:sdate(yearmonth(scenario_start_date)...), 2:end]), Array(macros[upper_p-p+1:sdate(yearmonth(scenario_start_date)...), 2:end]), τₙ; mean_macros)
    JLD2.save("standard/uncond_scenario.jld2", "projections", projections)
    # conditional prediction
    for i in 1:2
        projections = scenario_analysis(gen_scene(i), scenario_TP, scenario_horizon, saved_θ[1:iter_sub:end], Array(yields[upper_p-p+1:sdate(yearmonth(scenario_start_date)...), 2:end]), Array(macros[upper_p-p+1:sdate(yearmonth(scenario_start_date)...), 2:end]), τₙ; mean_macros)
        JLD2.save("standard/scenario$i.jld2", "projections", projections)
    end

    return []
end

function inferences(; upper_p, τₙ, macros, yields)

    sdate(yy, mm) = findall(x -> x == Date(yy, mm), macros[:, 1])[1]

    # from step 1
    opt = JLD2.load("standard/tuned.jld2")["opt"]
    tuned = JLD2.load("standard/tuned.jld2")["tuned"]
    p = tuned.p

    # from step 2
    saved_θ = JLD2.load("standard/posterior.jld2")["samples"]
    acceptPrMH = JLD2.load("standard/posterior.jld2")["acceptPrMH"]
    Pr_stationary = JLD2.load("standard/posterior.jld2")["Pr_stationary"]
    saved_TP = JLD2.load("standard/TP.jld2")["TP"]
    ineff = JLD2.load("standard/ineff.jld2")["ineff"]

    ## Convergence of MCMC
    @show (ineff[1], ineff[2], ineff[3] |> maximum, ineff[4] |> maximum, ineff[5] |> maximum, ineff[6] |> maximum)
    ineff_samples = Matrix{Float64}(undef, length(saved_θ), 6)
    ineff_samples[:, 1] = saved_θ[:κQ]
    ineff_samples[:, 2] = saved_θ[:kQ_infty]
    ineff_samples[:, 3] = ineff[3] |> findmax |> x -> x[2] |> x -> [saved_θ[:γ][i][x] for i in 1:length(saved_θ)]
    ineff_samples[:, 4] = ineff[4] |> findmax |> x -> x[2] |> x -> [saved_θ[:Σₒ][i][x] for i in 1:length(saved_θ)]
    ineff_samples[:, 5] = ineff[5] |> findmax |> x -> x[2] |> x -> [saved_θ[:σ²FF][i][x] for i in 1:length(saved_θ)]
    ineff_samples[:, 6] = ineff[6] |> findmax |> x -> x[2] |> x -> [saved_θ[:ϕ][i][x] for i in 1:length(saved_θ)]

    ineff_means = Matrix{Float64}(undef, length(saved_θ) - 99, 6)
    ineff_stds = Matrix{Float64}(undef, length(saved_θ) - 99, 6)
    for i in 100:length(saved_θ), j in 1:6
        ineff_means[i-99, j] = mean(ineff_samples[i-99:i, j])
        ineff_stds[i-99, j] = std(ineff_samples[i-99:i, j])
    end
    for i in axes(ineff_samples, 2)
        @show minimum(ineff_means[:, i]), median(ineff_means[:, i]), maximum(ineff_means[:, i])
        @show minimum(ineff_stds[:, i]), median(ineff_stds[:, i]), maximum(ineff_stds[:, i])
    end

    ## additional inferences
    saved_Xθ = latentspace(saved_θ, Array(yields[upper_p-p+1:end, 2:end]), τₙ)
    fits = fitted_YieldCurve(collect(1:τₙ[end]), saved_Xθ)
    decimal_yield = mean(fits)[:yields] / 1200
    log_price = -collect(1:τₙ[end])' .* decimal_yield[p:end, :]
    xr = log_price[2:end, 1:end-1] - log_price[1:end-1, 2:end] .- decimal_yield[p:end-1, 1]
    realized_SR = mean(xr, dims=1) ./ std(xr, dims=1) |> x -> x[1, :]
    reduced_θ = reducedform(saved_θ, Array(yields[upper_p-p+1:end, 2:end]), Array(macros[upper_p-p+1:end, 2:end]), τₙ)
    mSR = [reduced_θ[:mpr][i] |> x -> sqrt.(diag(x * x')) for i in eachindex(reduced_θ)] |> mean

    return opt, tuned, saved_θ, acceptPrMH, Pr_stationary, saved_TP, ineff, saved_Xθ, fits, realized_SR, reduced_θ, mSR
end

function graphs(; medium_τ, macros, yields, tuned, saved_θ, saved_TP, fits, reduced_θ)

    sdate(yy, mm) = findall(x -> x == Date(yy, mm), macros[:, 1])[1]
    TP_nolag = JLD2.load("nolag/TP.jld2")["TP"]
    TP_nomacro = JLD2.load("nomacro/TP.jld2")["TP"]

    set_default_plot_size(16cm, 8cm)

    ## check optimization performance
    PCs, ~, Wₚ = PCA(Array(yields[upper_p-tuned.p+1:end, 2:end]), tuned.p)
    ml_results = Vector{Vector}(undef, 10)
    ranges = [[collect(range(eps(), tuned.q[1, 1], length=6)); collect(range(tuned.q[1, 1], 0.5, length=6))],
        [collect(range(eps(), tuned.q[2, 1], length=6)); collect(range(tuned.q[2, 1], 0.05, length=6))],
        [collect(range(eps(), tuned.q[3, 1], length=6)); collect(range(tuned.q[3, 1], 8, length=6))],
        [collect(range(eps(), tuned.q[4, 1], length=6)); collect(range(tuned.q[4, 1], 0.03, length=6))],
        [collect(range(eps(), tuned.q[1, 2], length=6)); collect(range(tuned.q[1, 2], 0.5, length=6))],
        [collect(range(eps(), tuned.q[2, 2], length=6)); collect(range(tuned.q[2, 2], 0.05, length=6))],
        [collect(range(eps(), tuned.q[3, 2], length=6)); collect(range(tuned.q[3, 2], 8, length=6))],
        collect(range(eps(), 0.03, length=11)),
        [collect(range(33, tuned.ν0, length=6)); collect(range(tuned.ν0, 45, length=6))],
        collect(1:upper_p)]
    prog = Progress(10; dt=1, desc="plotting the optimization graphs")
    Threads.@threads for i in 1:10
        ind_range = ranges[i]
        if i == 10
            ml = Vector{Float64}(undef, upper_p)
            for j in eachindex(ind_range)
                PCs1 = PCA(Array(yields[upper_p-Int(ind_range[j])+1:end, 2:end]), Int(ind_range[j]))[1]
                ml[j] = log_marginal(PCs1, Array(macros[upper_p-Int(ind_range[j])+1:end, 2:end]), ρ, Hyperparameter(p=Int(ind_range[j]), q=tuned.q, ν0=tuned.ν0, Ω0=tuned.Ω0, μϕ_const=tuned.μϕ_const), τₙ, Wₚ; medium_τ, medium_τ_pr=ones(length(medium_τ)) ./ length(medium_τ), fix_const_PC1=false)
            end
            ml_results[i] = deepcopy(ml)
        elseif i == 9
            ml = Vector{Float64}(undef, length(ind_range))
            for j in eachindex(ind_range)
                ml[j] = log_marginal(PCs, Array(macros[upper_p-tuned.p+1:end, 2:end]), ρ, Hyperparameter(p=tuned.p, q=tuned.q, ν0=ind_range[j], Ω0=tuned.Ω0, μϕ_const=tuned.μϕ_const), τₙ, Wₚ; medium_τ, medium_τ_pr=ones(length(medium_τ)) ./ length(medium_τ), fix_const_PC1=false)
            end
            ml_results[i] = deepcopy(ml)
        elseif i > 4
            ml = Vector{Float64}(undef, length(ind_range))
            for j in eachindex(ind_range)
                q = deepcopy(tuned.q)
                q[i-4, 2] = ind_range[j]
                ml[j] = log_marginal(PCs, Array(macros[upper_p-tuned.p+1:end, 2:end]), ρ, Hyperparameter(p=tuned.p, q=q, ν0=tuned.ν0, Ω0=tuned.Ω0, μϕ_const=tuned.μϕ_const), τₙ, Wₚ; medium_τ, medium_τ_pr=ones(length(medium_τ)) ./ length(medium_τ), fix_const_PC1=false)
            end
            ml_results[i] = deepcopy(ml)
        else
            ml = Vector{Float64}(undef, length(ind_range))
            for j in eachindex(ind_range)
                q = deepcopy(tuned.q)
                q[i, 1] = ind_range[j]
                ml[j] = log_marginal(PCs, Array(macros[upper_p-tuned.p+1:end, 2:end]), ρ, Hyperparameter(p=tuned.p, q=q, ν0=tuned.ν0, Ω0=tuned.Ω0, μϕ_const=tuned.μϕ_const), τₙ, Wₚ; medium_τ, medium_τ_pr=ones(length(medium_τ)) ./ length(medium_τ), fix_const_PC1=false)
            end
            ml_results[i] = deepcopy(ml)
        end
        next!(prog)
    end
    finish!(prog)

    Plots.plot(ranges[1], ml_results[1], ylabel="log marginal likelihood", label="", linewidth=3)
    Plots.plot!(ranges[5], ml_results[5], xlabel=L"q_{11}(solid), q_{12}(dashed)", ylabel="log marginal likelihood", label="", linewidth=3, ls=:dash, ylims=(-38880, -38845), xtickfontsize=12, ytickfontsize=10, xguidefontsize=16, yguidefontsize=14) |> x -> Plots.pdf(x, "/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior_for_GDTSM/slide/ML_q1.pdf")
    Plots.plot(ranges[2], ml_results[2], ylabel="log marginal likelihood", label="", linewidth=3)
    Plots.plot!(ranges[6], ml_results[6], xlabel=L"q_{21}(solid), q_{22}(dashed)", ylabel="log marginal likelihood", label="", linewidth=3, ls=:dash, ylims=(-38950, -38840), xtickfontsize=12, ytickfontsize=10, xguidefontsize=16, yguidefontsize=14) |> x -> Plots.pdf(x, "/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior_for_GDTSM/slide/ML_q2.pdf")
    Plots.plot(ranges[3], ml_results[3], ylabel="log marginal likelihood", label="", linewidth=3)
    Plots.plot!(ranges[7], ml_results[7], xlabel=L"q_{31}(solid), q_{32}(dashed)", ylabel="log marginal likelihood", label="", linewidth=3, ls=:dash, ylims=(-39100, -38840), xtickfontsize=12, ytickfontsize=10, xguidefontsize=16, yguidefontsize=14) |> x -> Plots.pdf(x, "/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior_for_GDTSM/slide/ML_q3.pdf")
    Plots.plot(ranges[4], ml_results[4], ylabel="log marginal likelihood", label="", linewidth=3)
    Plots.plot!(ranges[8], ml_results[8], xlabel=L"q_{41}(solid), q_{42}(dashed)", ylabel="log marginal likelihood", label="", linewidth=3, ls=:dash, ylims=(-38852, -38846), xtickfontsize=12, ytickfontsize=10, xguidefontsize=16, yguidefontsize=14) |> x -> Plots.pdf(x, "/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior_for_GDTSM/slide/ML_q4.pdf")
    Plots.plot(ranges[9], ml_results[9], xlabel=L"ν_{0}", ylabel="log marginal likelihood", label="", linewidth=3, xtickfontsize=12, ytickfontsize=10, xguidefontsize=16, yguidefontsize=14) |> x -> Plots.pdf(x, "/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior_for_GDTSM/slide/ML_nu0.pdf")
    Plots.plot(ranges[10], ml_results[10], xlabel=L"p", ylabel="log marginal likelihood", label="", linewidth=3, xticks=1:2:upper_p, xtickfontsize=12, ytickfontsize=10, xguidefontsize=16, yguidefontsize=14) |> x -> Plots.pdf(x, "/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior_for_GDTSM/slide/ML_lag.pdf")

    ## decay parameter
    medium_τ_pr = length(medium_τ) |> x -> ones(x) / x
    κQ_support = [reverse(medium_τ) support(prior_κQ(medium_τ, medium_τ_pr))]
    Plots.histogram(saved_θ[:κQ], xticks=(κQ_support[:, 2], ["$(round(κQ_support[i,2],digits=4))\n(τ = $(round(Int,κQ_support[i,1])))" for i in axes(κQ_support, 1)]), bins=40, xlabel=L"\kappa_{Q} ( maturity \, \tau )", labels="") |> x -> Plots.pdf(x, "/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior_for_GDTSM/slide/decay.pdf")

    ## Transition equation error covarinace
    upper_idx = findall(x -> x > 0, quantile(reduced_θ, 0.025)[:ΩFF])
    lower_idx = findall(x -> x < 0, quantile(reduced_θ, 0.975)[:ΩFF])
    corr = Vector{Matrix}(undef, length(reduced_θ))
    for i in eachindex(reduced_θ)
        corr[i] = reduced_θ[:ΩFF][i] |> x -> diagm(1 ./ sqrt.(diag(x))) * x * diagm(1 ./ sqrt.(diag(x)))
    end
    corr_mean = mean(corr)

    sparse_corr_mean = zeros(dP, dP)
    for i in eachindex(upper_idx)
        sparse_corr_mean[upper_idx[i]] = corr_mean[upper_idx[i]]
    end
    for i in eachindex(lower_idx)
        sparse_corr_mean[lower_idx[i]] = corr_mean[lower_idx[i]]
    end
    for i in 1:dP, j in (i+1):dP
        sparse_corr_mean[i, j] = 0.0
    end
    Plots.heatmap(sparse_corr_mean, c=:RdBu, clim=(-1, 1), yflip=true)
    Plots.xticks!([1:31;], ["PC1"; "PC2"; "PC3"; names(macros[:, 2:end])], xrot=90)
    Plots.yticks!([1:31;], ["PC1"; "PC2"; "PC3"; names(macros[:, 2:end])], margin=10mm, size=(1000, 900), xtickfontsize=10, ytickfontsize=10) |> x -> Plots.pdf(x, "/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior_for_GDTSM/slide/cor.pdf")

    ## TP components
    rec_dates = DateTime.(["1990-07-01" "1991-03-01"
        "2001-03-01" "2001-11-01"
        "2007-12-01" "2009-06-01"
        "2020-02-01" "2020-04-01"])

    plot(
        layer(x=yields[sdate(1987, 1):end, 1], y=mean(saved_TP)[:TP], Geom.line, color=[colorant"#4682B4"], Theme(line_width=2pt)),
        layer(x=yields[sdate(1987, 1):end, 1], y=quantile(saved_TP, 0.025)[:TP], Geom.line, color=[colorant"#A9A9A9"], Theme(line_width=0.5pt, line_style=[:dash])),
        layer(x=yields[sdate(1987, 1):end, 1], y=quantile(saved_TP, 0.975)[:TP], Geom.line, color=[colorant"#A9A9A9"], Theme(line_width=0.5pt, line_style=[:dash])),
        layer(xmin=rec_dates[:, 1], xmax=rec_dates[:, 2], Geom.band(; orientation=:vertical), color=[colorant"#DCDCDC"]),
        Theme(major_label_font_size=10pt, minor_label_font_size=9pt, key_label_font_size=10pt, point_size=4pt), Guide.ylabel("percent per annum"), Guide.xlabel(""), Guide.xticks(ticks=DateTime("1986-07-01"):Month(54):DateTime("2023-06-01")), Guide.yticks(ticks=-4:2:4)
    ) |> PDF("/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior_for_GDTSM/slide/TP10.pdf")

    plot(
        layer(x=yields[sdate(1987, 1):end, 1], y=mean(saved_TP)[:TP], Geom.line, color=[colorant"#000000"], Theme(line_width=2pt)),
        layer(x=yields[sdate(1987, 1):end, 1], y=mean(TP_nomacro)[:TP], Geom.line, color=[colorant"#4682B4"], Theme(line_width=2pt, line_style=[:dash])),
        layer(x=yields[sdate(1987, 1):end, 1], y=mean(TP_nolag)[:TP], Geom.point, Geom.line, color=[colorant"#DC143C"], Theme(line_style=[:dot], point_size=2pt)),
        layer(xmin=rec_dates[:, 1], xmax=rec_dates[:, 2], Geom.band(; orientation=:vertical), color=[colorant"#DCDCDC"]),
        Theme(major_label_font_size=10pt, minor_label_font_size=9pt, key_label_font_size=10pt, point_size=4pt), Guide.ylabel("percent per annum"), Guide.xlabel(""), Guide.xticks(ticks=DateTime("1986-07-01"):Month(54):DateTime("2023-06-01")), Guide.yticks(ticks=-2:2:4)
    ) |> PDF("/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior_for_GDTSM/slide/TPs.pdf")

    ## individual TP components
    ind_TP_ratio = mean(saved_TP)[:timevarying_TP] |> x -> var(x, dims=1) ./ var(mean(saved_TP)[:TP]) |> x -> x[1, :]
    ind_TP_order = sortperm(ind_TP_ratio, rev=true)
    n_top = 5
    ind_TP_names = ["PC1"; "PC2"; "PC3"; names(macros[1, 2:end])]
    ind_TP_names[findall(x -> x .== "S&P 500", ind_TP_names)[1]] = "SP 500"

    ind_TP = mean(saved_TP)[:timevarying_TP][:, ind_TP_order[1]]
    mesh = [yields[sdate(1987, 1):end, 1] fill(ind_TP_names[ind_TP_order[1]], length(ind_TP)) ind_TP]
    for i in 2:n_top
        ind_TP = mean(saved_TP)[:timevarying_TP][:, ind_TP_order[i]]
        mesh = [mesh; [yields[sdate(1987, 1):end, 1] fill(ind_TP_names[ind_TP_order[i]], length(ind_TP)) ind_TP]]
    end
    df = DataFrame(dates=Date.(string.(mesh[:, 1]), DateFormat("yyyy-mm-dd")), macros=string.(mesh[:, 2]), TP=Float64.(mesh[:, 3]))

    plot(df,
        layer(x=:dates, y=:TP, Geom.line, color=:macros, Theme(line_width=1pt)),
        layer(xmin=rec_dates[:, 1], xmax=rec_dates[:, 2], Geom.band(; orientation=:vertical), color=[colorant"#DCDCDC"]),
        Theme(major_label_font_size=10pt, minor_label_font_size=9pt, key_label_font_size=9pt, point_size=4pt), Guide.ylabel("percent per annum"), Guide.xlabel(""), Guide.xticks(ticks=DateTime("1986-07-01"):Month(72):DateTime("2023-06-01"), orientation=:horizontal),
        Guide.yticks(ticks=[-8; collect(-6:2:10)])
    ) |> PDF("/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior_for_GDTSM/slide/ind_TP10.pdf")

    ## EH components
    plot(
        layer(x=yields[sdate(1987, 1):end, 1], y=mean(fits)[:yields][tuned.p+1:end, end] - mean(saved_TP)[:TP], Geom.line, color=[colorant"#4682B4"], Theme(line_width=2pt)),
        layer(xmin=rec_dates[:, 1], xmax=rec_dates[:, 2], Geom.band(; orientation=:vertical), color=[colorant"#DCDCDC"]),
        Theme(major_label_font_size=10pt, minor_label_font_size=9pt, key_label_font_size=10pt, point_size=4pt), Guide.ylabel("percent per annum"), Guide.xlabel(""), Guide.xticks(ticks=DateTime("1986-07-01"):Month(54):DateTime("2023-06-01")), Guide.yticks(ticks=[0; collect(1:6)])
    ) |> PDF("/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior_for_GDTSM/slide/EH10.pdf")

end

function scenario_graphs(idx_case, is_control::Bool, is_level::Bool; τₙ, macros)

    set_default_plot_size(16cm, 8cm)
    scenario_start_date = Date("2022-12-01", "yyyy-mm-dd")
    idx_date = sdate(yearmonth(scenario_start_date)...)
    if idx_case == 1
        horizon = 60
    elseif idx_case == 2
        horizon = 36
    end
    if idx_case == 1
        macros_of_interest = ["INDPRO", "PERMIT", "REALLN", "S&P 500", "AAA", "BAA", "OILPRICEx", "PPICMM", "PCEPI", "DTCTHFNM", "INVEST", "VIXCLSx"]
    else
        macros_of_interest = ["INDPRO", "CUMFNS", "UNRATE", "PERMIT", "S&P 500", "AAA", "BAA", "OILPRICEx", "PPICMM", "PCEPI", "UMCSENTx", "DTCTHFNM"]
    end

    ## constructing predictions
    # load results
    raw_projections = JLD2.load("standard/scenario$idx_case.jld2")["projections"]
    projections = Vector{Forecast}(undef, length(raw_projections))
    for i in eachindex(projections)
        predicted_factors = deepcopy(raw_projections[i][:factors])
        for j in 1:dP-dQ
            if idx_diff[j] == 1 && is_percent[j]
                predicted_factors[:, dQ+j] = [raw_macros[idx_date, 1+j]; predicted_factors[:, dQ+j]] |> cumsum |> x -> x[2:end]
            elseif idx_diff[j] == 0 && !is_percent[j]
                predicted_factors[:, dQ+j] = [logmacros[idx_date-11:idx_date, j]; predicted_factors[:, dQ+j]] |> x -> [x[t] - x[t-12] for t in 13:length(x)]
            elseif idx_diff[j] == 1 && !is_percent[j]
                predicted_factors[:, dQ+j] = [logmacros[idx_date, j]; predicted_factors[:, dQ+j] ./ 12] |> cumsum |> x -> [logmacros[idx_date-11:idx_date, j]; x[2:end]] |> x -> [x[t] - x[t-12] for t in 13:length(x)]
            end
        end
        projections[i] = Forecast(yields=deepcopy(raw_projections[i][:yields]), factors=deepcopy(predicted_factors), TP=deepcopy(raw_projections[i][:TP]))
    end

    if is_control # if there is a control group
        raw_projections = JLD2.load("standard/uncond_scenario.jld2")["projections"]
        for i in eachindex(projections)
            predicted_factors = deepcopy(raw_projections[i][:factors])
            for j in 1:dP-dQ
                if idx_diff[j] == 1 && is_percent[j]
                    predicted_factors[:, dQ+j] = [raw_macros[idx_date, 1+j]; predicted_factors[:, dQ+j]] |> cumsum |> x -> x[2:end]
                elseif idx_diff[j] == 0 && !is_percent[j]
                    predicted_factors[:, dQ+j] = [logmacros[idx_date-11:idx_date, j]; predicted_factors[:, dQ+j]] |> x -> [x[t] - x[t-12] for t in 13:length(x)]
                elseif idx_diff[j] == 1 && !is_percent[j]
                    predicted_factors[:, dQ+j] = [logmacros[idx_date, j]; predicted_factors[:, dQ+j] ./ 12] |> cumsum |> x -> [logmacros[idx_date-11:idx_date, j]; x[2:end]] |> x -> [x[t] - x[t-12] for t in 13:length(x)]
                end
            end
            projections[i] = Forecast(yields=deepcopy(projections[i][:yields] - raw_projections[i][:yields]), factors=deepcopy(projections[i][:factors] - predicted_factors), TP=deepcopy(projections[i][:TP] - raw_projections[i][:TP]))
        end
    end

    if is_level
        if is_control
            for i in eachindex(projections)
                predicted_factors = deepcopy(projections[i][:factors])
                for j in 1:dP-dQ
                    if !is_percent[j]
                        for k in 13:size(predicted_factors, 1)
                            predicted_factors[k, dQ+j] += predicted_factors[k-12, dQ+j]
                        end
                    end
                end
                projections[i] = Forecast(yields=deepcopy(projections[i][:yields]), factors=deepcopy(predicted_factors), TP=deepcopy(projections[i][:TP]))
            end
        else
            for i in eachindex(projections)
                predicted_factors = deepcopy(projections[i][:factors])
                for j in 1:dP-dQ
                    if !is_percent[j]
                        aux = [logmacros[idx_date-11:idx_date, j]; predicted_factors[:, dQ+j]]
                        for k in 13:size(aux, 1)
                            aux[k] += aux[k-12]
                        end
                        aux ./= 100
                        predicted_factors[:, dQ+j] = exp.(aux[13:end])
                    end
                end
                projections[i] = Forecast(yields=deepcopy(projections[i][:yields]), factors=deepcopy(predicted_factors), TP=deepcopy(projections[i][:TP]))
            end
        end
    end

    # yields
    yield_res = mean(projections)[:yields]
    Plots.heatmap(τₙ, 1:horizon, yield_res[1:horizon, :], xlabel="maturity (months)", ylabel="horizon (months)", c=:Blues) |> x -> Plots.pdf(x, "/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior_for_GDTSM/slide/proj3D_yield$idx_case,control=$is_control,level=$is_level.pdf")

    p = []
    for i in [2, 7, 18]
        ind_p = Plots.plot(1:horizon, mean(projections)[:yields][1:horizon, i], fillrange=quantile(projections, 0.16)[:yields][1:horizon, i], labels="", title="yields(τ = $(τₙ[i]))", titlefontsize=10, c=colorant"#4682B4", alpha=0.6)
        Plots.plot!(ind_p, 1:horizon, mean(projections)[:yields][1:horizon, i], fillrange=quantile(projections, 0.84)[:yields][1:horizon, i], labels="", c=colorant"#4682B4", alpha=0.6)
        Plots.plot!(ind_p, 1:horizon, mean(projections)[:yields][1:horizon, i], fillrange=quantile(projections, 0.025)[:yields][1:horizon, i], labels="", c=colorant"#4682B4", alpha=0.6)
        Plots.plot!(ind_p, 1:horizon, mean(projections)[:yields][1:horizon, i], fillrange=quantile(projections, 0.975)[:yields][1:horizon, i], labels="", c=colorant"#4682B4", alpha=0.6)
        push!(p, ind_p)
    end
    Plots.plot(p[1], p[2], p[3], layout=(1, 3), xlabel="", size=(600, 200)) |> x -> Plots.pdf(x, "/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior_for_GDTSM/slide/proj_yield$idx_case,control=$is_control,level=$is_level.pdf")

    # EH & TP
    EHTP_res = [mean(projections)[:yields][:, [7, 13]] - mean(projections)[:TP][:, 2:3] mean(projections)[:TP][:, 4]]
    EH_res_dist_24 = Matrix{Float64}(undef, length(projections), size(EHTP_res, 1))
    for i in axes(EH_res_dist_24, 1)
        EH_res_dist_24[i, :] = projections[:yields][i][:, 7] - projections[:TP][i][:, 2]
    end
    EH_res_dist_60 = Matrix{Float64}(undef, length(projections), size(EHTP_res, 1))
    for i in axes(EH_res_dist_60, 1)
        EH_res_dist_60[i, :] = projections[:yields][i][:, 13] - projections[:TP][i][:, 3]
    end
    TP_res_dist_120 = Matrix{Float64}(undef, length(projections), size(EHTP_res, 1))
    for i in axes(TP_res_dist_120, 1)
        TP_res_dist_120[i, :] = projections[:TP][i][:, 4]
    end

    p = []
    for i in 1:3
        if i == 1
            EHTP_res_dist = deepcopy(EH_res_dist_24)
            ind_name = "EH(τ = 24)"
        elseif i == 2
            EHTP_res_dist = deepcopy(EH_res_dist_60)
            ind_name = "EH(τ = 60)"
        else
            EHTP_res_dist = deepcopy(TP_res_dist_120)
            ind_name = "TP(τ = 120)"
        end
        ind_p = Plots.plot(1:horizon, EHTP_res[1:horizon, i], fillrange=[quantile(EHTP_res_dist[:, j], 0.16) for j in axes(EHTP_res_dist, 2)][1:horizon], labels="", title=ind_name, titlefontsize=10, c=colorant"#4682B4", alpha=0.6)
        Plots.plot!(ind_p, 1:horizon, EHTP_res[1:horizon, i], fillrange=[quantile(EHTP_res_dist[:, j], 0.84) for j in axes(EHTP_res_dist, 2)][1:horizon], labels="", c=colorant"#4682B4", alpha=0.6)
        Plots.plot!(ind_p, 1:horizon, EHTP_res[1:horizon, i], fillrange=[quantile(EHTP_res_dist[:, j], 0.025) for j in axes(EHTP_res_dist, 2)][1:horizon], labels="", c=colorant"#4682B4", alpha=0.6)
        Plots.plot!(ind_p, 1:horizon, EHTP_res[1:horizon, i], fillrange=[quantile(EHTP_res_dist[:, j], 0.975) for j in axes(EHTP_res_dist, 2)][1:horizon], labels="", c=colorant"#4682B4", alpha=0.6)
        push!(p, ind_p)
    end
    Plots.plot(p[1], p[2], p[3], layout=(1, 3), xlabel="", size=(600, 200)) |> x -> Plots.pdf(x, "/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior_for_GDTSM/slide/proj_EHTP$idx_case,control=$is_control,level=$is_level.pdf")

    # macros
    p = []
    for i in macros_of_interest
        ind_macro = findall(x -> x == string(i), names(macros[1, 2:end]))[1]

        ind_p = Plots.plot(1:horizon, mean(projections)[:factors][1:horizon, dimQ()+ind_macro], fillrange=quantile(projections, 0.025)[:factors][1:horizon, dimQ()+ind_macro], labels="", title=string(i), titlefontsize=10, c=colorant"#4682B4", alpha=0.6)
        Plots.plot!(ind_p, 1:horizon, mean(projections)[:factors][1:horizon, dimQ()+ind_macro], fillrange=quantile(projections, 0.975)[:factors][1:horizon, dimQ()+ind_macro], c=colorant"#4682B4", label="", fillalpha=0.6)
        Plots.plot!(ind_p, 1:horizon, mean(projections)[:factors][1:horizon, dimQ()+ind_macro], fillrange=quantile(projections, 0.16)[:factors][1:horizon, dimQ()+ind_macro], c=colorant"#4682B4", label="", fillalpha=0.6)
        Plots.plot!(ind_p, 1:horizon, mean(projections)[:factors][1:horizon, dimQ()+ind_macro], fillrange=quantile(projections, 0.84)[:factors][1:horizon, dimQ()+ind_macro], c=colorant"#4682B4", label="", fillalpha=0.6)
        push!(p, ind_p)
    end
    Plots.plot(p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], layout=(3, 4), xlabel="", size=(800, 600)) |> x -> Plots.pdf(x, "/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior_for_GDTSM/slide/proj_macro$idx_case,control=$is_control,level=$is_level.pdf")

end

## Do
read_only = false
if read_only
    opt, tuned, saved_θ, acceptPrMH, Pr_stationary, saved_TP, ineff, saved_Xθ, fits, realized_SR, reduced_θ, mSR = inferences(; upper_p, τₙ, macros, yields)
else
    tuned, opt = tuning_hyperparameter(Array(yields[:, 2:end]), Array(macros[:, 2:end]), τₙ, ρ; upper_p, upper_q, μkQ_infty, σkQ_infty, medium_τ)
    JLD2.save("standard/tuned.jld2", "tuned", tuned, "opt", opt)

    saved_θ, p = estimation(; upper_p, τₙ, medium_τ, iteration, burnin, ρ, macros, yields, μkQ_infty, σkQ_infty)

    do_projection(saved_θ, p; upper_p, τₙ, macros, yields)

    opt, tuned, saved_θ, acceptPrMH, Pr_stationary, saved_TP, ineff, saved_Xθ, fits, realized_SR, reduced_θ, mSR = inferences(; upper_p, τₙ, macros, yields)

    graphs(; medium_τ, macros, yields, tuned, saved_θ, saved_TP, fits, reduced_θ)

    for i in 1:2, j = [true, false], k = [true, false]
        scenario_graphs(i, j, k; τₙ, macros)
    end
end
