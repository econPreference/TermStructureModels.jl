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
            macros_growth[2:end, i] = log.(macros[2:end, i+1]) - log.(macros[1:end-1, i+1]) |> x -> 1200 * x
            macros[:, i+1] = log.(macros[:, i+1]) |> x -> 100 * x
            ρ[i] = 1.0
            idx_diff[i] = 0
        else
            macros_growth[2:end, i] = log.(macros[2:end, i+1]) - log.(macros[1:end-1, i+1]) |> x -> 1200 * x
            macros[2:end, i+1] = macros_growth[2:end, i]
            ρ[i] = 0.0
            idx_diff[i] = 1
        end
    end

    raw_macros = raw_macros[3:end, :]
    macros = macros[3:end, :]
    macros_growth = macros_growth[3:end, :]
    mean_macros = mean(macros[:, 2:end] |> Array, dims=1)[1, :]
    macros[:, 2:end] .-= mean_macros'

    ## Yield data
    raw_yield = XLSX.readdata("LW_monthly.xlsx", "Sheet1", "A293:DQ748") |> x -> [Date.(string.(x[:, 1]), DateFormat("yyyymm")) convert(Matrix{Float64}, x[:, τₙ.+1])] |> x -> DataFrame(x, ["date"; ["Y$i" for i in τₙ]])
    yields = raw_yield[findall(x -> x == yearmonth(date_start), yearmonth.(raw_yield[:, 1]))[1]:findall(x -> x == yearmonth(date_end), yearmonth.(raw_yield[:, 1]))[1], :]
    yields = yields[3:end, :]

    yields = [Date.(string.(yields[:, 1]), DateFormat("yyyy-mm-dd")) Float64.(yields[:, 2:end])]
    rename!(yields, Dict(:x1 => "date"))

    return ρ, is_percent, idx_diff, macros_growth, raw_macros, macros, mean_macros, yields
end
ρ, is_percent, idx_diff, macros_growth, raw_macros, macros, mean_macros, yields = data_loading(; date_start, date_end, τₙ)
sdate(yy, mm) = findall(x -> x == Date(yy, mm), macros[:, 1])[1]

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

# scenario analysis
dQ = dimQ()
dP = size(macros, 2) - 1 + dQ
scenario_TP = [24, 120]
scenario_horizon = 12
scenario_start_date = Date("2022-01-01", "yyyy-mm-dd")
function gen_scene(scale)

    scene = Vector{Scenario}(undef, 0)
    combs = zeros(1, dP - dQ + length(τₙ))
    vals = zeros(size(combs, 1))

    combs[1, 1] = 1
    vals[1] = scale * yields[sdate(2022, 1), 2]
    push!(scene, Scenario(combinations=deepcopy(combs), values=deepcopy(vals)))
    for h = 2:12
        local combs = zeros(1, dP - dQ + length(τₙ))
        local combs[1, 1] = 1
        local vals = [scale * yields[sdate(2022, h), 2]]
        push!(scene, Scenario(combinations=deepcopy(combs), values=deepcopy(vals)))
    end
    return scene
end
##

function estimation(; upper_p, τₙ, medium_τ, iteration, burnin, ρ, macros, mean_macros, yields, scenario_TP, scenario_horizon, scenario_start_date, μkQ_infty, σkQ_infty)

    sdate(yy, mm) = findall(x -> x == Date(yy, mm), macros[:, 1])[1]

    tuned = JLD2.load("tuned.jld2")["tuned"]
    p = tuned.p

    saved_θ, acceptPrMH = posterior_sampler(Array(yields[upper_p-p+1:end, 2:end]), Array(macros[upper_p-p+1:end, 2:end]), τₙ, ρ, iteration, tuned; medium_τ, μkQ_infty, σkQ_infty)
    saved_θ = saved_θ[burnin+1:end]
    iteration = length(saved_θ)

    saved_θ, Pr_stationary = erase_nonstationary_param(saved_θ)
    iteration = length(saved_θ)
    JLD2.save("posterior.jld2", "samples", saved_θ, "acceptPrMH", acceptPrMH, "Pr_stationary", Pr_stationary)

    ineff = ineff_factor(saved_θ)
    JLD2.save("ineff.jld2", "ineff", ineff)

    saved_TP = term_premium(TPτ_interest, τₙ, saved_θ, Array(yields[upper_p-p+1:end, 2:end]), Array(macros[upper_p-p+1:end, 2:end]))
    JLD2.save("TP.jld2", "TP", saved_TP)

    saved_prediction_50 = conditional_forecasts(gen_scene(0.5), scenario_TP, scenario_horizon, saved_θ, Array(yields[upper_p-p+1:sdate(yearmonth(scenario_start_date)...)-1, 2:end]), Array(macros[upper_p-p+1:sdate(yearmonth(scenario_start_date)...)-1, 2:end]), τₙ; mean_macros)

    saved_prediction_100 = conditional_forecasts(gen_scene(1), scenario_TP, scenario_horizon, saved_θ, Array(yields[upper_p-p+1:sdate(yearmonth(scenario_start_date)...)-1, 2:end]), Array(macros[upper_p-p+1:sdate(yearmonth(scenario_start_date)...)-1, 2:end]), τₙ; mean_macros)

    saved_prediction_150 = conditional_forecasts(gen_scene(1.5), scenario_TP, scenario_horizon, saved_θ, Array(yields[upper_p-p+1:sdate(yearmonth(scenario_start_date)...)-1, 2:end]), Array(macros[upper_p-p+1:sdate(yearmonth(scenario_start_date)...)-1, 2:end]), τₙ; mean_macros)
    JLD2.save("scenario.jld2", "forecasts_50", saved_prediction_50, "forecasts_100", saved_prediction_100, "forecasts_150", saved_prediction_150)

    return []
end

function inferences(; upper_p, τₙ, ρ, is_percent, idx_diff, macros, macros_growth, yields, scenario_start_date)

    sdate(yy, mm) = findall(x -> x == Date(yy, mm), macros[:, 1])[1]

    # from step 1
    opt = JLD2.load("tuned.jld2")["opt"]
    tuned = JLD2.load("tuned.jld2")["tuned"]
    p = tuned.p

    # from step 2
    saved_θ = JLD2.load("posterior.jld2")["samples"]
    acceptPrMH = JLD2.load("posterior.jld2")["acceptPrMH"]
    Pr_stationary = JLD2.load("posterior.jld2")["Pr_stationary"]
    saved_TP = JLD2.load("TP.jld2")["TP"]
    ineff = JLD2.load("ineff.jld2")["ineff"]
    @show (ineff[1], ineff[2], ineff[3] |> maximum, ineff[4] |> maximum, ineff[5] |> maximum, ineff[6] |> maximum)

    saved_Xθ = latentspace(saved_θ, Array(yields[upper_p-p+1:end, 2:end]), τₙ)
    fitted = fitted_YieldCurve(collect(1:τₙ[end]), saved_Xθ)
    decimal_yield = mean(fitted)[:yields] / 1200
    log_price = -collect(1:τₙ[end])' .* decimal_yield[p:end, :]
    xr = log_price[2:end, 1:end-1] - log_price[1:end-1, 2:end] .- decimal_yield[p:end-1, 1]
    realized_SR = mean(xr, dims=1) ./ std(xr, dims=1) |> x -> x[1, :]
    reduced_θ = reducedform(saved_θ, Array(yields[upper_p-p+1:end, 2:end]), Array(macros[upper_p-p+1:end, 2:end]), τₙ)
    mSR = [reduced_θ[:mpr][i] |> x -> sqrt.(diag(x * x')) for i in eachindex(reduced_θ)] |> mean

    raw_prediction = JLD2.load("scenario.jld2")["forecasts_50"]
    saved_prediction_50 = Vector{Forecast}(undef, length(raw_prediction))
    for i in eachindex(saved_prediction_50)
        predicted_factors = deepcopy(raw_prediction[i][:factors])
        for j in 1:dP-dQ
            if idx_diff[j] == 2
                predicted_factors[:, dQ+j] = [macros_growth[sdate(yearmonth(scenario_start_date)...)-1, j]; predicted_factors[:, dQ+j]] |> cumsum |> x -> x[2:end]
            elseif idx_diff[j] == 1 && is_percent[j]
                predicted_factors[:, dQ+j] = [raw_macros[sdate(yearmonth(scenario_start_date)...)-1, 1+j]; predicted_factors[:, dQ+j]] |> cumsum |> x -> x[2:end]
            elseif idx_diff[j] == 0 && !is_percent[j]
                predicted_factors[:, dQ+j] = 12 * [macros[sdate(yearmonth(scenario_start_date)...)-1, 1+j]; predicted_factors[:, dQ+j]] |> diff
            end
        end
        saved_prediction_50[i] = Forecast(yields=deepcopy(raw_prediction[i][:yields]), factors=deepcopy(predicted_factors), TP=deepcopy(raw_prediction[i][:TP]))
    end

    raw_prediction = JLD2.load("scenario.jld2")["forecasts_100"]
    saved_prediction_100 = Vector{Forecast}(undef, length(raw_prediction))
    for i in eachindex(saved_prediction_100)
        predicted_factors = deepcopy(raw_prediction[i][:factors])
        for j in 1:dP-dQ
            if idx_diff[j] == 2
                predicted_factors[:, dQ+j] = [macros_growth[sdate(yearmonth(scenario_start_date)...)-1, j]; predicted_factors[:, dQ+j]] |> cumsum |> x -> x[2:end]
            elseif idx_diff[j] == 1 && is_percent[j]
                predicted_factors[:, dQ+j] = [raw_macros[sdate(yearmonth(scenario_start_date)...)-1, 1+j]; predicted_factors[:, dQ+j]] |> cumsum |> x -> x[2:end]
            elseif idx_diff[j] == 0 && !is_percent[j]
                predicted_factors[:, dQ+j] = 12 * [macros[sdate(yearmonth(scenario_start_date)...)-1, 1+j]; predicted_factors[:, dQ+j]] |> diff
            end
        end
        saved_prediction_100[i] = Forecast(yields=deepcopy(raw_prediction[i][:yields]), factors=deepcopy(predicted_factors), TP=deepcopy(raw_prediction[i][:TP]))
    end

    raw_prediction = JLD2.load("scenario.jld2")["forecasts_150"]
    saved_prediction_150 = Vector{Forecast}(undef, length(raw_prediction))
    for i in eachindex(saved_prediction_150)
        predicted_factors = deepcopy(raw_prediction[i][:factors])
        for j in 1:dP-dQ
            if idx_diff[j] == 2
                predicted_factors[:, dQ+j] = [macros_growth[sdate(yearmonth(scenario_start_date)...)-1, j]; predicted_factors[:, dQ+j]] |> cumsum |> x -> x[2:end]
            elseif idx_diff[j] == 1 && is_percent[j]
                predicted_factors[:, dQ+j] = [raw_macros[sdate(yearmonth(scenario_start_date)...)-1, 1+j]; predicted_factors[:, dQ+j]] |> cumsum |> x -> x[2:end]
            elseif idx_diff[j] == 0 && !is_percent[j]
                predicted_factors[:, dQ+j] = 12 * [macros[sdate(yearmonth(scenario_start_date)...)-1, 1+j]; predicted_factors[:, dQ+j]] |> diff
            end
        end
        saved_prediction_150[i] = Forecast(yields=deepcopy(raw_prediction[i][:yields]), factors=deepcopy(predicted_factors), TP=deepcopy(raw_prediction[i][:TP]))
    end

    return (;
        opt=deepcopy(opt),
        tuned=deepcopy(tuned),
        saved_θ=deepcopy(saved_θ),
        acceptPrMH=deepcopy(acceptPrMH),
        Pr_stationary=deepcopy(Pr_stationary),
        saved_TP=deepcopy(saved_TP),
        ineff=deepcopy(ineff),
        saved_Xθ=deepcopy(saved_Xθ),
        fitted_yields=deepcopy(fitted),
        realized_SR=deepcopy(realized_SR),
        reduced_θ=deepcopy(reduced_θ),
        mSR=deepcopy(mSR),
        saved_prediction_50=deepcopy(saved_prediction_50),
        saved_prediction_100=deepcopy(saved_prediction_100),
        saved_prediction_150=deepcopy(saved_prediction_150))
end

function graphs(; τₙ, medium_τ, macros, macros_growth, raw_macros, yields, tuned, saved_θ, saved_TP, fitted, saved_prediction_50, is_percent, saved_prediction_100, saved_prediction_150)

    sdate(yy, mm) = findall(x -> x == Date(yy, mm), macros[:, 1])[1]
    set_default_plot_size(16cm, 8cm)

    ## decay parameter
    medium_τ_pr = length(medium_τ) |> x -> ones(x) / x
    κQ_support = [reverse(medium_τ) support(prior_κQ(medium_τ, medium_τ_pr))]
    Plots.histogram(saved_θ[:κQ], xticks=(κQ_support[:, 2], ["$(round(κQ_support[i,2],digits=4))\n(τ = $(round(Int,κQ_support[i,1])))" for i in axes(κQ_support, 1)]), bins=40, xlabel=L"\kappa_{Q} ( maturity \, \tau )", labels="") |> x -> Plots.pdf(x, "/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior_for_GDTSM/slide/decay.pdf")

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

    ## individual TP components
    ind_TP_ratio = mean(saved_TP)[:timevarying_TP] |> x -> var(x, dims=1) ./ var(mean(saved_TP)[:TP]) |> x -> x[1, :]
    ind_TP_order = sortperm(ind_TP_ratio, rev=true)
    n_top = 7
    ind_TP_names = ["PC1"; "PC2"; "PC3"; names(macros[1, 2:end])]

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
        Theme(major_label_font_size=10pt, minor_label_font_size=9pt, key_label_font_size=10pt, point_size=4pt), Guide.ylabel("percent per annum"), Guide.xlabel(""), Guide.xticks(ticks=DateTime("1986-07-01"):Month(72):DateTime("2023-06-01"), orientation=:horizontal),
        Guide.yticks(ticks=[-8; collect(-6:2:10)])
    ) |> PDF("/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior_for_GDTSM/slide/ind_TP10.pdf")

    ## EH components
    plot(
        layer(x=yields[sdate(1987, 1):end, 1], y=mean(fitted)[:yields][tuned.p+1:end, end] - mean(saved_TP)[:TP], Geom.line, color=[colorant"#4682B4"], Theme(line_width=2pt)),
        layer(xmin=rec_dates[:, 1], xmax=rec_dates[:, 2], Geom.band(; orientation=:vertical), color=[colorant"#DCDCDC"]),
        Theme(major_label_font_size=10pt, minor_label_font_size=9pt, key_label_font_size=10pt, point_size=4pt), Guide.ylabel("percent per annum"), Guide.xlabel(""), Guide.xticks(ticks=DateTime("1986-07-01"):Month(54):DateTime("2023-06-01")), Guide.yticks(ticks=[0; collect(1:7)])
    ) |> PDF("/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior_for_GDTSM/slide/EH10.pdf")

    ## Scenario analysis(yields)
    yield_res = mean(saved_prediction_100)[:yields]
    Plots.surface(τₙ, DateTime("2022-01-01"):Month(1):DateTime("2022-12-01"), yield_res, xlabel="maturity (months)", zlabel="yield", camera=(15, 30), legend=:none, linetype=:wireframe) |> x -> Plots.pdf(x, "/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior_for_GDTSM/slide/res_yield.pdf")

    p = []
    for i in [3, 7, 13, 18]
        ind_p = Plots.plot(Date(2022, 01):Month(1):Date(2022, 12), mean(saved_prediction_100)[:yields][:, i], fillrange=quantile(saved_prediction_100, 0.16)[:yields][:, i], labels="", title="yields(τ = $(τₙ[i]))", xticks=([Date(2022, 01):Month(3):Date(2022, 12);], ["Jan", "Apr", "Jul", "Oct"]), titlefontsize=10, c=colorant"#4682B4", alpha=0.6)
        Plots.plot!(ind_p, Date(2022, 01):Month(1):Date(2022, 12), mean(saved_prediction_100)[:yields][:, i], fillrange=quantile(saved_prediction_100, 0.84)[:yields][:, i], labels="", c=colorant"#4682B4", alpha=0.6)
        Plots.plot!(ind_p, Date(2022, 01):Month(1):Date(2022, 12), mean(saved_prediction_100)[:yields][:, i], fillrange=quantile(saved_prediction_100, 0.025)[:yields][:, i], labels="", c=colorant"#4682B4", alpha=0.6)
        Plots.plot!(ind_p, Date(2022, 01):Month(1):Date(2022, 12), mean(saved_prediction_100)[:yields][:, i], fillrange=quantile(saved_prediction_100, 0.975)[:yields][:, i], labels="", c=colorant"#4682B4", alpha=0.6)
        Plots.plot!(ind_p, Date(2022, 01):Month(1):Date(2022, 12), yields[sdate(2022, 1):sdate(2022, 12), 1+i], c=colorant"#DC143C", label="")
        push!(p, ind_p)
    end
    Plots.plot(p[1], p[2], p[3], p[4], layout=(2, 2), xlabel="") |> x -> Plots.pdf(x, "/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior_for_GDTSM/slide/res_yield2.pdf")

    ## Scenario analysis(EH)
    EH_res = mean(saved_prediction_100)[:yields][:, [7, 18]] - mean(saved_prediction_100)[:TP]
    EH_res_dist_24 = Matrix{Float64}(undef, length(saved_prediction_100), size(EH_res, 1))
    for i in axes(EH_res_dist_24, 1)
        EH_res_dist_24[i, :] = saved_prediction_100[:yields][i][:, 7] - saved_prediction_100[:TP][i][:, 1]
    end
    EH_res_dist_120 = Matrix{Float64}(undef, length(saved_prediction_100), size(EH_res, 1))
    for i in axes(EH_res_dist_120, 1)
        EH_res_dist_120[i, :] = saved_prediction_100[:yields][i][:, 18] - saved_prediction_100[:TP][i][:, 2]
    end

    p = []
    for i in 1:2
        if i == 1
            EH_res_dist = deepcopy(EH_res_dist_24)
            ind_name = "EH(τ = 24)"
        else
            EH_res_dist = deepcopy(EH_res_dist_120)
            ind_name = "EH(τ = 120)"
        end
        ind_p = Plots.plot(Date(2022, 01):Month(1):Date(2022, 12), EH_res[:, i], fillrange=[quantile(EH_res_dist[:, i], 0.16) for i in axes(EH_res_dist, 2)], labels="", title=ind_name, xticks=([Date(2022, 01):Month(3):Date(2022, 12);], ["Jan", "Apr", "Jul", "Oct"]), titlefontsize=10, c=colorant"#4682B4", alpha=0.6)
        Plots.plot!(ind_p, Date(2022, 01):Month(1):Date(2022, 12), EH_res[:, i], fillrange=[quantile(EH_res_dist[:, i], 0.84) for i in axes(EH_res_dist, 2)], labels="", c=colorant"#4682B4", alpha=0.6)
        Plots.plot!(ind_p, Date(2022, 01):Month(1):Date(2022, 12), EH_res[:, i], fillrange=[quantile(EH_res_dist[:, i], 0.025) for i in axes(EH_res_dist, 2)], labels="", c=colorant"#4682B4", alpha=0.6)
        Plots.plot!(ind_p, Date(2022, 01):Month(1):Date(2022, 12), EH_res[:, i], fillrange=[quantile(EH_res_dist[:, i], 0.975) for i in axes(EH_res_dist, 2)], labels="", c=colorant"#4682B4", alpha=0.6)
        push!(p, ind_p)
    end
    Plots.plot(p[1], p[2], layout=(1, 2), xlabel="", ylims=(-1, 5)) |> x -> Plots.pdf(x, "/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior_for_GDTSM/slide/res_EH.pdf")

    ## Scenario analysis(macros)
    p = []
    for i in ["RPI", "INDPRO", "CPIAUCSL", "S&P 500", "INVEST", "HOUST"]
        ind_macro = findall(x -> x == string(i), names(macros[1, 2:end]))[1]

        ind_p = Plots.plot(Date(2022, 01):Month(1):Date(2022, 12), mean(saved_prediction_100)[:factors][:, dimQ()+ind_macro], fillrange=quantile(saved_prediction_100, 0.025)[:factors][:, dimQ()+ind_macro], labels="", title=string(i), xticks=([Date(2022, 01):Month(3):Date(2022, 12);], ["Jan", "Apr", "Jul", "Oct"]), titlefontsize=10, c=colorant"#4682B4", alpha=0.6)
        Plots.plot!(ind_p, Date(2022, 01):Month(1):Date(2022, 12), mean(saved_prediction_100)[:factors][:, dimQ()+ind_macro], fillrange=quantile(saved_prediction_100, 0.975)[:factors][:, dimQ()+ind_macro], c=colorant"#4682B4", label="", fillalpha=0.6)
        Plots.plot!(ind_p, Date(2022, 01):Month(1):Date(2022, 12), mean(saved_prediction_100)[:factors][:, dimQ()+ind_macro], fillrange=quantile(saved_prediction_100, 0.16)[:factors][:, dimQ()+ind_macro], c=colorant"#4682B4", label="", fillalpha=0.6)
        Plots.plot!(ind_p, Date(2022, 01):Month(1):Date(2022, 12), mean(saved_prediction_100)[:factors][:, dimQ()+ind_macro], fillrange=quantile(saved_prediction_100, 0.84)[:factors][:, dimQ()+ind_macro], c=colorant"#4682B4", label="", fillalpha=0.6)
        if is_percent[ind_macro]
            Plots.plot!(ind_p, Date(2022, 01):Month(1):Date(2022, 12), raw_macros[sdate(2022, 1):sdate(2022, 12), 1+ind_macro], c=colorant"#DC143C", label="")
        else
            Plots.plot!(ind_p, Date(2022, 01):Month(1):Date(2022, 12), macros_growth[sdate(2022, 1):sdate(2022, 12), ind_macro], c=colorant"#DC143C", label="")
        end
        push!(p, ind_p)
    end
    Plots.plot(p[1], p[2], p[3], p[4], p[5], p[6], layout=(3, 2), xlabel="") |> x -> Plots.pdf(x, "/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior_for_GDTSM/slide/res_macro.pdf")
end

## Do

tuned, opt = tuning_hyperparameter(Array(yields[:, 2:end]), Array(macros[:, 2:end]), τₙ, ρ; upper_p, upper_q, μkQ_infty, σkQ_infty, medium_τ, init_ν0)
JLD2.save("tuned.jld2", "tuned", tuned, "opt", opt)

estimation(; upper_p, τₙ, medium_τ, iteration, burnin, ρ, macros, mean_macros, yields, scenario_TP, scenario_horizon, scenario_start_date, μkQ_infty, σkQ_infty)

results = inferences(; upper_p, τₙ, ρ, is_percent, idx_diff, macros, macros_growth, yields, scenario_start_date)

graphs(; τₙ, medium_τ, macros, macros_growth, raw_macros, yields, tuned=results.tuned, saved_θ=results.saved_θ, saved_TP=results.saved_TP, fitted=results.fitted_yields, saved_prediction_50=results.saved_prediction_50, is_percent, saved_prediction_100=results.saved_prediction_100, saved_prediction_150=results.saved_prediction_150)