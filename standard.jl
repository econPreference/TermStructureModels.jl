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
init_ν0 = 40

# estimation
iteration = 15_000
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

    saved_TP = term_premium(TPτ_interest, τₙ, saved_θ, Array(yields[upper_p-p+1:end, 2:end]), Array(macros[upper_p-p+1:end, 2:end]))
    JLD2.save("standard/TP.jld2", "TP", saved_TP)

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
    @show (ineff[1], ineff[2], ineff[3] |> maximum, ineff[4] |> maximum, ineff[5] |> maximum, ineff[6] |> maximum)

    saved_Xθ = latentspace(saved_θ, Array(yields[upper_p-p+1:end, 2:end]), τₙ)
    fitted = fitted_YieldCurve(collect(1:τₙ[end]), saved_Xθ)
    decimal_yield = mean(fitted)[:yields] / 1200
    log_price = -collect(1:τₙ[end])' .* decimal_yield[p:end, :]
    xr = log_price[2:end, 1:end-1] - log_price[1:end-1, 2:end] .- decimal_yield[p:end-1, 1]
    realized_SR = mean(xr, dims=1) ./ std(xr, dims=1) |> x -> x[1, :]
    reduced_θ = reducedform(saved_θ, Array(yields[upper_p-p+1:end, 2:end]), Array(macros[upper_p-p+1:end, 2:end]), τₙ)
    mSR = [reduced_θ[:mpr][i] |> x -> sqrt.(diag(x * x')) for i in eachindex(reduced_θ)] |> mean

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
        mSR=deepcopy(mSR))
end

function graphs(; medium_τ, macros, yields, tuned, saved_θ, saved_TP, fitted)

    sdate(yy, mm) = findall(x -> x == Date(yy, mm), macros[:, 1])[1]
    TP_nolag = JLD2.load("nolag/TP.jld2")["TP"]
    TP_nomacro = JLD2.load("nomacro/TP.jld2")["TP"]

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

    plot(
        layer(x=yields[sdate(1987, 1):end, 1], y=mean(saved_TP)[:TP], Geom.line, color=[colorant"#000000"], Theme(line_width=2pt)),
        layer(x=yields[sdate(1987, 1):end, 1], y=mean(TP_nomacro)[:TP], Geom.line, color=[colorant"#4682B4"], Theme(line_width=2pt, line_style=[:dash])),
        layer(x=yields[sdate(1987, 1):end, 1], y=mean(TP_nolag)[:TP], Geom.line, color=[colorant"#DC143C"], Theme(line_width=2pt, line_style=[:dot])),
        layer(xmin=rec_dates[:, 1], xmax=rec_dates[:, 2], Geom.band(; orientation=:vertical), color=[colorant"#DCDCDC"]),
        Theme(major_label_font_size=10pt, minor_label_font_size=9pt, key_label_font_size=10pt, point_size=4pt), Guide.ylabel("percent per annum"), Guide.xlabel(""), Guide.xticks(ticks=DateTime("1986-07-01"):Month(54):DateTime("2023-06-01")), Guide.yticks(ticks=-2:2:4)
    ) |> PDF("/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior_for_GDTSM/slide/TPs.pdf")

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

end

## Do

tuned, opt = tuning_hyperparameter(Array(yields[:, 2:end]), Array(macros[:, 2:end]), τₙ, ρ; upper_p, upper_q, μkQ_infty, σkQ_infty, medium_τ, init_ν0)
JLD2.save("standard/tuned.jld2", "tuned", tuned, "opt", opt)

estimation(; upper_p, τₙ, medium_τ, iteration, burnin, ρ, macros, yields, μkQ_infty, σkQ_infty)

results = inferences(; upper_p, τₙ, macros, yields)

graphs(; medium_τ, macros, yields, tuned=results.tuned, saved_θ=results.saved_θ, saved_TP=results.saved_TP, fitted=results.fitted_yields)