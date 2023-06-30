using GDTSM
import StatsPlots: @df
using LinearAlgebra, Cairo, Fontconfig, Colors, XLSX

set_default_plot_size(16cm, 8cm)

## load unrestricted results
unres_lag = 6
unres_saved_θ = load("standard/posterior.jld2")["samples"]
unres_saved_TP = load("standard/TP.jld2")["TP"]
unres_ineff = load("standard/ineff.jld2")["ineff"]
unres_iteration = length(unres_saved_θ)
unres_reduced_θ = reducedform(unres_saved_θ[1:ceil(Int, maximum(unres_ineff)):unres_iteration], Array(yields[p_max-unres_lag+1:end, 2:end]), Array(macros[p_max-unres_lag+1:end, 2:end]), τₙ)
unres_mSR = [unres_reduced_θ[:mpr][i] |> x -> sqrt.(diag(x * x')) for i in eachindex(unres_reduced_θ)] |> mean

## Pareto frontier
mesh = [1 * ones(length(pf[1][1])) pf[1][2] pf[1][1]]
for i in 2:p_max
    global mesh = vcat(mesh, [i * ones(length(pf[1][1])) pf[i][2] pf[i][1]])
end
df = DataFrame(lag=mesh[:, 1], skew=mesh[:, 2], ML=mesh[:, 3])
rename!(df, Dict(:ML => "log marginal likelihood", :skew => "skewness"))
plot(
    df, x="skewness", y="log marginal likelihood", color=:lag, Geom.point,
    Guide.xticks(ticks=[collect(0:0.5:1); collect(1:0.5:4.5)]),
    Theme(major_label_font_size=12pt, minor_label_font_size=10pt, key_label_font_size=10pt, point_size=3pt, key_title_font_size=12pt), Scale.color_continuous(minvalue=0, maxvalue=9),
    Coord.cartesian(; ymin=-36900, ymax=-36200), Guide.yticks(ticks=-36900:100:-36200),
    #Guide.yticks(ticks=-36350:25:-36225), Coord.cartesian(; ymin=-36350, ymax=-36225)
) |> PDF("/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior for TS/slide/pf.pdf")

## TP components
rec_dates = DateTime.(["1990-07-01" "1991-03-01"
    "2001-03-01" "2001-11-01"
    "2007-12-01" "2009-06-01"
    "2020-02-01" "2020-04-01"])

plot(
    layer(x=yields[10:end, 1], y=mean(saved_TP)[:TP], Geom.line, color=[colorant"#4682B4"], Theme(line_width=2pt)),
    layer(x=yields[10:end, 1], y=quantile(saved_TP, 0.025)[:TP], Geom.line, color=[colorant"#A9A9A9"], Theme(line_width=0.5pt, line_style=[:dash])),
    layer(x=yields[10:end, 1], y=quantile(saved_TP, 0.975)[:TP], Geom.line, color=[colorant"#A9A9A9"], Theme(line_width=0.5pt, line_style=[:dash])),
    layer(x=yields[10:end, 1], y=mean(unres_saved_TP)[:TP], Geom.line, color=[colorant"#DC143C"], Theme(line_width=2pt)),
    layer(xmin=rec_dates[:, 1], xmax=rec_dates[:, 2], Geom.band(; orientation=:vertical), color=[colorant"#DCDCDC"]),
    Theme(major_label_font_size=10pt, minor_label_font_size=9pt, key_label_font_size=10pt, point_size=4pt), Guide.ylabel("percent per annum"), Guide.xlabel("time"), Guide.xticks(ticks=DateTime("1986-07-01"):Month(54):DateTime("2020-08-01")), Guide.yticks(ticks=[-2; 0; collect(2:2:5)])
) |> PDF("/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior for TS/slide/TP10.pdf")

## EH components
survey = XLSX.readdata("Dispersion_BILL10.xlsx", "D1", "B104:C217")[1:4:end, :] |> x -> convert(Matrix{Float64}, x)
plot(
    layer(x=yields[10:end, 1], y=mean(fitted)[:yields][tuned.p+1:end, end] - mean(saved_TP)[:TP], Geom.line, color=[colorant"#4682B4"], Theme(line_width=2pt)),
    layer(x=yields[10+12*5, 1]:Month(12):yields[end, 1], y=survey[:, 1], ymin=survey[:, 1], ymax=survey[:, 2], Geom.errorbar, color=[colorant"#A9A9A9"], Theme(line_width=0.75pt)),
    layer(xmin=rec_dates[:, 1], xmax=rec_dates[:, 2], Geom.band(; orientation=:vertical), color=[colorant"#DCDCDC"]),
    Theme(major_label_font_size=10pt, minor_label_font_size=9pt, key_label_font_size=10pt, point_size=4pt), Guide.ylabel("percent per annum"), Guide.xlabel("time"), Guide.xticks(ticks=DateTime("1986-07-01"):Month(54):DateTime("2020-08-01")), Guide.yticks(ticks=[0; collect(2:2:8)])
) |> PDF("/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior for TS/slide/EH10.pdf")

## maximum Sharpe ratio
plot(
    layer(x=yields[10:end, 1], y=mSR, Geom.line, color=[colorant"#4682B4"], Theme(line_width=2pt)),
    layer(x=yields[10:end, 1], y=unres_mSR, Geom.line, color=[colorant"#DC143C"], Theme(line_width=1pt)),
    layer(xmin=rec_dates[:, 1], xmax=rec_dates[:, 2], Geom.band(; orientation=:vertical), color=[colorant"#DCDCDC"]),
    Theme(major_label_font_size=10pt, minor_label_font_size=9pt, key_label_font_size=10pt, point_size=4pt), Guide.ylabel("percent per annum"), Guide.xlabel("time"), Guide.xticks(ticks=DateTime("1986-07-01"):Month(54):DateTime("2020-08-01")), Guide.yticks(ticks=collect(0:5))
) |> PDF("/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior for TS/slide/mSR.pdf")

## Scenario analysis(yields)
yield_res = max.(mean(saved_prediction)[:yields], 0)
yield_res[:, 1] .= 0
Plots.surface(τₙ, DateTime("2020-03-01"):Month(1):DateTime("2020-12-01"), yield_res, xlabel="maturity (months)", zlabel="yield", camera=(15, 30), legend=:none, linetype=:wireframe) |> x -> Plots.pdf(x, "/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior for TS/slide/res_yield.pdf")

## Scenario analysis(EH)
EH_res = max.(mean(saved_prediction)[:yields][:, [4, 12]] - mean(saved_prediction)[:TP], 0)
EH_res_dist_24 = Matrix{Float64}(undef, length(saved_prediction), size(EH_res, 1))
for i in axes(EH_res_dist_24, 1)
    EH_res_dist_24[i, :] = saved_prediction[:yields][i][:, 4] - saved_prediction[:TP][i][:, 1]
end
EH_res_dist_24 = max.(EH_res_dist_24, 0)
EH_res_dist_120 = Matrix{Float64}(undef, length(saved_prediction), size(EH_res, 1))
for i in axes(EH_res_dist_120, 1)
    EH_res_dist_120[i, :] = saved_prediction[:yields][i][:, end] - saved_prediction[:TP][i][:, 2]
end
EH_res_dist_120 = max.(EH_res_dist_120, 0)

scenario_dates = DateTime("2020-03-01"):Month(1):DateTime("2020-12-01")
plot(
    layer(x=scenario_dates, y=EH_res[:, 1], Geom.line, color=[colorant"#4682B4"], Theme(line_width=2pt)),
    layer(x=scenario_dates, y=[quantile(EH_res_dist_24[:, i], 0.25) for i in axes(EH_res_dist_24, 2)], Geom.line, color=[colorant"#4682B4"], Theme(line_width=0.5pt, line_style=[:dash])),
    layer(x=scenario_dates, y=[quantile(EH_res_dist_24[:, i], 0.75) for i in axes(EH_res_dist_24, 2)], Geom.line, color=[colorant"#4682B4"], Theme(line_width=0.5pt, line_style=[:dash])),
    layer(x=scenario_dates, y=EH_res[:, 2], Geom.line, color=[colorant"#DC143C"], Theme(line_width=2pt)),
    layer(x=scenario_dates, y=[quantile(EH_res_dist_120[:, i], 0.25) for i in axes(EH_res_dist_120, 2)], Geom.line, color=[colorant"#DC143C"], Theme(line_width=0.5pt, line_style=[:dash])),
    layer(x=scenario_dates, y=[quantile(EH_res_dist_120[:, i], 0.75) for i in axes(EH_res_dist_120, 2)], Geom.line, color=[colorant"#DC143C"], Theme(line_width=0.5pt, line_style=[:dash])),
    Theme(major_label_font_size=10pt, minor_label_font_size=9pt, key_label_font_size=10pt, point_size=4pt), Guide.ylabel("percent per annum"), Guide.xlabel("time"), Guide.yticks(ticks=collect(0:0.5:2)), Guide.xticks(ticks=DateTime("2020-02-01"):Month(2):DateTime("2021-01-01")),
    Coord.cartesian(; xmin=DateTime("2020-02-01"), xmax=DateTime("2021-01-01"))
) |> PDF("/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior for TS/slide/res_EH.pdf")

## Scenario analysis(macros)
macro_res = mean(saved_prediction)[:factors][:, 4:end] |> x -> DataFrame([collect(DateTime("2020-03-01"):Month(1):DateTime("2020-12-01")) x], ["dates"; names(macros[:, 2:end])])
rename!(macro_res, Dict("S&P 500" => "SP500"))
@df macro_res Plots.plot(:dates, [:RPI :INDPRO :CPIAUCSL :SP500 :INVEST :HOUST], xlabel="time", ylabel="M/M (%)", tickfont=(10), legendfontsize=10, linewidth=2, label=["RPI" "INDPRO" "CPIAUCSL" "SP500" "INVEST" "HOUST"]) |> x -> Plots.pdf(x, "/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior for TS/slide/res_macro.pdf")