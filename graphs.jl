using GDTSM
import StatsPlots: @df
using LinearAlgebra, Cairo, Fontconfig, Colors, XLSX, LaTeXStrings, Distributions

set_default_plot_size(16cm, 8cm)

## load unrestricted results
unres_saved_TP = load("standard/TP.jld2")["TP"]
unres_tuned = load("standard/tuned.jld2")["tuned"]

## hyperparameter plots
q_set = Matrix{Float64}(undef, length(tuned_set) + 1, 8)
q_set[1, :] = vec(unres_tuned.q)
for i in eachindex(tuned_set)
    q_set[i+1, :] = vec(tuned_set[i].q)
end
ν0_set = [tuned_set[i].ν0 for i in eachindex(tuned_set)]
ν0_set = [unres_tuned.ν0; ν0_set]
Plots.plot(1:9, q_set[:, 1], ylabel=L"${q}_{11}$ and $0.001 \times \nu_0$", c=colorant"#4682B4", labels="", linewidth=2, ylims=(0, 0.15))
Plots.plot!(1:9, ν0_set ./ 1000, c=colorant"#008000", labels="", linewidth=2, ls=:dot)
Plots.plot!(Plots.twinx(), q_set[:, 4], ylabel=L"${q}_{41}$", c=colorant"#DC143C", labels="", linewidth=2, ls=:dash)
Plots.plot!(xticks=([1:9;], ["unrestricted"; string.(q41_list)]))
Plots.xlabel!(L"restriction: ${q}_{41} {\leq}$ above values") |> x -> Plots.pdf(x, "/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior for TS/slide/hyperparameters.pdf")

## decay parameter
κQ_support = [reverse(medium_τ) support(prior_κQ(medium_τ))]
Plots.histogram(saved_θ[:κQ], xticks=(κQ_support[:, 2], string.(κQ_support[:, 1])), bins=40, xlabel="maturity") |> x -> Plots.pdf(x, "/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior for TS/slide/decay.pdf")

## TP components
rec_dates = DateTime.(["1990-07-01" "1991-03-01"
    "2001-03-01" "2001-11-01"
    "2007-12-01" "2009-06-01"
    "2020-02-01" "2020-04-01"])

plot(
    layer(x=yields[16:end, 1], y=mean(saved_TP)[:TP], Geom.line, color=[colorant"#4682B4"], Theme(line_width=2pt)),
    layer(x=yields[16:end, 1], y=quantile(saved_TP, 0.025)[:TP], Geom.line, color=[colorant"#A9A9A9"], Theme(line_width=0.5pt, line_style=[:dash])),
    layer(x=yields[16:end, 1], y=quantile(saved_TP, 0.975)[:TP], Geom.line, color=[colorant"#A9A9A9"], Theme(line_width=0.5pt, line_style=[:dash])),
    layer(x=yields[16:end, 1], y=mean(unres_saved_TP)[:TP], Geom.line, color=[colorant"#DC143C"], Theme(line_width=2pt)),
    layer(xmin=rec_dates[:, 1], xmax=rec_dates[:, 2], Geom.band(; orientation=:vertical), color=[colorant"#DCDCDC"]),
    Theme(major_label_font_size=10pt, minor_label_font_size=9pt, key_label_font_size=10pt, point_size=4pt), Guide.ylabel("percent per annum"), Guide.xlabel("time"), Guide.xticks(ticks=DateTime("1986-07-01"):Month(54):DateTime("2023-06-01")), Guide.yticks(ticks=-4:2:4)
) |> PDF("/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior for TS/slide/TP10.pdf")

plot(
    layer(x=yields[16:end, 1], y=quantile(unres_saved_TP, 0.025)[:TP], Geom.line, color=[colorant"#A9A9A9"], Theme(line_width=0.5pt, line_style=[:dash])),
    layer(x=yields[16:end, 1], y=quantile(unres_saved_TP, 0.975)[:TP], Geom.line, color=[colorant"#A9A9A9"], Theme(line_width=0.5pt, line_style=[:dash])),
    layer(x=yields[16:end, 1], y=mean(unres_saved_TP)[:TP], Geom.line, color=[colorant"#DC143C"], Theme(line_width=2pt)),
    layer(xmin=rec_dates[:, 1], xmax=rec_dates[:, 2], Geom.band(; orientation=:vertical), color=[colorant"#DCDCDC"]),
    Theme(major_label_font_size=10pt, minor_label_font_size=9pt, key_label_font_size=10pt, point_size=4pt), Guide.ylabel("percent per annum"), Guide.xlabel("time"), Guide.xticks(ticks=DateTime("1986-07-01"):Month(54):DateTime("2023-06-01")), Guide.yticks(ticks=-4:2:6)
) |> PDF("/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior for TS/slide/unres_TP10.pdf")

## individual TP components
ind_TP_ratio = mean(saved_TP)[:timevarying_TP] |> x -> var(x, dims=1) ./ var(mean(saved_TP)[:TP]) |> x -> x[1, :]
ind_TP_order = sortperm(ind_TP_ratio, rev=true)
n_top = 7
ind_TP_names = ["PC1"; "PC2"; "PC3"; names(macros[1, 2:end])]

ind_TP = mean(saved_TP)[:timevarying_TP][:, ind_TP_order[1]]
mesh = [yields[16:end, 1] fill(ind_TP_names[ind_TP_order[1]], size(ind_TP)) ind_TP]
for i in 2:n_top
    local ind_TP = mean(saved_TP)[:timevarying_TP][:, ind_TP_order[i]]
    global mesh = [mesh; [yields[16:end, 1] fill(ind_TP_names[ind_TP_order[i]], size(ind_TP)) ind_TP]]
end
df = DataFrame(dates=Date.(string.(mesh[:, 1]), DateFormat("yyyy-mm-dd")), macros=string.(mesh[:, 2]), TP=Float64.(mesh[:, 3]))

plot(df,
    layer(x=:dates, y=:TP, Geom.line, color=:macros, Theme(line_width=1pt)),
    layer(xmin=rec_dates[:, 1], xmax=rec_dates[:, 2], Geom.band(; orientation=:vertical), color=[colorant"#DCDCDC"]),
    Theme(major_label_font_size=10pt, minor_label_font_size=9pt, key_label_font_size=10pt, point_size=4pt), Guide.ylabel("percent per annum"), Guide.xlabel(""), Guide.xticks(ticks=DateTime("1986-07-01"):Month(72):DateTime("2023-06-01"), orientation=:horizontal),
    Guide.yticks(ticks=[-8; collect(-6:2:10)])
) |> PDF("/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior for TS/slide/ind_TP10.pdf")

## EH components
survey = XLSX.readdata("Dispersion_BILL10.xlsx", "D1", "B104:C229")[1:4:end, :] |> x -> convert(Matrix{Float64}, x)
plot(
    layer(x=yields[16:end, 1], y=mean(fitted)[:yields][tuned.p+1:end, end] - mean(saved_TP)[:TP], Geom.line, color=[colorant"#4682B4"], Theme(line_width=2pt)),
    layer(x=yields[16+12*5-1, 1]:Month(12):yields[end, 1], y=survey[:, 1], ymin=survey[:, 1], ymax=survey[:, 2], Geom.errorbar, color=[colorant"#A9A9A9"], Theme(line_width=0.75pt)),
    layer(xmin=rec_dates[:, 1], xmax=rec_dates[:, 2], Geom.band(; orientation=:vertical), color=[colorant"#DCDCDC"]),
    Theme(major_label_font_size=10pt, minor_label_font_size=9pt, key_label_font_size=10pt, point_size=4pt), Guide.ylabel("percent per annum"), Guide.xlabel("time"), Guide.xticks(ticks=DateTime("1986-07-01"):Month(54):DateTime("2023-06-01")), Guide.yticks(ticks=[0; collect(1:7)])
) |> PDF("/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior for TS/slide/EH10.pdf")

## Scenario analysis(yields)
yield_res = mean(saved_prediction)[:yields]
Plots.surface(τₙ, DateTime("2020-03-01"):Month(1):DateTime("2020-12-01"), yield_res, xlabel="maturity (months)", zlabel="yield", camera=(15, 30), legend=:none, linetype=:wireframe) |> x -> Plots.pdf(x, "/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior for TS/slide/res_yield.pdf")

## Scenario analysis(EH)
EH_res = mean(saved_prediction)[:yields][:, [7, 18]] - mean(saved_prediction)[:TP]
EH_res_dist_24 = Matrix{Float64}(undef, length(saved_prediction), size(EH_res, 1))
for i in axes(EH_res_dist_24, 1)
    EH_res_dist_24[i, :] = saved_prediction[:yields][i][:, 7] - saved_prediction[:TP][i][:, 1]
end
EH_res_dist_120 = Matrix{Float64}(undef, length(saved_prediction), size(EH_res, 1))
for i in axes(EH_res_dist_120, 1)
    EH_res_dist_120[i, :] = saved_prediction[:yields][i][:, end] - saved_prediction[:TP][i][:, 2]
end

scenario_dates = DateTime("2020-03-01"):Month(1):DateTime("2020-12-01")
plot(
    layer(x=scenario_dates, y=EH_res[:, 1], Geom.line, color=[colorant"#4682B4"], Theme(line_width=2pt)),
    layer(x=scenario_dates, y=[quantile(EH_res_dist_24[:, i], 0.25) for i in axes(EH_res_dist_24, 2)], Geom.line, color=[colorant"#4682B4"], Theme(line_width=0.5pt, line_style=[:dash])),
    layer(x=scenario_dates, y=[quantile(EH_res_dist_24[:, i], 0.75) for i in axes(EH_res_dist_24, 2)], Geom.line, color=[colorant"#4682B4"], Theme(line_width=0.5pt, line_style=[:dash])),
    layer(x=scenario_dates, y=EH_res[:, 2], Geom.line, color=[colorant"#DC143C"], Theme(line_width=2pt)),
    layer(x=scenario_dates, y=[quantile(EH_res_dist_120[:, i], 0.25) for i in axes(EH_res_dist_120, 2)], Geom.line, color=[colorant"#DC143C"], Theme(line_width=0.5pt, line_style=[:dash])),
    layer(x=scenario_dates, y=[quantile(EH_res_dist_120[:, i], 0.75) for i in axes(EH_res_dist_120, 2)], Geom.line, color=[colorant"#DC143C"], Theme(line_width=0.5pt, line_style=[:dash])),
    Theme(major_label_font_size=10pt, minor_label_font_size=9pt, key_label_font_size=10pt, point_size=4pt), Guide.ylabel("percent per annum"), Guide.xlabel("time"), Guide.yticks(ticks=collect(0:0.5:3.5)), Guide.xticks(ticks=DateTime("2020-02-01"):Month(2):DateTime("2021-01-01")),
    Coord.cartesian(; xmin=DateTime("2020-02-01"), xmax=DateTime("2021-01-01"))
) |> PDF("/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior for TS/slide/res_EH.pdf")

## Scenario analysis(macros)
macro_res = mean(saved_prediction)[:factors][:, dimQ()+1:end] |> x -> DataFrame([collect(DateTime("2020-03-01"):Month(1):DateTime("2020-12-01")) x], ["dates"; names(macros[:, 2:end])])
rename!(macro_res, Dict("S&P 500" => "SP500"))
@df macro_res Plots.plot(:dates, [:RPI :INDPRO :CPIAUCSL :SP500 :INVEST :HOUST], xlabel="time", ylabel="M/M (%) or ΔM/M (%p)", tickfont=(10), legendfontsize=10, linewidth=2, label=["RPI" "INDPRO" "CPIAUCSL" "S&P 500" "INVEST" "HOUST"]) |> x -> Plots.pdf(x, "/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior for TS/slide/res_macro.pdf")
