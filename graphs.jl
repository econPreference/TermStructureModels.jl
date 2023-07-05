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
begin #overall
    Plots.scatter3d()
    colors = [colorant"#FFA07A", colorant"#FF0000", colorant"#800000", colorant"#7CFC00", colorant"#006400", colorant"#E6E6FA", colorant"#87CEFA", colorant"#4682B4", colorant"#0000FF"]
    for i in 1:p_max
        Plots.scatter3d!(pf[i][:, 2], pf[i][:, 3], pf[i][:, 1], label="lag $i", camera=(45, 30), legend=:right, color=colors[i])
    end
    Plots.xlabel!("skewness")
    Plots.ylabel!("mSR_const")
    Plots.zlabel!("log marginal likelihood")
end
Plots.scatter3d!() |> x -> Plots.pdf(x, "/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior for TS/slide/pf.pdf")

begin #zoom in
    Plots.scatter3d()
    colors = [colorant"#FFA07A", colorant"#FF0000", colorant"#800000", colorant"#7CFC00", colorant"#006400", colorant"#E6E6FA", colorant"#87CEFA", colorant"#4682B4", colorant"#0000FF"]
    for i in 1:p_max
        Plots.scatter3d!(pf[i][:, 2], pf[i][:, 3], pf[i][:, 1], label="lag $i", camera=(45, 30), legend=:right, color=colors[i])
    end
    Plots.xlabel!("skewness")
    Plots.ylabel!("mSR_const")
    Plots.zlabel!("log marginal likelihood")
    Plots.zlims!(-36350, -36240)
end
Plots.scatter3d!() |> x -> Plots.pdf(x, "/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior for TS/slide/pf_zoom.pdf")

begin #x-axis
    Plots.scatter3d()
    colors = [colorant"#FFA07A", colorant"#FF0000", colorant"#800000", colorant"#7CFC00", colorant"#006400", colorant"#E6E6FA", colorant"#87CEFA", colorant"#4682B4", colorant"#0000FF"]
    for i in 1:p_max
        Plots.scatter3d!(pf[i][:, 2], pf[i][:, 3], pf[i][:, 1], label="lag $i", camera=(0, 0), legend=:right, color=colors[i])
    end
    Plots.xlabel!("skewness")
    Plots.ylabel!("mSR_const")
    Plots.zlabel!("log marginal likelihood")
    Plots.zlims!(-36350, -36240)
    Plots.yaxis!(false)
end
Plots.scatter3d!() |> x -> Plots.pdf(x, "/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior for TS/slide/pf_x.pdf")

begin # y-axis
    Plots.scatter3d()
    colors = [colorant"#FFA07A", colorant"#FF0000", colorant"#800000", colorant"#7CFC00", colorant"#006400", colorant"#E6E6FA", colorant"#87CEFA", colorant"#4682B4", colorant"#0000FF"]
    for i in 1:p_max
        Plots.scatter3d!(pf[i][:, 2], pf[i][:, 3], pf[i][:, 1], label="lag $i", camera=(90, 0), legend=:right, color=colors[i])
    end
    Plots.xlabel!("skewness")
    Plots.ylabel!("mSR_const")
    Plots.yticks!(0.05:0.05:0.25)
    Plots.zlabel!("log marginal likelihood")
    Plots.zlims!(-36350, -36240)
    Plots.xaxis!(false)
end
Plots.scatter3d!() |> x -> Plots.pdf(x, "/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior for TS/slide/pf_y.pdf")

begin #z-axis
    Plots.scatter3d()
    colors = [colorant"#FFA07A", colorant"#FF0000", colorant"#800000", colorant"#7CFC00", colorant"#006400", colorant"#E6E6FA", colorant"#87CEFA", colorant"#4682B4", colorant"#0000FF"]
    for i in 1:p_max
        Plots.scatter3d!(pf[i][:, 2], pf[i][:, 3], pf[i][:, 1], label="lag $i", camera=(0, 90), legend=:right, color=colors[i])
    end
    Plots.xlabel!("skewness")
    Plots.ylabel!("mSR_const", yrotation=90)
    Plots.yticks!(0.05:0.05:0.25)
    Plots.ylims!(0.05, 0.25)
    Plots.zlabel!("log marginal likelihood")
    Plots.zaxis!(false)
end
Plots.scatter3d!() |> x -> Plots.pdf(x, "/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior for TS/slide/pf_z.pdf")

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

plot(
    layer(x=yields[10:end, 1], y=quantile(unres_saved_TP, 0.025)[:TP], Geom.line, color=[colorant"#DC143C"], Theme(line_width=0.5pt, line_style=[:dash])),
    layer(x=yields[10:end, 1], y=quantile(unres_saved_TP, 0.975)[:TP], Geom.line, color=[colorant"#DC143C"], Theme(line_width=0.5pt, line_style=[:dash])),
    layer(x=yields[10:end, 1], y=mean(unres_saved_TP)[:TP], Geom.line, color=[colorant"#DC143C"], Theme(line_width=2pt)),
    layer(xmin=rec_dates[:, 1], xmax=rec_dates[:, 2], Geom.band(; orientation=:vertical), color=[colorant"#DCDCDC"]),
    Theme(major_label_font_size=10pt, minor_label_font_size=9pt, key_label_font_size=10pt, point_size=4pt), Guide.ylabel("percent per annum"), Guide.xlabel("time"), Guide.xticks(ticks=DateTime("1986-07-01"):Month(54):DateTime("2020-08-01")), Guide.yticks(ticks=[-2; 0; collect(2:2:5)])
) |> PDF("/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior for TS/slide/unres_TP10.pdf")

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

plot(
    layer(x=yields[10:end, 1], y=unres_mSR, Geom.line, color=[colorant"#DC143C"], Theme(line_width=1pt)),
    layer(xmin=rec_dates[:, 1], xmax=rec_dates[:, 2], Geom.band(; orientation=:vertical), color=[colorant"#DCDCDC"]),
    Theme(major_label_font_size=10pt, minor_label_font_size=9pt, key_label_font_size=10pt, point_size=4pt), Guide.ylabel("percent per annum"), Guide.xlabel("time"), Guide.xticks(ticks=DateTime("1986-07-01"):Month(54):DateTime("2020-08-01")), Guide.yticks(ticks=collect(0:5))
) |> PDF("/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior for TS/slide/unres_mSR.pdf")

## Scenario analysis(yields)
yield_res = mean(saved_prediction)[:yields]
yield_res[:, 1] .= 0
Plots.surface(τₙ, DateTime("2020-03-01"):Month(1):DateTime("2020-12-01"), yield_res, xlabel="maturity (months)", zlabel="yield", camera=(15, 30), legend=:none, linetype=:wireframe) |> x -> Plots.pdf(x, "/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior for TS/slide/res_yield.pdf")

## Scenario analysis(EH)
EH_res = mean(saved_prediction)[:yields][:, [4, 12]] - mean(saved_prediction)[:TP]
EH_res_dist_24 = Matrix{Float64}(undef, length(saved_prediction), size(EH_res, 1))
for i in axes(EH_res_dist_24, 1)
    EH_res_dist_24[i, :] = saved_prediction[:yields][i][:, 4] - saved_prediction[:TP][i][:, 1]
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
    Theme(major_label_font_size=10pt, minor_label_font_size=9pt, key_label_font_size=10pt, point_size=4pt), Guide.ylabel("percent per annum"), Guide.xlabel("time"), Guide.yticks(ticks=collect(0:0.5:2)), Guide.xticks(ticks=DateTime("2020-02-01"):Month(2):DateTime("2021-01-01")),
    Coord.cartesian(; xmin=DateTime("2020-02-01"), xmax=DateTime("2021-01-01"))
) |> PDF("/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior for TS/slide/res_EH.pdf")

## Scenario analysis(macros)
macro_res = mean(saved_prediction)[:factors][:, 4:end] |> x -> DataFrame([collect(DateTime("2020-03-01"):Month(1):DateTime("2020-12-01")) x], ["dates"; names(macros[:, 2:end])])
rename!(macro_res, Dict("S&P 500" => "SP500"))
@df macro_res Plots.plot(:dates, [:RPI :INDPRO :CPIAUCSL :SP500 :INVEST :HOUST], xlabel="time", ylabel="M/M (%)", tickfont=(10), legendfontsize=10, linewidth=2, label=["RPI" "INDPRO" "CPIAUCSL" "SP500" "INVEST" "HOUST"]) |> x -> Plots.pdf(x, "/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior for TS/slide/res_macro.pdf")

## Comparing mSR
begin # MOVE data
    raw_MOVE = CSV.File("MOVE.csv", types=[Date; Float64]) |> DataFrame |> x -> x[9:end, :] |> reverse
    idx = month.(raw_MOVE[:, 1]) |> x -> (x .!= [x[2:end]; x[end]])
    MOVE = raw_MOVE[idx, :]
    MOVE = MOVE[1:findall(x -> x == yearmonth(date_end), yearmonth.(MOVE[:, 1]))[1], :]
end

PCs = PCA(Array(yields[p_max-lag+1:end, 2:end]), lag)[1]
ΩPP = [AR_res_var(PCs[lag+1:end, i], lag)[1] for i in 1:dimQ()] |> diagm

mSR_prior, mSR_const = maximum_SR(Array(yields[p_max-lag+1:end, 2:end]), Array(macros[p_max-lag+1:end, 2:end]), tuned, τₙ, ρ; κQ=mean(prior_κQ(medium_τ)), kQ_infty=μkQ_infty, ΩPP)
mSR_simul, mSR_const_simul = maximum_SR_simul(Array(yields[p_max-lag+1:end, 2:end]), Array(macros[p_max-lag+1:end, 2:end]), tuned, τₙ, ρ; κQ=mean(prior_κQ(medium_τ)), kQ_infty=μkQ_infty, ΩPP)
aux_idx = length(mSR_prior)-length(MOVE[:, 1])+1:length(mSR_prior)
std_MOVE = MOVE[:, 2] |> x -> (x .- mean(x)) ./ std(x) |> x -> std(mSR_prior[aux_idx]) * x |> x -> x .+ mean(mSR_prior[aux_idx])
plot(
    layer(x=yields[10:end, 1][aux_idx], y=std_MOVE, Geom.point, color=[RGBA(0, 128 / 255, 0, 0)], Theme(point_size=1.5pt)),
    layer(x=yields[10:end, 1], y=mSR_prior, Geom.line, color=[colorant"#DC143C"], Theme(line_width=1pt, line_style=[:dash])),
    layer(x=yields[10:end, 1], y=mean(mSR_simul, dims=1)[1, :], Geom.line, color=[colorant"#4682B4"], Theme(line_width=2pt)),
    layer(xmin=rec_dates[:, 1], xmax=rec_dates[:, 2], Geom.band(; orientation=:vertical), color=[colorant"#DCDCDC"]),
    Theme(major_label_font_size=10pt, minor_label_font_size=9pt, key_label_font_size=10pt, point_size=4pt), Guide.xlabel("time"), Guide.xticks(ticks=DateTime("1986-07-01"):Month(54):DateTime("2020-08-01")), Guide.ylabel(""),
    Guide.xlabel("Constant part: simul = $(round(mean(mSR_const_simul),digits=4)), approx = $(round(mSR_const,digits=4))")
) |> PDF("/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior for TS/slide/mSR_prior.pdf")

plot(
    layer(x=yields[10:end, 1], y=mSR_prior, Geom.line, color=[colorant"#DC143C"], Theme(line_width=1pt, line_style=[:dash])),
    layer(x=yields[10:end, 1], y=mSR, Geom.line, color=[colorant"#4682B4"], Theme(line_width=2pt)),
    layer(xmin=rec_dates[:, 1], xmax=rec_dates[:, 2], Geom.band(; orientation=:vertical), color=[colorant"#DCDCDC"]),
    Theme(major_label_font_size=10pt, minor_label_font_size=9pt, key_label_font_size=10pt, point_size=4pt), Guide.xlabel("time"), Guide.ylabel("maximum SR"), Guide.xticks(ticks=DateTime("1986-07-01"):Month(54):DateTime("2020-08-01"))
) |> PDF("/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior for TS/slide/mSR_post.pdf")