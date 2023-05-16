using GDTSM
import StatsPlots: @df
using LinearAlgebra, Cairo, Fontconfig, Colors

set_default_plot_size(16cm, 8cm)
## Graphs
mesh = [1 * ones(length(pf[1][1])) pf[1][2] pf[1][1]]
for i in 2:p_max
    mesh = vcat(mesh, [i * ones(length(pf[1][1])) pf[i][2] pf[i][1]])
end
df = DataFrame(lag=mesh[:, 1], mSR=mesh[:, 2], ML=mesh[:, 3])
rename!(df, Dict(:ML => "log marginal likelihood", :mSR => "quantile(maximum SR, 0.95)"))
plot(df, x="quantile(maximum SR, 0.95)", y="log marginal likelihood", color=:lag, Geom.point, Guide.yticks(ticks=24250:1250:30500), Guide.xticks(ticks=[collect(0:4:20); collect(20:10:70)]), Theme(major_label_font_size=12pt, minor_label_font_size=10pt, key_label_font_size=10pt, point_size=3pt, key_title_font_size=12pt), Scale.color_continuous(minvalue=0, maxvalue=12)) |> PDF("/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior for TS/slide/pf.pdf")

rec_dates = DateTime.(["1990-07-01" "1991-03-01"
    "2001-03-01" "2001-11-01"
    "2007-12-01" "2009-06-01"
    "2020-02-01" "2020-04-01"])
plot(
    layer(x=yields[13:end, 1], y=mean(fitted)[:yields][tuned.p+1:end, 1], Geom.line, color=[colorant"blue"]),
    layer(x=yields[13:end, 1], y=mean(fitted)[:yields][tuned.p+1:end, end] - mean(saved_TP)[:TP], Geom.line, linestyle=[:dash], color=[colorant"red"]),
    layer(xmin=rec_dates[:, 1], xmax=rec_dates[:, 2], Geom.band(; orientation=:vertical), color=[colorant"grey"]),
    Guide.manual_color_key("", ["one month yield", "expected one month yield over 10 years", "NBER recessions"], ["blue", "red", "grey"]), Theme(line_width=2pt, key_position=:top, major_label_font_size=10pt, minor_label_font_size=9pt, key_label_font_size=10pt, point_size=4pt), Guide.ylabel("percent per annum"), Guide.xlabel("time"), Guide.xticks(ticks=DateTime("1986-07-01"):Month(54):DateTime("2020-08-01")), Guide.yticks(ticks=[-1; 0; collect(2:3:10)])
) |> PDF("/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior for TS/slide/vanilla_EH10.pdf")

plot(
    layer(x=yields[13:end, 1], y=mSR, Geom.line, color=[colorant"blue"]),
    layer(x=[], y=[]),
    layer(xmin=rec_dates[:, 1], xmax=rec_dates[:, 2], Geom.band(; orientation=:vertical), color=[colorant"grey"]),
    Guide.manual_color_key("", ["maximum SR", " ", "NBER recessions"], ["blue", "white", "grey"]), Theme(line_width=2pt, key_position=:top, major_label_font_size=10pt, minor_label_font_size=9pt, key_label_font_size=10pt, point_size=4pt), Guide.ylabel(""), Guide.xlabel("time"), Guide.xticks(ticks=DateTime("1986-07-01"):Month(60):DateTime("2020-08-01")), Guide.yticks(ticks=0:0.2:maximum(mSR)+0.2)
) |> PDF("/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior for TS/slide/vanilla_mSR.pdf")

# Plots.histogram(mSR, bins=range(0, 3, length=31), normalize=:pdf, labels="maximum SR", alpha=0.9)
# Plots.histogram!(rand(realized_SR, length(mSR)), bins=range(0, 3, length=31), normalize=:pdf, labels="realized SR", xlabel="Sharpe ratio", ylabel="density", alpha=0.9) |> x -> Plots.pdf(x, "/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior for TS/slide/vanilla_mSR_hist.pdf")

dP = size(macros, 2) - 1 + dimQ()
PCs = PCA(Array(yields[:, 2:end]), tuned.p)[1]
starting = []
for i in 1:dP
    push!(starting, AR_res_var([PCs Array(macros[:, 2:end])][:, i], tuned.p))
end
Plots.histogram(tuned.Ω0 / (tuned.ν0 - dP - 1) |> x -> x ./ starting, bins=range(0.8, 1.2, length=21), label="", ylabel="counts") |> x -> Plots.pdf(x, "/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior for TS/slide/Optimized_Omega.pdf")

rec_dates = DateTime.(["1990-07-01" "1991-03-01"
    "2001-03-01" "2001-11-01"
    "2007-12-01" "2009-06-01"
    "2020-02-01" "2020-04-01"])
plot(
    layer(x=yields[13:end, 1], y=mean(saved_TP)[:TP], Geom.line, color=[colorant"blue"]),
    layer(xmin=rec_dates[:, 1], xmax=rec_dates[:, 2], Geom.band(; orientation=:vertical), color=[colorant"grey"]),
    Guide.manual_color_key("", ["term premium", "", "NBER recessions"], ["blue", "white", "grey"]), Theme(line_width=2pt, key_position=:top, major_label_font_size=10pt, minor_label_font_size=9pt, key_label_font_size=10pt, point_size=4pt), Guide.ylabel("percent per annum"), Guide.xlabel("time"), Guide.xticks(ticks=DateTime("1986-07-01"):Month(54):DateTime("2020-08-01")), Guide.yticks(ticks=0:1:5)
) |> PDF("/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior for TS/slide/TP.pdf")

Plots.histogram(mSR_prior0, normalize=:pdf, label="", xlabel="Sharpe ratio", ylabel="density", tickfont=(10)) |> x -> Plots.pdf(x, "/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior for TS/slide/prior_mSR1.pdf")
Plots.histogram(mSR_prior1, normalize=:pdf, label="", xlabel="Sharpe ratio", ylabel="density", tickfont=(10)) |> x -> Plots.pdf(x, "/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior for TS/slide/prior_mSR2.pdf")

Plots.histogram(mSR, bins=range(0, maximum([mSR; mSR_prior]), length=41), normalize=:pdf, label="posterior", alpha=0.9)
Plots.histogram!(mSR_prior, bins=range(0, maximum([mSR; mSR_prior]), length=41), normalize=:pdf, label="prior", xlabel="Sharpe ratio", ylabel="density", tickfont=(10), alpha=0.6) |> x -> Plots.pdf(x, "/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior for TS/slide/post_mSR.pdf")

plot(x=[vec(mean(saved_θ)[:ψ]); vec(mean(saved_θ)[:ψ0])], Geom.histogram, Scale.x_log10, Theme(line_width=2pt, key_position=:top, major_label_font_size=10pt, minor_label_font_size=9pt, key_label_font_size=10pt, point_size=4pt), Guide.ylabel("counts"), Guide.xlabel("E[ψ|data] in log scale")) |> PDF("/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior for TS/slide/psi_hist.pdf")

dP = size(macros, 2) - 1 + dimQ()
Plots.histogram(100 * trace_sparsity / dP^2, label="", xlabel="Ratio of non-zeros (%)", ylabel="counts", tickfont=(10)) |> x -> Plots.pdf(x, "/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior for TS/slide/prec_hist.pdf")

rec_dates = DateTime.(["1990-07-01" "1991-03-01"
    "2001-03-01" "2001-11-01"
    "2007-12-01" "2009-06-01"
    "2020-02-01" "2020-04-01"])
plot(
    layer(x=yields[13:end, 1], y=mean(load("standard/TP.jld2")["TP"])[:TP], Geom.line, color=[colorant"black"]),
    layer(x=yields[13:end, 1], y=mean(load("mSR/TP.jld2")["TP"])[:TP], Geom.line, color=[colorant"blue"]),
    layer(x=yields[13:end, 1], y=mean(load("mSR+sparsity/TP.jld2")["TP"])[:TP], Geom.line, color=[colorant"red"], linestyle=[:dash]),
    layer(x=yields[13:end, 1], y=mean(load("mSR+prec/TP.jld2")["TP"])[:TP], Geom.line, color=[colorant"green"], linestyle=[:dot]),
    layer(x=yields[13:end, 1], y=mean(load("mSR+sparsity+prec/TP.jld2")["TP"])[:TP], Geom.line, color=[colorant"purple"], linestyle=[:dashdot]),
    layer(xmin=rec_dates[:, 1], xmax=rec_dates[:, 2], Geom.band(; orientation=:vertical), color=[colorant"grey"]),
    Guide.manual_color_key("", ["", "no restriction", "restricted SR", "sparse slope", "", "", "", "", "", "sparse precision", "full sparse", "NBER recessions", "", "", "", ""], [RGBA(1, 1, 1, 0.0000001), "black", "blue", "red", RGBA(1, 1, 1, 0.0000002), RGBA(1, 1, 1, 0.0000003), RGBA(1, 1, 1, 0.0000004), RGBA(1, 1, 1, 0.0000005), RGBA(1, 1, 1, 0.0000006), "green", "purple", "grey", RGBA(1, 1, 1, 0.0000007), RGBA(1, 1, 1, 0.0000008), RGBA(1, 1, 1, 0.0000009), RGBA(1, 1, 1, 0.000001)]),
    Theme(line_width=1.5pt, key_position=:top, major_label_font_size=10pt, minor_label_font_size=9pt, key_label_font_size=10pt, point_size=4pt),
    Guide.ylabel("percent per annum"), Guide.xlabel("time"), Guide.xticks(ticks=DateTime("1986-07-01"):Month(54):DateTime("2020-08-01")), Guide.yticks(ticks=0:1:5)
) |> PDF("/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior for TS/slide/extended_TP.pdf")

## Scenario analysis
yield_res = max.(mean(prediction)[:yields], 0)
yield_res[:, 1:3] .= 0
Plots.surface(τₙ, DateTime("2020-03-01"):Month(1):DateTime("2020-12-01"), yield_res, xlabel="maturity (months)", zlabel="yield", camera=(15, 30), legend=:none, linetype=:wireframe) |> x -> Plots.pdf(x, "/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior for TS/slide/res_yield.pdf")
macro_res = mean(prediction)[:factors][:, 4:end] .+ mean_macro |> x -> DataFrame([collect(DateTime("2020-03-01"):Month(1):DateTime("2020-12-01")) x], ["dates"; names(macros[:, 2:end])])
rename!(macro_res, Dict("S&P 500" => "SP500"))
@df macro_res[1:4, :] Plots.plot(:dates, [:RPI :INDPRO :CPIAUCSL :SP500 :INVEST], xlabel="time", tickfont=(10), legendfontsize=10, linewidth=2, label=["RPI" "INDPRO" "CPIAUCSL" "SP500" "INVEST"]) |> x -> Plots.pdf(x, "/Users/preference/Library/CloudStorage/Dropbox/Working Paper/Prior for TS/slide/res_macro.pdf")