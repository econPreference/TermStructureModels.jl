using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
Pkg.precompile()
using TermStructureModels
using CSV, Dates, DataFrames, XLSX, JLD2, Plots

date_start = Date("1987-01-01", "yyyy-mm-dd") |> x -> x - Month(18 + 2)
date_end = Date("2022-12-01", "yyyy-mm-dd")

tau_n = [1; 3; 6; 9; collect(12:6:60); collect(72:12:120)]
function data_loading(; date_start, date_end, tau_n)

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

    rho = Vector{Float64}(undef, size(macros[:, 2:end], 2))
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
            rho[i] = 0.0
            idx_diff[i] = 1
        elseif names(macros[:, 2:end])[i] ∈ ["CUMFNS", "UNRATE"]
            rho[i] = 1.0
            idx_diff[i] = 0
        elseif names(macros[:, 2:end])[i] ∈ ["CES0600000007", "VIXCLSx"]
            macros[:, i+1] = log.(macros[:, i+1]) |> x -> 100 * x
            rho[i] = 1.0
            idx_diff[i] = 0
        else
            macros[2:end, i+1] = log.(macros[2:end, i+1]) - log.(macros[1:end-1, i+1]) |> x -> 1200 * x
            rho[i] = 0.0
            idx_diff[i] = 1
        end
    end

    raw_macros = raw_macros[3:end, :]
    macros = macros[3:end, :]
    logmacros = logmacros[3:end, :]
    mean_macros = mean(macros[:, 2:end] |> Array, dims=1)[1, :]
    macros[:, 2:end] .-= mean_macros'

    ## Yield data
    raw_yield = XLSX.readdata("LW_monthly.xlsx", "Sheet1", "A293:DQ748") |> x -> [Date.(string.(x[:, 1]), DateFormat("yyyymm")) convert(Matrix{Float64}, x[:, tau_n.+1])] |> x -> DataFrame(x, ["date"; ["Y$i" for i in tau_n]])
    yields = raw_yield[findall(x -> x == yearmonth(date_start), yearmonth.(raw_yield[:, 1]))[1]:findall(x -> x == yearmonth(date_end), yearmonth.(raw_yield[:, 1]))[1], :]
    yields = yields[3:end, :]

    yields = [Date.(string.(yields[:, 1]), DateFormat("yyyy-mm-dd")) Float64.(yields[:, 2:end])]
    rename!(yields, Dict(:x1 => "date"))

    return rho, is_percent, idx_diff, logmacros, raw_macros, macros, mean_macros, yields
end
rho, is_percent, idx_diff, logmacros, raw_macros, macros, mean_macros, yields = data_loading(; date_start, date_end, tau_n)

sdate(yy, mm) = findall(x -> x == Date(yy, mm), macros[:, 1])[1]

dQ = dimQ()
dP = size(macros, 2) - 1 + dQ
medium_tau = collect(36:42)
std_kQ_infty = 0.2

iteration = 2_000
burnin = 1_000
TP_tau = 120

tuned = JLD2.load("tuned.jld2")["tuned"]
p = tuned.p
tuned = Hyperparameter(p=deepcopy(tuned.p), q=deepcopy(tuned.q), nu0=deepcopy(tuned.ν0), Omega0=deepcopy(tuned.Ω0), mean_phi_const=deepcopy(tuned.μϕ_const))

# saved_params, acceptPrMH = posterior_sampler(Array(yields[18-p+1:end, 2:end]), Array(macros[18-p+1:end, 2:end]), tau_n, rho, iteration, tuned; medium_tau, std_kQ_infty)

# saved_params = saved_params[burnin+1:end]
# iteration = length(saved_params)

# saved_params, Pr_stationary = erase_nonstationary_param(saved_params)
# iteration = length(saved_params)

# JLD2.save("est.jld2", "saved_params", saved_params)
saved_params = JLD2.load("est.jld2")["saved_params"]

scenario_TP = [12, 24, 60, 120]
scenario_horizon = 12

# uncond_projections = scenario_analysis([], scenario_TP, scenario_horizon, saved_params[1:end], Array(yields[18-p+1:end, 2:end]), Array(macros[18-p+1:end, 2:end]), tau_n; mean_macros)
# JLD2.save("uncond_scenario.jld2", "projections", uncond_projections)
uncond_projections = JLD2.load("uncond_scenario.jld2")["projections"]

function gen_scene()
    scene = Vector{Scenario}(undef, 12)
    for h in 1:12
        combs = zeros(2, dP - dQ + length(tau_n))
        vals = zeros(2)

        combs[1, 1] = 1.0
        combs[2, 18+23] = 1.0

        vals[1] = mean(uncond_projections)[:yields][h, 1] + 1.0
        vals[2] = mean(uncond_projections)[:factors][h, 3+23] - mean_macros[23] - 0.2h
        scene[h] = Scenario(combinations=deepcopy(combs), values=deepcopy(vals))
    end
    return scene
end
# projections = scenario_analysis(gen_scene(), scenario_TP, scenario_horizon, saved_params, Array(yields[18-p+1:end, 2:end]), Array(macros[18-p+1:end, 2:end]), tau_n; mean_macros)
# JLD2.save("scenario.jld2", "projections", projections)

scenario_start_date = Date("2022-12-01", "yyyy-mm-dd")
idx_date = sdate(yearmonth(scenario_start_date)...)
## constructing predictions
# load results
raw_projections = JLD2.load("scenario.jld2")["projections"]
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

raw_projections = JLD2.load("uncond_scenario.jld2")["projections"]
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
