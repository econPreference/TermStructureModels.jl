using Pkg, Revise
Pkg.activate(@__DIR__)
Pkg.instantiate()
Pkg.precompile()
using TermStructureModels

using CSV, Dates, DataFrames, XLSX, JLD2, Distributions

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
medium_tau_pr = [truncated(Normal(1, 0.25); upper=1), truncated(Normal(0.95, 0.25); upper=1), truncated(Normal(0.9, 0.25); upper=1)]
std_kQ_infty = 0.2

iteration = 2_000
burnin = 500
TP_tau = 120

scenario_TP = [12, 24, 60, 120]
scenario_horizon = 60
function gen_scene(idx_case)

    if idx_case == 1
        scene = Vector{Scenario}(undef, 36)
        for h in 1:36
            combs = zeros(1, dP - dQ + length(tau_n))
            vals = [0.0]
            scene[h] = Scenario(combinations=deepcopy(combs), values=deepcopy(vals))
        end

        combs = [1 zeros(1, dP - dQ + length(tau_n) - 1)]
        vals = [5.1]
        scene[12] = Scenario(combinations=deepcopy(combs), values=deepcopy(vals))

        combs = [1 zeros(1, dP - dQ + length(tau_n) - 1)]
        vals = [4.1]
        scene[24] = Scenario(combinations=deepcopy(combs), values=deepcopy(vals))

        combs = [1 zeros(1, dP - dQ + length(tau_n) - 1)]
        vals = [3.1]
        scene[end] = Scenario(combinations=deepcopy(combs), values=deepcopy(vals))
        return scene
    elseif idx_case == 2
        scene = Vector{Scenario}(undef, 10)
        VIX_path = raw_macros[sdate(2008, 9):sdate(2009, 6), end]
        for h in 1:10
            combs = zeros(1, dP - dQ + length(tau_n))
            vals = zeros(size(combs, 1))

            combs[1, end] = 1.0
            vals[1] = 100log(VIX_path[h]) - mean_macros[end]
            scene[h] = Scenario(combinations=deepcopy(combs), values=deepcopy(vals))
        end
        return scene
    end
end

p = 2
est, hess, results = MLE(Array(yields[18-p+1:end, 2:end]), Array(macros[18-p+1:end, 2]), tau_n, p)

tuned, opt = tuning_hyperparameter(Array(yields[:, 2:end]), Array(macros[:, 2:end]), tau_n, rho; std_kQ_infty, medium_tau, medium_tau_pr, maxiter=200, upper_p=4)
p = tuned.p

saved_params, acceptPrMH = posterior_sampler(Array(yields[18-p+1:end, 2:end]), Array(macros[18-p+1:end, 2:end]), tau_n, rho, 10, tuned; medium_tau, std_kQ_infty, medium_tau_pr)

saved_params = saved_params[burnin+1:end]
iteration = length(saved_params)

saved_params, Pr_stationary = erase_nonstationary_param(saved_params)
iteration = length(saved_params)

ineff = ineff_factor(saved_params)

reduced_params = reducedform(saved_params, Array(yields[18-p+1:end, 2:end]), Array(macros[18-p+1:end, 2:end]), tau_n)

saved_latent_params = latentspace(saved_params, Array(yields[18-p+1:end, 2:end]), tau_n)
fitted_yields = fitted_YieldCurve(collect(1:tau_n[end]), saved_latent_params)

iter_sub = (ineff[1], ineff[2], ineff[3] |> maximum, ineff[4] |> maximum, ineff[5] |> maximum, ineff[6] |> maximum) |> maximum |> ceil |> Int
saved_TP = term_premium(TP_tau, tau_n, saved_params[1:iter_sub:end], Array(yields[18-p+1:end, 2:end]), Array(macros[18-p+1:end, 2:end]))

JLD2.save("TP.jld2", "TP", saved_TP)
TP = nothing
# GC.gc() # It's better to let the garbage collector work automatically, so we remove this line.

projections = scenario_analysis([], scenario_TP, scenario_horizon, saved_params[1:iter_sub:end], Array(yields[18-p+1:end, 2:end]), Array(macros[18-p+1:end, 2:end]), tau_n; mean_macros)

JLD2.save("uncond_scenario.jld2", "projections", projections)
projections = nothing
# GC.gc() # It's better to let the garbage collector work automatically, so we remove this line.

for i in 1:2
    projections = scenario_analysis(gen_scene(i), scenario_TP, scenario_horizon, saved_params[1:iter_sub:end], Array(yields[18-p+1:end, 2:end]), Array(macros[18-p+1:end, 2:end]), tau_n; mean_macros)

    JLD2.save("scenario$i.jld2", "projections", projections)
    projections = nothing
    # GC.gc() # It's better to let the garbage collector work automatically, so we remove this line.
end