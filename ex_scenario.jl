scenario_date = Date("2020-03-01", "yyyy-mm-dd")

begin ## Data: macro data
    raw_fred = CSV.File("current.csv") |> DataFrame |> x -> x[314:774, :]
    raw_fred = [Date.(raw_fred[:, 1], DateFormat("mm/dd/yyyy")) raw_fred[:, 2:end]]
    raw_fred = raw_fred[findall(x -> x == yearmonth(date_start), yearmonth.(raw_fred[:, 1]))[1]:findall(x -> x == yearmonth(scenario_date), yearmonth.(raw_fred[:, 1]))[1], :]

    excluded = ["FEDFUNDS", "CP3Mx", "TB3MS", "TB6MS", "GS1", "GS5", "GS10", "TB3SMFFM", "TB6SMFFM", "T1YFFM", "T5YFFM", "T10YFFM", "COMPAPFFx", "AAAFFM", "BAAFFM"]
    scenario_macros = raw_fred[:, findall(x -> !(x ∈ excluded), names(raw_fred))]
    idx = ones(Int, 1)
    for i in axes(scenario_macros[:, 2:end], 2)
        if sum(ismissing.(scenario_macros[:, i+1])) == 0
            push!(idx, i + 1)
        end
    end
    scenario_macros = scenario_macros[:, idx]
    excluded = ["W875RX1", "IPFPNSS", "IPFINAL", "IPCONGD", "IPDCONGD", "IPNCONGD", "IPBUSEQ", "IPMAT", "IPDMAT", "IPNMAT", "IPMANSICS", "IPB51222S", "IPFUELS", "HWIURATIO", "CLF16OV", "CE16OV", "UEMPLT5", "UEMP5TO14", "UEMP15OV", "UEMP15T26", "UEMP27OV", "USGOOD", "CES1021000001", "USCONS", "MANEMP", "DMANEMP", "NDMANEMP", "SRVPRD", "USTPU", "USWTRADE", "USTRADE", "USFIRE", "USGOVT", "AWOTMAN", "AWHMAN", "CES2000000008", "CES3000000008", "HOUSTNE", "HOUSTMW", "HOUSTS", "HOUSTW", "PERMITNE", "PERMITMW", "PERMITS", "PERMITW", "NONBORRES", "DTCOLNVHFNM", "AAAFFM", "BAAFFM", "EXSZUSx", "EXJPUSx", "EXUSUKx", "EXCAUSx", "WPSFD49502", "WPSID61", "WPSID62", "CPIAPPSL", "CPITRNSL", "CPIMEDSL", "CUSR0000SAC", "CUSR0000SAS", "CPIULFSL", "CUSR0000SA0L2", "CUSR0000SA0L5", "DDURRG3M086SBEA", "DNDGRG3M086SBEA", "DSERRG3M086SBEA"]
    push!(excluded, "CMRMTSPLx", "RETAILx", "HWI", "UEMPMEAN", "CLAIMSx", "AMDMNOx", "ANDENOx", "AMDMUOx", "BUSINVx", "ISRATIOx", "BUSLOANS", "NONREVSL", "CONSPI", "S&P: indust", "S&P div yield", "S&P PE ratio", "M1SL", "BOGMBASE")
    scenario_macros = scenario_macros[:, findall(x -> !(x ∈ excluded), names(scenario_macros))]
    scenario_macros = [scenario_macros[:, 1] Float64.(scenario_macros[:, 2:end])]
    rename!(scenario_macros, Dict(:x1 => "date"))

    ρ = Vector{Float64}(undef, size(scenario_macros[:, 2:end], 2))
    idx_diff = Vector{Float64}(undef, size(scenario_macros[:, 2:end], 2))
    scenario_macros_growth = similar(scenario_macros[:, 2:end] |> Array)
    for i in axes(scenario_macros[:, 2:end], 2) # i'th macro variable (excluding date)
        if names(scenario_macros[:, 2:end])[i] ∈ ["AAA", "BAA"]
            scenario_macros[2:end, i+1] = scenario_macros[2:end, i+1] - scenario_macros[1:end-1, i+1]
            ρ[i] = 0.0
            idx_diff[i] = 1
        elseif names(scenario_macros[:, 2:end])[i] ∈ ["CUMFNS", "UNRATE", "CES0600000007", "VIXCLSx"]
            ρ[i] = 1.0
            idx_diff[i] = 0
        elseif names(scenario_macros[:, 2:end])[i] ∈ ["HOUST", "PERMIT", "REALLN", "S&P 500", "CPIAUCSL", "PCEPI", "CES0600000008", "DTCTHFNM"]
            scenario_macros_growth[2:end, i] = log.(scenario_macros[2:end, i+1]) - log.(scenario_macros[1:end-1, i+1]) |> x -> 1200 * x
            scenario_macros[2:end, i+1] = scenario_macros_growth[2:end, i]
            scenario_macros[2:end, i+1] = scenario_macros[2:end, i+1] - scenario_macros[1:end-1, i+1]
            ρ[i] = 0.0
            idx_diff[i] = 2
        else
            scenario_macros[2:end, i+1] = log.(scenario_macros[2:end, i+1]) - log.(scenario_macros[1:end-1, i+1]) |> x -> 1200 * x
            ρ[i] = 0.0
            idx_diff[i] = 1
        end
    end
    scenario_macros = scenario_macros[3:end, :]
    scenario_macros_growth = scenario_macros_growth[3:end, :]
    scenario_macros[:, 2:end] .-= mean_macros
end

begin ## Data: yield data
    raw_yield = XLSX.readdata("LW_monthly.xlsx", "Sheet1", "A132:DQ748") |> x -> [Date.(string.(x[:, 1]), DateFormat("yyyymm")) convert(Matrix{Float64}, x[:, τₙ.+1])] |> x -> DataFrame(x, ["date"; ["Y$i" for i in τₙ]])
    scenario_yields = raw_yield[findall(x -> x == yearmonth(date_start), yearmonth.(raw_yield[:, 1]))[1]:findall(x -> x == yearmonth(scenario_date), yearmonth.(raw_yield[:, 1]))[1], :]
    scenario_yields = scenario_yields[3:end, :]

    scenario_yields = [Date.(string.(scenario_yields[:, 1]), DateFormat("yyyy-mm-dd")) Float64.(scenario_yields[:, 2:end])]
    rename!(scenario_yields, Dict(:x1 => "date"))
end

dQ = dimQ()
dP = size(scenario_macros, 2) - 1 + dQ
scene = Vector{Scenario}(undef, 0)
combs = zeros(dP - dQ + 1, dP - dQ + length(τₙ))
vals = Vector{Float64}(undef, size(combs, 1))
combs[1, 1] = 1
vals[1] = 0.0
combs[2:end, length(τₙ)+1:length(τₙ)+dP-dQ] = I(dP - dQ)
vals[2:end] = scenario_macros[end, 2:end] |> Array
push!(scene, Scenario(combinations=combs, values=vals))
for h = 2:10
    local combs = zeros(1, dP - dQ + length(τₙ))
    local combs[1, 1] = 1
    local vals = [0.0]
    push!(scene, Scenario(combinations=combs, values=vals))
end

par_prediction = @showprogress 1 "Scenario..." pmap(1:ceil(Int, maximum(ineff)):iteration) do i
    scenario_sampler(scene, [24, 120], 10, [saved_θ[i]], Array(scenario_yields[upper_lag-lag+1:end, 2:end]), Array(scenario_macros[upper_lag-lag+1:end, 2:end]), τₙ; mean_macros)
end

# saved_prediction = [par_prediction[i][1] for i in eachindex(par_prediction)]
saved_prediction = Vector{Forecast}(undef, length(par_prediction))
for i in eachindex(saved_prediction)
    predicted_factors = deepcopy(par_prediction[i][1][:factors])
    for j in 1:dP-dQ
        if idx_diff[j] == 2
            predicted_factors[1, dQ+j] = scenario_macros_growth[end, j]
            predicted_factors[:, dQ+j] = predicted_factors[:, dQ+j] |> cumsum
        end
    end
    saved_prediction[i] = Forecast(yields=par_prediction[i][1][:yields], factors=predicted_factors, TP=par_prediction[i][1][:TP])
end

save("scenario.jld2", "forecasts", saved_prediction)