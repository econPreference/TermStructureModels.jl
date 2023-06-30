date_start_extended = Date("1986-02-01", "yyyy-mm-dd")
date_end_extended = Date("2020-12-01", "yyyy-mm-dd")

## Scenario Analysis
begin ## Data: macro data
    R"library(fbi)"
    raw_fred = rcopy(rcall(:fredmd, file="current.csv", date_start=date_start_extended, date_end=date_end_extended, transform=false))
    excluded = ["FEDFUNDS", "CP3Mx", "TB3MS", "TB6MS", "GS1", "GS5", "GS10", "TB3SMFFM", "TB6SMFFM", "T1YFFM", "T5YFFM", "T10YFFM", "COMPAPFFx", "AAAFFM", "BAAFFM"]
    macros_extended = raw_fred[:, findall(x -> !(x ∈ excluded), names(raw_fred))]
    idx = ones(Int, 1)
    for i in axes(macros_extended[:, 2:end], 2)
        if sum(ismissing.(macros_extended[:, i+1])) == 0
            push!(idx, i + 1)
        end
    end
    macros_extended = macros_extended[:, idx]
    excluded = ["W875RX1", "IPFPNSS", "IPFINAL", "IPCONGD", "IPDCONGD", "IPNCONGD", "IPBUSEQ", "IPMAT", "IPDMAT", "IPNMAT", "IPMANSICS", "IPB51222S", "IPFUELS", "HWIURATIO", "CLF16OV", "CE16OV", "UEMPLT5", "UEMP5TO14", "UEMP15OV", "UEMP15T26", "UEMP27OV", "USGOOD", "CES1021000001", "USCONS", "MANEMP", "DMANEMP", "NDMANEMP", "SRVPRD", "USTPU", "USWTRADE", "USTRADE", "USFIRE", "USGOVT", "AWOTMAN", "AWHMAN", "CES2000000008", "CES3000000008", "HOUSTNE", "HOUSTMW", "HOUSTS", "HOUSTW", "PERMITNE", "PERMITMW", "PERMITS", "PERMITW", "NONBORRES", "DTCOLNVHFNM", "AAAFFM", "BAAFFM", "EXSZUSx", "EXJPUSx", "EXUSUKx", "EXCAUSx", "WPSFD49502", "WPSID61", "WPSID62", "CPIAPPSL", "CPITRNSL", "CPIMEDSL", "CUSR0000SAC", "CUSR0000SAS", "CPIULFSL", "CUSR0000SA0L2", "CUSR0000SA0L5", "DDURRG3M086SBEA", "DNDGRG3M086SBEA", "DSERRG3M086SBEA"]
    push!(excluded, "CMRMTSPLx", "RETAILx", "HWI", "UEMPMEAN", "CLAIMSx", "AMDMNOx", "ANDENOx", "AMDMUOx", "BUSINVx", "ISRATIOx", "BUSLOANS", "NONREVSL", "CONSPI", "S&P: indust", "S&P div yield", "S&P PE ratio", "M1SL", "BOGMBASE")
    macros_extended = macros_extended[:, findall(x -> !(x ∈ excluded), names(macros_extended))]
    ρ = Vector{Float64}(undef, size(macros_extended[:, 2:end], 2))
    idx_trans = Vector{Float64}(undef, size(macros_extended[:, 2:end], 2))
    macros_extended_growth = similar(macros_extended[:, 2:end] |> Array)
    for i in axes(macros_extended[:, 2:end], 2) # i'th macro variable (excluding date)
        if names(macros_extended[:, 2:end])[i] ∈ ["CUMFNS", "AAA", "UNRATE", "BAA"]
            macros_extended[2:end, i+1] = macros_extended[2:end, i+1] - macros_extended[1:end-1, i+1]
            ρ[i] = 0.0
            idx_trans[i] = 0
        elseif names(macros_extended[:, 2:end])[i] ∈ ["DPCERA3M086SBEA", "HOUST", "M2SL", "M2REAL", "REALLN", "WPSFD49207", "PCEPI", "DTCTHFNM", "INVEST"]
            macros_extended_growth[2:end, i] = log.(macros_extended[2:end, i+1]) - log.(macros_extended[1:end-1, i+1]) |> x -> 1200 * x
            macros_extended[2:end, i+1] = macros_extended_growth[2:end, i]
            macros_extended[2:end, i+1] = macros_extended[2:end, i+1] - macros_extended[1:end-1, i+1]
            ρ[i] = 0.0
            idx_trans[i] = 2
        else
            macros_extended[2:end, i+1] = log.(macros_extended[2:end, i+1]) - log.(macros_extended[1:end-1, i+1]) |> x -> 1200 * x
            ρ[i] = 0.0
            idx_trans[i] = 1
        end
    end
    macros_extended = macros_extended[3:end, :]
    macros_extended_growth = macros_extended_growth[3:end, :]
    macros_extended[:, 2:end] .-= mean_macros
end

begin ## Data: yield data
    # yield(3 months) and yield(6 months)
    raw_yield = CSV.File("FRB_H15.csv", missingstring="ND", types=[Date; fill(Float64, 11)]) |> DataFrame |> (x -> [x[5137:end, 1] x[5137:end, 3:4]]) |> dropmissing
    idx = month.(raw_yield[:, 1]) |> x -> (x .!= [x[2:end]; x[end]])
    yield_month = raw_yield[idx, :]
    yield_month = yield_month[findall(x -> x == yearmonth(date_start_extended), yearmonth.(yield_month[:, 1]))[1]:findall(x -> x == yearmonth(date_end_extended), yearmonth.(yield_month[:, 1]))[1], :] |> x -> x[:, 2:end]
    # longer than one year
    raw_yield = CSV.File("feds200628.csv", missingstring="NA", types=[Date; fill(Float64, 99)]) |> DataFrame |> (x -> [x[8:end, 1] x[8:end, 69:78]]) |> dropmissing
    idx = month.(raw_yield[:, 1]) |> x -> (x .!= [x[2:end]; x[end]])
    yield_year = raw_yield[idx, :]
    yield_year = yield_year[findall(x -> x == yearmonth(date_start_extended), yearmonth.(yield_year[:, 1]))[1]:findall(x -> x == yearmonth(date_end_extended), yearmonth.(yield_year[:, 1]))[1], :]
    yields_extended = DataFrame([Matrix(yield_month) Matrix(yield_year[:, 2:end])], [:M3, :M6, :Y1, :Y2, :Y3, :Y4, :Y5, :Y6, :Y7, :Y8, :Y9, :Y10])
    yields_extended = [yield_year[:, 1] yields_extended]
    rename!(yields_extended, Dict(:x1 => "date"))
    yields_extended = yields_extended[3:end, :]
end

dQ = dimQ()
dP = size(macros_extended, 2) - 1 + dQ
scene = Vector{Scenario}(undef, 0)
combs = zeros(dP - dQ + 1, dP - dQ + length(τₙ))
vals = Vector{Float64}(undef, size(combs, 1))
combs[1, 1] = 1
vals[1] = 0.0
combs[2:end, length(τₙ)+1:length(τₙ)+dP-dQ] = I(dP - dQ)
vals[2:end] = macros_extended[end-9, 2:end] |> Array
push!(scene, Scenario(combinations=combs, values=vals))
for h = 2:3
    local combs = zeros(1, dP - dQ + length(τₙ))
    local combs[1, 1] = 1
    local vals = [0.0]
    push!(scene, Scenario(combinations=combs, values=vals))
end

par_prediction = @showprogress 1 "Scenario..." pmap(1:ceil(Int, maximum(ineff)):iteration) do i
    scenario_sampler(scene, [24, 120], 10, [saved_θ[i]], Array(yields[p_max-lag+1:end, 2:end]), Array(macros[p_max-lag+1:end, 2:end]), τₙ; mean_macros)
end
# saved_prediction = [par_prediction[i][1] for i in eachindex(par_prediction)]
saved_prediction = Vector{Forecast}(undef, length(par_prediction))
for i in eachindex(saved_prediction)
    predicted_factors = par_prediction[i][1][:factors]
    for j in 1:dP-dQ
        if idx_trans[j] == 2
            predicted_factors[1, dQ+j] = macros_extended_growth[end-9, j]
            predicted_factors[:, dQ+j] = predicted_factors[1:end, dQ+j] |> cumsum
        end
    end
    saved_prediction[i] = Forecast(yields=par_prediction[i][1][:yields], factors=predicted_factors, TP=par_prediction[i][1][:TP])
end

