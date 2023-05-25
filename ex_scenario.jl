
## Scenario Analysis
begin ## Data: macro data
    R"library(fbi)"
    raw_fred = rcopy(rcall(:fredmd, file="current.csv", date_start=Date("1985-11-01", "yyyy-mm-dd"), date_end=Date("2020-12-01", "yyyy-mm-dd"), transform=false))
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
    push!(excluded, "CMRMTSPLx", "RETAILx", "HWI", "UEMPMEAN", "CLAIMSx", "AMDMNOx", "ANDENOx", "AMDMUOx", "BUSINVx", "ISRATIOx", "BUSLOANS", "NONREVSL", "CONSPI", "S&P: indust", "S&P div yield", "S&P PE ratio", "M1SL", "BOGMBASE", "TOTRESNS", "DTCTHFNM")
    macros_extended = macros_extended[:, findall(x -> !(x ∈ excluded), names(macros_extended))]
    ρ = Vector{Float64}(undef, size(macros_extended[:, 2:end], 2))
    for i in axes(macros_extended[:, 2:end], 2) # i'th macro variable (excluding date)
        if rcopy(rcall(:describe_md, names(macros_extended[:, 2:end])))[:, :fred][i] ∈ ["CUMFNS", "AAA", "UNRATE", "BAA"]
            macros_extended[2:end, i+1] = macros_extended[2:end, i+1] - macros_extended[1:end-1, i+1]
            ρ[i] = 0.0
        elseif rcopy(rcall(:describe_md, names(macros_extended[:, 2:end])))[:, :fred][i] ∈ ["HOUST", "PERMIT", "M2REAL", "REALLN", "WPSFD49207", "CPIAUCSL", "CUSR0000SAD", "PCEPI", "CES0600000008", "CES0600000007"]
            macros_extended[2:end, i+1] = log.(macros_extended[2:end, i+1]) - log.(macros_extended[1:end-1, i+1])
            macros_extended[2:end, i+1] = macros_extended[2:end, i+1] - macros_extended[1:end-1, i+1]
            ρ[i] = 0.0
        else
            macros_extended[2:end, i+1] = log.(macros_extended[2:end, i+1]) - log.(macros_extended[1:end-1, i+1])
            ρ[i] = 0.0
        end
    end
    macros_extended = macros_extended[3:end, :]
end

dP = size(macros_extended, 2) - 1 + dimQ()
scene = Vector{Scenario}(undef, 0)
combs = zeros(dP - dimQ() + 3, dP - dimQ() + length(τₙ))
vals = Vector{Float64}(undef, size(combs, 1))
combs[1:3, 1:3] = I(3)
vals[1:3] = zeros(3)
combs[4:end, length(τₙ)+1:length(τₙ)+dP-dimQ()] = I(dP - dimQ())
vals[4:end] = macros_extended[end-9, 2:end] |> Array
push!(scene, Scenario(combinations=combs, values=vals))
for h = 2:10
    local combs = zeros(3, dP - dimQ() + length(τₙ))
    local combs[1:3, 1:3] = I(3)
    local vals = zeros(3)
    push!(scene, Scenario(combinations=combs, values=vals))
end
prediction = scenario_sampler(scene, 24, 10, saved_θ, Array(yields[p_max-lag+1:end, 2:end]), Array(macros[p_max-lag+1:end, 2:end]), τₙ)
