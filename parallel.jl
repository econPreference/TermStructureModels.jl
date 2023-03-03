## Setting
using Distributed
n_core = 7
addprocs(n_core)
@everywhere begin
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
    Pkg.precompile()
end
@everywhere begin
    using GDTSM, BlackBoxOptim
end
using RCall, CSV, DataFrames, Dates, Plots
date_start = Date("1987-01-01", "yyyy-mm-dd")
date_end = Date("2020-02-01", "yyyy-mm-dd")

begin ## Data: macro data
    R"library(fbi)"
    raw_fred = rcopy(rcall(:fredmd, file="/Users/preference/Dropbox/code/Julia/GDTSM/current.csv", date_start=date_start, date_end=date_end, transform=true))
    excluded = ["FEDFUNDS", "TB3MS", "TB6MS", "GS1", "GS5", "GS10", "TB3SMFFM", "TB6SMFFM", "T1YFFM", "T5YFFM", "T10YFFM", "ACOGNO"]
    macros = raw_fred[:, findall(x -> !(x ∈ excluded), names(raw_fred))]
    # # scaling
    # macros[:, 21] /= 1200 #HWI
    # macros[:, 116] /= 12 #VIXCLSx 
    # macros[:, 112] /= 12 #UMCSENTx
    excluded = ["W875RX1", "IPFPNSS", "IPFINAL", "IPCONGD", "IPDCONGD", "IPNCONGD", "IPBUSEQ", "IPMAT", "IPDMAT", "IPNMAT", "IPMANSICS", "IPB51222S", "IPFUELS", "HWIURATIO", "CLF16OV", "CE16OV", "UEMPLT5", "UEMP5TO14", "UEMP15OV", "UEMP15T26", "UEMP27OV", "USGOOD", "CES1021000001", "USCONS", "MANEMP", "DMANEMP", "NDMANEMP", "SRVPRD", "USTPU", "USWTRADE", "USTRADE", "USFIRE", "USGOVT", "AWOTMAN", "AWHMAN", "CES2000000008", "CES3000000008", "HOUSTNE", "HOUSTMW", "HOUSTS", "HOUSTW", "PERMITNE", "PERMITMW", "PERMITS", "PERMITW", "NONBORRES", "DTCOLNVHFNM", "AAAFFM", "BAAFFM", "EXSZUSx", "EXJPUSx", "EXUSUKx", "EXCAUSx", "WPSFD49502", "WPSID61", "WPSID62", "CPIAPPSL", "CPITRNSL", "CPIMEDSL", "CUSR0000SAC", "CUSR0000SAS", "CPIULFSL", "CUSR0000SA0L2", "CUSR0000SA0L5", "DDURRG3M086SBEA", "DNDGRG3M086SBEA", "DSERRG3M086SBEA"]
    macros = macros[:, findall(x -> !(x ∈ excluded), names(macros))]
end

begin ## Data: yield data
    # yield(3 months) and yield(6 months)
    raw_fred = rcopy(rcall(:fredmd, file="/Users/preference/Dropbox/code/Julia/GDTSM/current.csv", date_start=date_start, date_end=date_end, transform=false))
    Y3M = raw_fred[:, :TB3MS]
    Y6M = raw_fred[:, :TB6MS]
    # longer than one year
    raw_yield = CSV.File("feds200628.csv", missingstring="NA", types=[Date; fill(Float64, 99)]) |> DataFrame |> (x -> [x[8:end, 1] x[8:end, 69:78]]) |> dropmissing
    idx = month.(raw_yield[:, 1]) |> x -> (x .!= [x[2:end]; x[end]])
    yield_year = raw_yield[idx, :]
    yield_year = yield_year[findall(x -> x == yearmonth(date_start), yearmonth.(yield_year[:, 1]))[1]:findall(x -> x == yearmonth(date_end), yearmonth.(yield_year[:, 1]))[1], :]
    yields = DataFrame([Matrix([Y3M Y6M]) Matrix(yield_year[:, 2:end])], [:M3, :M6, :Y1, :Y2, :Y3, :Y4, :Y5, :Y6, :Y7, :Y8, :Y9, :Y10])
    yields = [yield_year[:, 1] yields]
    rename!(yields, Dict(:x1 => "date"))
end
## Tuning hyper-parameters

begin
    ρ = Vector{Float64}(undef, size(macros[:, 2:end], 2))
    for i in eachindex(ρ)
        if rcopy(rcall(:describe_md, names(macros[:, 2:end])))[:, :tcode][i] ∈ ["1", "4"]
            ρ[i] = 0.9
        else
            ρ[i] = 0
        end
    end
end
tuned = tuning_hyperparameter(Array(yields[:, 2:end]), Array(macros[:, 2:end]), ρ; maxtime_EA=1200, maxtime_NM=600)

## Estimation
τₙ = [3; 6; collect(12:12:120)]
burn_in = 2_000
iteration = 10_500
issparsity = true
init_θ = posterior_sampler(Array(yields[:, 2:end]), Array(macros[:, 2:end]), τₙ, ρ, burn_in, tuned; sparsity=issparsity)[1]
par_posterior = pmap(i -> posterior_sampler(Array(yields[:, 2:end]), Array(macros[:, 2:end]), τₙ, ρ, Int(iteration / n_core), tuned; sparsity=issparsity, init_param=init_θ[(floor.(Int, collect(range(0.5burn_in, burn_in, length=n_core))))[i]]), WorkerPool(collect(2:(n_core+1))), 1:n_core)
rmprocs(2:(n_core+1))
for i in 1:n_core
    if i == 1
        global saved_θ = par_posterior[i][1]
        global acceptPr_C_σ²FF = par_posterior[i][2] * Int(iteration / n_core) / 100
        global acceptPr_ηψ = par_posterior[i][3] * Int(iteration / n_core) / 100
    else
        global saved_θ = vcat(saved_θ, par_posterior[i][1])
        acceptPr_C_σ²FF += par_posterior[i][2] * Int(iteration / n_core) / 100
        acceptPr_ηψ += par_posterior[i][3] * Int(iteration / n_core) / 100
        if i == n_core
            acceptPr_C_σ²FF *= 100 / iteration
            acceptPr_ηψ *= 100 / iteration
        end
    end
end
saved_θ, accept_rate = stationary_θ(saved_θ)
reduced_θ = reducedform(saved_θ)
sparse_θ, trace_λ, trace_sparsity = sparse_precision(saved_θ, Array(yields[:, 2:end]), Array(macros[:, 2:end]), τₙ)
reduced_sparse_θ = reducedform(sparse_θ)
