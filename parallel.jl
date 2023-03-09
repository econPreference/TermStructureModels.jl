## Setting
using Distributed
# addprocs(16)
n_core = nworkers()
@everywhere begin
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
    Pkg.precompile()
end
@everywhere begin
    using GDTSM
end
using RCall, CSV, DataFrames, Dates, Plots, JLD2
date_start = Date("1986-12-01", "yyyy-mm-dd")
date_end = Date("2020-02-01", "yyyy-mm-dd")

begin ## Data: macro data
    R"library(fbi)"
    raw_fred = rcopy(rcall(:fredmd, file="current.csv", date_start=date_start, date_end=date_end, transform=false))
    excluded = ["FEDFUNDS", "TB3MS", "TB6MS", "GS1", "GS5", "GS10", "TB3SMFFM", "TB6SMFFM", "T1YFFM", "T5YFFM", "T10YFFM"]
    macros = raw_fred[:, findall(x -> !(x ∈ excluded), names(raw_fred))]
    idx = ones(Int, 1)
    for i in axes(macros[:, 2:end], 2)
        if sum(ismissing.(macros[:, i+1])) == 0
            push!(idx, i + 1)
        end
    end
    macros = macros[:, idx]
    excluded = ["W875RX1", "IPFPNSS", "IPFINAL", "IPCONGD", "IPDCONGD", "IPNCONGD", "IPBUSEQ", "IPMAT", "IPDMAT", "IPNMAT", "IPMANSICS", "IPB51222S", "IPFUELS", "HWIURATIO", "CLF16OV", "CE16OV", "UEMPLT5", "UEMP5TO14", "UEMP15OV", "UEMP15T26", "UEMP27OV", "USGOOD", "CES1021000001", "USCONS", "MANEMP", "DMANEMP", "NDMANEMP", "SRVPRD", "USTPU", "USWTRADE", "USTRADE", "USFIRE", "USGOVT", "AWOTMAN", "AWHMAN", "CES2000000008", "CES3000000008", "HOUSTNE", "HOUSTMW", "HOUSTS", "HOUSTW", "PERMITNE", "PERMITMW", "PERMITS", "PERMITW", "NONBORRES", "DTCOLNVHFNM", "AAAFFM", "BAAFFM", "EXSZUSx", "EXJPUSx", "EXUSUKx", "EXCAUSx", "WPSFD49502", "WPSID61", "WPSID62", "CPIAPPSL", "CPITRNSL", "CPIMEDSL", "CUSR0000SAC", "CUSR0000SAS", "CPIULFSL", "CUSR0000SA0L2", "CUSR0000SA0L5", "DDURRG3M086SBEA", "DNDGRG3M086SBEA", "DSERRG3M086SBEA"]
    macros = macros[:, findall(x -> !(x ∈ excluded), names(macros))]
    ρ = Vector{Float64}(undef, size(macros[:, 2:end], 2))
    for i in axes(macros[:, 2:end], 2) # i'th macro variable (excluding date)
        if rcopy(rcall(:describe_md, names(macros[:, 2:end])))[:, :tcode][i] ∈ ["1", "2", "3", "4"]
            if sum(macros[:, i+1] .<= 0) == 0
                macros[:, i+1] = log.(macros[:, i+1])
            end
            ρ[i] = 0.9
        elseif rcopy(rcall(:describe_md, names(macros[:, 2:end])))[:, :tcode][i] ∈ ["7"]
            macros[2:end, i+1] = 1200((macros[2:end, i+1]) ./ (macros[1:end-1, i+1]) .- 1)
            ρ[i] = 0
        else
            macros[2:end, i+1] = 1200(log.(macros[2:end, i+1]) - log.(macros[1:end-1, i+1]))
            ρ[i] = 0
        end
    end
    macros = macros[2:end, :]
end

begin ## Data: yield data
    # yield(3 months) and yield(6 months)
    raw_fred = rcopy(rcall(:fredmd, file="current.csv", date_start=date_start, date_end=date_end, transform=false))
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
    yields = yields[2:end, :]
end

## Tuning hyper-parameters
tuned = tuning_hyperparameter(Array(yields[:, 2:end]), Array(macros[:, 2:end]), ρ)
save("tuned.jld2", "tuned", tuned)
# tuned = load("tuned.jld2")["tuned"]

## Estimation
τₙ = [3; 6; collect(12:12:120)]
burn_in = 6_000
iteration = 20_000 |> x -> n_core + x - x % n_core
issparsity = true
init_θ = posterior_sampler(Array(yields[:, 2:end]), Array(macros[:, 2:end]), τₙ, ρ, burn_in, tuned; sparsity=issparsity)[1]
save("init_theta.jld2", "samples", init_θ)
# init_θ = load("init_theta.jld2")["samples"]
par_posterior = pmap(i -> posterior_sampler(Array(yields[:, 2:end]), Array(macros[:, 2:end]), τₙ, ρ, Int(iteration / n_core), tuned; sparsity=issparsity, init_param=init_θ[(floor.(Int, collect(range(burn_in - 1000, burn_in, length=n_core))))[i]]), 1:n_core)
# rmprocs(2:(n_core+1))
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
save("posterior.jld2", "samples", saved_θ)
# saved_θ = load("posterior.jld2")["samples"]
saved_θ, accept_rate = stationary_θ(saved_θ)
reduced_θ = reducedform(saved_θ)
sparse_θ, trace_sparsity = sparse_precision(saved_θ, size(macros, 1) - tuned.p)
save("sparse.jld2", "samples", sparse_θ)
# sparse_θ = load("sparse.jld2")["samples"]
reduced_sparse_θ = reducedform(sparse_θ)
