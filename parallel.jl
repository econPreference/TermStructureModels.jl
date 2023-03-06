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
date_start = Date("1986-12-01", "yyyy-mm-dd")
date_end = Date("2020-02-01", "yyyy-mm-dd")

begin ## Data: macro data
    R"library(fbi)"
    raw_fred = rcopy(rcall(:fredmd, file="/Users/preference/Dropbox/code/Julia/GDTSM/current.csv", date_start=date_start, date_end=date_end, transform=false))
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
    yields = yields[2:end, :]
end

begin ## parallelize hyper-parameter tuning
    @everywhere macros = $macros
    @everywhere yields = $yields
    @everywhere ρ = $ρ
    @everywhere dQ = dimQ()
    @everywhere dP = dQ + size(Array(macros[:, 2:end]), 2)
    p_max = 4 # initial guess for the maximum lag
    @everywhere medium_τ = 12 * [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

    @everywhere function negative_log_marginal(input, p_max_)

        # parameters
        PCs = PCA(Array(yields[:, 2:end]), p_max_)[1]

        p = Int(input[1])
        if p < 1
            return Inf
        end
        q = input[2:5]
        ν0 = input[6] + dP + 1
        Ω0 = input[7:end]

        return -log_marginal(PCs[(p_max_-p)+1:end, :], Array(macros[(p_max_-p)+1:end, 2:end]), ρ, HyperParameter(p=p, q=q, ν0=ν0, Ω0=Ω0); medium_τ) # Although the input data should contains initial observations, the argument of the marginal likelihood should be the same across the candidate models. Therefore, we should align the length of the dependent variable across the models.

    end

    PCs = PCA(Array(yields[:, 2:end]), p_max)[1]
    starting = [1, 0.1, 0.1, 2, 2, 1]
    for i in 1:dP
        push!(starting, AR_res_var([PCs Array(macros[:, 2:end])][:, i], p_max))
    end
    lx = 0.0 .+ [1; zeros(4); 0; zeros(dP)]
    ux = 0.0 .+ [p_max; [1, 1, 4, 10]; 0.5size(macros, 1); 10starting[7:end]]

    ss = MixedPrecisionRectSearchSpace(lx, ux, [0; -1ones(Int64, 5 + dP)])
    obj_GSS0(x) = negative_log_marginal(x, Int(ux[1]))
    GSS_opt = bboptimize(obj_GSS0, starting; SearchSpace=ss, Method=:generating_set_search, MaxTime=60)
    corner_idx = findall([false; best_candidate(GSS_opt)[2:end] .> 0.9ux[2:end]])
    corner_p = best_candidate(GSS_opt)[1] == ux[1]

    while ~isempty(corner_idx) || corner_p
        if ~isempty(corner_idx)
            ux[corner_idx] += ux[corner_idx]
        end
        if corner_p
            ux[1] += 1
        end
        ss = MixedPrecisionRectSearchSpace(lx, ux, [0; -1ones(Int64, 5 + dP)])
        obj_GSS(x) = negative_log_marginal(x, Int(ux[1]))
        GSS_opt = bboptimize(obj_GSS, best_candidate(GSS_opt); SearchSpace=ss, Method=:generating_set_search, MaxTime=10)

        corner_idx = findall([false; best_candidate(GSS_opt)[2:end] .> 0.9ux[2:end]])
        corner_p = best_candidate(GSS_opt)[1] == ux[1]
    end

    @everywhere lx = $lx
    @everywhere ux = $ux
    @everywhere obj_EA(x) = negative_log_marginal(x, Int(ux[1]))
    EA_opt = bboptimize(bbsetup(obj_EA; SearchSpace=ss, Workers=workers()), best_candidate(GSS_opt), MaxTime=60 * 60 * 12)

    p = best_candidate(EA_opt)[1] |> Int
    q = best_candidate(EA_opt)[2:5]
    ν0 = best_candidate(EA_opt)[6] + dP + 1
    Ω0 = best_candidate(EA_opt)[7:end]

    tuned = HyperParameter(p=p, q=q, ν0=ν0, Ω0=Ω0)
end
save("tuned.jld2", "tuned", tuned)
tuned = load("tuned.jld2")["tuned"]

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
