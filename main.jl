## Setting
using GDTSM, RCall, CSV, DataFrames, Dates, Plots
date_start = Date("1987-01-01", "yyyy-mm-dd")
date_end = Date("2020-02-01", "yyyy-mm-dd")

begin ## Data: macro data
    R"library(fbi)"
    raw_fred = rcopy(rcall(:fredmd, file="/Users/preference/Dropbox/code/Julia/GDTSM/current.csv", date_start=date_start, date_end=date_end, transform=true))
    excluded = ["FEDFUNDS", "TB3MS", "TB6MS", "GS1", "GS5", "GS10", "TB3SMFFM", "TB6SMFFM", "T1YFFM", "T5YFFM", "T10YFFM", "ACOGNO"]
    macros = raw_fred[:, findall(x -> !(x ∈ excluded), names(raw_fred))]
    # scaling
    macros[:, 21] /= 200
    macros[:, 116] /= 5
    macros[:, 112] /= 5
end
# 21, 116, 112
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
tuned = tuning_hyperparameter(Array(yields[:, 2:end]), Array(macros[:, 2:end]), ρ; maxtime_EA=600, maxtime_NM=600, maxtime_LBFGS=600, isLBFGS=true)

## Estimation
τₙ = [3; 6; collect(12:12:120)]
iteration = 10_000
saved_θ, acceptPr_C_σ²FF, acceptPr_ηψ = posterior_sampler(Array(yields[:, 2:end]), Array(macros[:, 2:end]), τₙ, ρ, iteration, tuned; sparsity=true)
saved_θ = saved_θ[round(Int, 0.1iteration):end]
saved_θ, accept_rate = stationary_θ(saved_θ)
reduced_θ = reducedform(saved_θ)
sparse_θ, trace_λ, trace_sparsity = sparse_precision(saved_θ, Array(yields[:, 2:end]), Array(macros[:, 2:end]), τₙ)
reduced_sparse_θ = reducedform(sparse_θ)
