## Setting
using GDTSM, RCall, CSV, DataFrames, Dates, Plots
date_start = Date("1987-01-01", "yyyy-mm-dd")
date_end = Date("2020-02-01", "yyyy-mm-dd")

## Data: yield(3 months) and yield(6 months)
R"library(fbi)"
raw_fred = rcopy(rcall(:fredmd, file="/Users/preference/Dropbox/code/Julia/GDTSM/current.csv", date_start=date_start, date_end=date_end, transform=false))
Y3M = raw_fred[:, :TB3MS]
Y6M = raw_fred[:, :TB6MS]
## Data: macro data
raw_fred = rcopy(rcall(:fredmd, file="/Users/preference/Dropbox/code/Julia/GDTSM/current.csv", date_start=date_start, date_end=date_end, transform=true))
macros = [raw_fred[:, 1+1:1+57] raw_fred[:, 1+59:1+83] raw_fred[:, 1+106:end]]
# selected_macros = [:DPCERA3M086SBEA, :INDPRO, :IPFINAL, :PAYEMS, :MANEMP, :CE16OV, :UNRATE, :HOUST, :PERMIT, :CPIAUCSL, :M2REAL, Symbol("S&P 500"), :TOTRESNS]
# macros = macros[:, selected_macros]

## Data: yield data
raw_yield = CSV.File("feds200628.csv", missingstring="NA", types=[Date; fill(Float64, 99)]) |> DataFrame |> (x -> [x[8:end, 1] x[8:end, 69:78]]) |> dropmissing
idx = month.(raw_yield[:, 1]) |> x -> (x .!= [x[2:end]; x[end]])
yield_year = raw_yield[idx, :]
yield_year = yield_year[findall(x -> x == yearmonth(date_start), yearmonth.(yield_year[:, 1]))[1]:findall(x -> x == yearmonth(date_end), yearmonth.(yield_year[:, 1]))[1], :]
yields = DataFrame([Y3M Y6M Matrix(yield_year[:, 2:end])], [:M3, :M6, :Y1, :Y2, :Y3, :Y4, :Y5, :Y6, :Y7, :Y8, :Y9, :Y10])

## Tuning hyper-parameters

ρ = Vector{Float64}(undef, size(macros, 2))
for i in eachindex(ρ)
    if rcopy(rcall(:describe_md, names(macros)))[:, :tcode][i] ∈ ["1", "4"]
        ρ[i] = 0.9
    else
        ρ[i] = 0
    end
end
tuned = tuning_hyperparameter(Array(yields), Array(macros), ρ)
p =
    q =
        ν0 =
            Ω0 =

            ## Estimation
                τₙ = [3; 6; 12 * [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
iteration = 10_000
saved_θ, acceptPr_C_σ²FF, acceptPr_ηψ = posterior_sampler(Array(yields), Array(macros), τₙ, ρ, iteration, tuned; sparsity=true)
saved_θ = saved_θ[round(Int, 0.1iteration):end]
saved_θ, accept_rate = stationary_θ(saved_θ)
saved_θ, trace_λ, trace_sparsity = sparse_precision(saved_θ, Array(yields), Array(macros), τₙ)
reduced_θ = reducedform(saved_θ)