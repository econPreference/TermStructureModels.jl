begin # MOVE data
    raw_MOVE = CSV.File("MOVE.csv", missingstring="null", types=[Date; fill(Float64, 6)]) |> DataFrame |> (x -> [x[2:end, 1:1] x[2:end, 5:5]]) |> dropmissing
    idx = month.(raw_MOVE[:, 1]) |> x -> (x .!= [x[2:end]; x[end]])
    MOVE = raw_MOVE[idx, :]
    MOVE = MOVE[1:findall(x -> x == yearmonth(date_end), yearmonth.(MOVE[:, 1]))[1], :]
end

PCs = PCA(Array(yields[p_max-lag+1:end, 2:end]), lag)[1]
ΩPP = [AR_res_var(PCs[lag+1:end, i], lag)[1] for i in 1:dimQ()] |> diagm

mSR_upper = 1.5

tuned_set = pf_input[lag][findall(x -> x < mSR_upper, pf[lag][2])]
log_ml = pf[lag][1][findall(x -> x < mSR_upper, pf[lag][2])]
tuned = tuned_set[sortperm(log_ml, rev=true)][1]

mSR_prior = maximum_SR(Array(yields[p_max-lag+1:end, 2:end]), Array(macros[p_max-lag+1:end, 2:end]), tuned, τₙ, ρ; κQ=mean(prior_κQ(medium_τ)), kQ_infty=μkQ_infty, ΩPP)
mSR_prior |> mean
mSR_simul = maximum_SR_simul(Array(yields[p_max-lag+1:end, 2:end]), Array(macros[p_max-lag+1:end, 2:end]), tuned, τₙ, ρ; κQ=mean(prior_κQ(medium_τ)), kQ_infty=μkQ_infty, ΩPP)
Plots.plot(mSR_prior)
#Plots.plot!(mean(mSR_simul, dims=1)[1, :])
Plots.ylims!((0, 1.1maximum(mSR_prior)))
Plots.plot!(191:398, MOVE[:, 2] |> x -> (x .- mean(x)) ./ std(x) |> x -> std(mSR_prior[191:398]) * x |> x -> x .+ mean(mSR_prior[191:398]))