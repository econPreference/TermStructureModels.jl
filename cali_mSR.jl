pf = load("mSR/tuned_pf.jld2")["pf"]
pf_input = load("mSR/tuned_pf.jld2")["pf_input"]

begin # MOVE data
    raw_MOVE = CSV.File("MOVE.csv", types=[Date; Float64]) |> DataFrame |> x -> x[9:end, :] |> reverse
    idx = month.(raw_MOVE[:, 1]) |> x -> (x .!= [x[2:end]; x[end]])
    MOVE = raw_MOVE[idx, :]
    MOVE = MOVE[1:findall(x -> x == yearmonth(date_end), yearmonth.(MOVE[:, 1]))[1], :]
end

mSR_upper = [1; 0.1]
pf_vec = Matrix{Float64}(undef, p_max * opt_size, 3)
pf_input_vec = Vector{Hyperparameter}(undef, p_max * opt_size)
for i in 1:p_max
    pf_vec[opt_size*(i-1)+1:opt_size*i, :] = pf[i]
    pf_input_vec[opt_size*(i-1)+1:opt_size*i] = pf_input[i]
end
idx = (pf_vec[:, 2] .< mSR_upper[1]) .* (pf_vec[:, 3] .< mSR_upper[2])
tuned_set = pf_input_vec[idx]
log_ml = pf_vec[idx, 1]
tuned = tuned_set[sortperm(log_ml, rev=true)][1]
@show lag = tuned.p

PCs = PCA(Array(yields[p_max-lag+1:end, 2:end]), lag)[1]
ΩPP = [AR_res_var(PCs[lag+1:end, i], lag)[1] for i in 1:dimQ()] |> diagm

@show prior_const_TP(tuned, 120, Array(yields[p_max-lag+1:end, 2:end]), τₙ, ρ; iteration=1000) |> std

mSR_prior, mSR_const = maximum_SR(Array(yields[p_max-lag+1:end, 2:end]), Array(macros[p_max-lag+1:end, 2:end]), tuned, τₙ, ρ; κQ=mean(prior_κQ(medium_τ)), kQ_infty=μkQ_infty, ΩPP)
@show mSR_prior |> mean
@show tuned.q
@show mSR_const
mSR_simul, mSR_const_simul = maximum_SR_simul(Array(yields[p_max-lag+1:end, 2:end]), Array(macros[p_max-lag+1:end, 2:end]), tuned, τₙ, ρ; κQ=mean(prior_κQ(medium_τ)), kQ_infty=μkQ_infty, ΩPP)
Plots.plot(yields[10:end, 1], mSR_prior, linewidth=2)
# Plots.plot!(yields[10:end, 1], mean(mSR_simul, dims=1)[1, :], line=:dash, linewidth=2)
# mSR_const_simul |> mean
aux_idx = length(mSR_prior)-length(MOVE[:, 1])+1:length(mSR_prior)
Plots.plot!(yields[10:end, 1][aux_idx], MOVE[:, 2] |> x -> (x .- mean(x)) ./ std(x) |> x -> std(mSR_prior[aux_idx]) * x |> x -> x .+ mean(mSR_prior[aux_idx]), line=:dash, linewidth=2)