sdate(yy, mm) = findall(x -> x == Date(yy, mm), macros[:, 1])[1]

dQ = dimQ()
dP = size(macros, 2) - 1 + dQ
scene = Vector{Scenario}(undef, 0)
combs = zeros(dP - dQ + 1, dP - dQ + length(τₙ))
vals = Vector{Float64}(undef, size(combs, 1))
combs[1, 1] = 1
vals[1] = yields[sdate(2020, 3), 2]
combs[2:end, length(τₙ)+1:length(τₙ)+dP-dQ] = I(dP - dQ)
vals[2:end] = macros[sdate(2020, 3), 2:end] |> Array
push!(scene, Scenario(combinations=combs, values=vals))
for h = 2:10
    local combs = zeros(1, dP - dQ + length(τₙ))
    local combs[1, 1] = 1
    local vals = [yields[sdate(2020, 2 + h), 2]]
    push!(scene, Scenario(combinations=combs, values=vals))
end

par_prediction = @showprogress 1 "Scenario..." pmap(1:ceil(Int, maximum(ineff)):iteration) do i
    scenario_sampler(scene, [24, 120], 10, [saved_θ[i]], Array(yields[sdate(1987, 1):sdate(2020, 2), 2:end]), Array(macros[sdate(1987, 1):sdate(2020, 2), 2:end]), τₙ; mean_macros)
end

# saved_prediction = [par_prediction[i][1] for i in eachindex(par_prediction)]
saved_prediction = Vector{Forecast}(undef, length(par_prediction))
for i in eachindex(saved_prediction)
    predicted_factors = deepcopy(par_prediction[i][1][:factors])
    for j in 1:dP-dQ
        if idx_diff[j] == 2
            predicted_factors[1, dQ+j] = macros_growth[sdate(2020, 3), j]
            predicted_factors[:, dQ+j] = predicted_factors[:, dQ+j] |> cumsum
        end
    end
    saved_prediction[i] = Forecast(yields=par_prediction[i][1][:yields], factors=predicted_factors, TP=par_prediction[i][1][:TP])
end

save("scenario.jld2", "forecasts", saved_prediction)