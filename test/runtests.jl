using Statistics, Distributions
using Test
using .GDTSM

@testset "Empirical.jl" begin

    # LDLt(X)
    X = [4 12 -16
        12 37 -43
        -16 -43 98]
    L, D = LDLt(X)
    @test X == L * D * L'

end

@testset "GDTSM.jl" begin

    # Calculation of the inefficiency factor
    saved_θ = []
    iteration = 10_000
    for iter in 1:iteration
        κQ = randn()
        kQ_infty = randn()
        σ²FF = randn()
        ϕ = randn(2)
        ηψ = randn()
        ψ0 = randn()
        ψ = randn(2)
        Σₒ = randn()
        γ = randn()
        push!(saved_θ,
            Dict("κQ" => κQ, "kQ_infty" => kQ_infty, "ϕ" => ϕ, "σ²FF" => σ²FF, "ηψ" => ηψ, "ψ" => ψ, "ψ0" => ψ0, "Σₒ" => Σₒ, "γ" => γ)
        )
    end
    @test mean(ineff_factor(saved_θ)) < 1.1

end

@testset "priors.jl" begin

    medium_τ = 12 * [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    prior_κQ_ = prior_κQ(medium_τ)
    for i in eachindex(medium_τ)
        @test dcurvature_dτ(medium_τ[i]; κQ=support(prior_κQ_)[i]) < 0.01
    end

end
