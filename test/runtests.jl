using Statistics, Distributions, LinearAlgebra
using Test
using GDTSM

@testset "empirical.jl" begin

    # LDL(X)
    X = [4 12 -16
        12 37 -43
        -16 -43 98]
    L, D = LDL(X)
    @test X == L * D * L'

end

@testset "inference.jl" begin

    # Calculation of the inefficiency factor
    iteration = 10_000
    saved_θ = Vector{Parameter}(undef, iteration)
    for iter in 1:iteration
        κQ = randn()
        kQ_infty = randn()
        σ²FF = randn(2)
        ϕ = randn(2, 2)
        ηψ = randn()
        ψ0 = randn(2)
        ψ = randn(2, 2)
        Σₒ = randn(2)
        γ = randn(2)
        saved_θ[iter] = Parameter(κQ, kQ_infty, ϕ,
            σ²FF, ηψ, ψ, ψ0, Σₒ, γ)
    end
    @test abs(mean(ineff_factor(saved_θ)) - 1) < 0.1

end

@testset "prior.jl" begin

    medium_τ = 12 * [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    prior_κQ_ = prior_κQ(medium_τ)
    for i in eachindex(medium_τ)
        @test dcurvature_dτ(medium_τ[i]; κQ=support(prior_κQ_)[i]) < 0.01
    end

end
