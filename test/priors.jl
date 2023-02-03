using GDTSM
using Test

@testset "priors.jl" begin

    for i = 1:length(medium_maturities)
        @test dcurvature_dτ.(κQ_candidate[i], medium_maturities[i]) < eps()
    end

end
