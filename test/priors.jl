using GDTSM
using Test

@testset "priors.jl" begin

    for i = 1:length(medium_maturities)
        @test dcurvature_dτ.(medium_maturities[i]; κQ_candidate[i]) < eps()
    end

end
