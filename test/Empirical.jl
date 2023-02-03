using GDTSM
using Test

@testset "Empirical.jl" begin

    # LDLt(X)
    X = [4 12 -16
        12 37 -43
        -16 -43 98]
    L, D = LDLt(X)
    @test X == L * D * L'

end
