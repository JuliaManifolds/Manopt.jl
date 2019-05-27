using Manopt
using Test
@testset "Manopt.jl Manifold Tests" begin
    include("manifolds/testRn.jl")
    include("manifolds/testSn.jl")
    include("manifolds/testSO.jl")
    include("manifolds/testSPD.jl")
    include("manifolds/testGraphConstruction.jl")
end
@testset "Manopt.jl Algorithms" begin
    include("algorithms/testBasicsAlgs.jl")
end
@testset "Manopt.jl Function Tests" begin
    include("functions/testDifferentials.jl")
    include("functions/testGradients.jl")
    include("functions/testProximalMaps.jl")
end
@testset "Manopt.jl Solver Tests" begin
    include("testGradDesc.jl")
end
