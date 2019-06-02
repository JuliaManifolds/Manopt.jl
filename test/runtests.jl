using Manopt
using Test
@testset "Manifold Tests     " begin
    include("manifolds/testGr.jl")
    include("manifolds/testGraph.jl")
    include("manifolds/testHn.jl")
    include("manifolds/testRn.jl")
    include("manifolds/testS1.jl")
    include("manifolds/testSn.jl")
    include("manifolds/testSO.jl")
    include("manifolds/testSPD.jl")
    include("manifolds/testStiefel.jl")
    include("manifolds/testSym.jl")
    include("manifolds/testGraphConstruction.jl")
    #
    include("manifolds/testManifold.jl")
    include("manifolds/testExtended.jl")
    include("manifolds/testCombined.jl")
end
@testset "Plan Tests         " begin 
    include("plans/testDebug.jl")
    include("plans/testRecord.jl")
    include("plans/testGradientPlan.jl")
    include("plans/testSubGradientPlan.jl")
end
@testset "Algorithm Tests    " begin
    include("algorithms/testBasicsAlgs.jl")
end
@testset "Function Tests     " begin
    include("functions/testAdjointDifferentials.jl")
    include("functions/testDifferentials.jl")
    include("functions/testCosts.jl")
    include("functions/testGradients.jl")
    include("functions/testProximalMaps.jl")
end
@testset "Helper & Data Tests" begin
    include("helpers/testErrorMeasures.jl")
    include("helpers/testData.jl")
end
@testset "Solver Tests       " begin
    include("solvers/testDR.jl")
    include("solvers/testCPP.jl")
    include("solvers/testGradDesc.jl")
end
