using Manopt, ManifoldsBase, Manifolds, LinearAlgebra, Test

@testset "Plan Tests         " begin
    include("plans/test_options.jl")
    include("plans/testDebug.jl")
    include("plans/test_nelder_mead_plan.jl")
    include("plans/testGradientPlan.jl")
    include("plans/testRecord.jl")
    include("plans/testStoppingCriteria.jl")
    include("plans/testSubGradientPlan.jl")
end
@testset "Function Tests     " begin
    include("functions/testAdjointDifferentials.jl")
    include("functions/testDifferentials.jl")
    include("functions/testCosts.jl")
    include("functions/testGradients.jl")
    include("functions/testProximalMaps.jl")
    include("functions/test_manifold.jl")
end
@testset "Helper & Data Tests" begin
    include("helpers/testErrorMeasures.jl")
    include("helpers/testData.jl")
end
@testset "Solver Tests       " begin
    include("solvers/testDR.jl")
    include("solvers/testCPP.jl")
    include("solvers/testGradDesc.jl")
    include("solvers/testNelderMead.jl")
    include("solvers/testTrustRegions.jl")
end
