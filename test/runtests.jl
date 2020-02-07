using Manopt
using Test
@testset "Plan Tests         " begin
    include("plans/testDebug.jl")
    include("plans/testRecord.jl")
    include("plans/testGradientPlan.jl")
    include("plans/testSubGradientPlan.jl")
    include("plans/testStoppingCriteria.jl")
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
    include("solvers/testNelderMead.jl")
    include("solvers/testTrustRegions.jl")
end
