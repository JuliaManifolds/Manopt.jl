using Manopt, ManifoldsBase, Manifolds, LinearAlgebra, Test

@testset "Plan Tests         " begin
    include("plans/test_options.jl")
    include("plans/test_debug.jl")
    include("plans/test_nelder_mead_plan.jl")
    include("plans/test_gradient_plan.jl")
    include("plans/test_record.jl")
    include("plans/test_stopping_criteria.jl")
    include("plans/test_subgradient_plan.jl")
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
    include("solvers/test_conjugate_gradient.jl")
    include("solvers/test_Douglas_Rachford.jl")
    include("solvers/test_cyclic_proximal_point.jl")
    include("solvers/test_gradient_descent.jl")
    include("solvers/test_Nelder_Mead.jl")
    include("solvers/test_trust_regions.jl")
end
