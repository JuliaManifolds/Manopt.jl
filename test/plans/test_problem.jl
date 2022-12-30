using Manopt, Manifolds

@testset "Test generic Problem functions" begin
    M = Euclidean(3)
    f(M, p) = norm(p)^2
    grad_f(M, p) = 2 * p
    moa = ManifoldGradientObjective(f, grad_f)
    cpa = DefaultManoptProblem(M, moa)
    @test Manopt.evaluation_type(cpa) === AllocatingEvaluation

    grad_f!(M, X, p) = (X .= 2 * p)
    moi = ManifoldGradientObjective(f, grad_f!; evaluation=InplaceEvaluation())
    cpi = DefaultManoptProblem(M, moi)
    @test Manopt.evaluation_type(cpi) === InplaceEvaluation
end
