using Manopt, ManifoldsBase, Manifolds, LinearAlgebra, Test

@testset "Subgradient Plan" begin
    M = Euclidean(2)
    x = [4.0, 2.0]
    x0 = [5.0, 2.0]
    o = SubGradientMethodOptions(M, x0, StopAfterIteration(200), ConstantStepsize(1.0))
    o.∂ = [1.0, 0.0]
    f = y -> distance(M, y, x)
    ∂f = y -> if distance(M, x, y) == 0
        zero_tangent_vector(M, y)
    else
        -2 * log(M, y, x) / max(10 * eps(Float64), distance(M, x, y))
    end
    p = SubGradientProblem(M, f, ∂f)
    oR = solve(p, o)
    xHat = get_solver_result(oR)
    @test get_initial_stepsize(p, o) == 1.0
    @test get_stepsize(p, o, 1) == 1.0
    @test get_last_stepsize(p, o, 1) == 1.0
    # Check Fallbacks of Problen
    @test get_cost(p, x) == 0.0
    @test norm(M, x, get_subgradient(p, x)) == 0
    @test_throws MethodError get_gradient(p, o.x)
    @test_throws MethodError get_proximal_map(p, 1.0, o.x, 1)
    o2 = subgradient_method(M, f, ∂f, copy(x0); return_options=true)
    xhat2 = get_solver_result(o2)
    @test f(xhat2) <= f(x0)
end
