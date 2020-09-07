using Manifolds
@testset "Subgradient Plan" begin
    M = Euclidean(2)
    x = [4.0, 2.0]
    x0 = [5.0, 2.0]
    o = SubGradientMethodOptions(M, x0, StopAfterIteration(200), DecreasingStepsize(0.1))
    o.∂ = [1.0, 0.0]
    f = y -> distance(M, y, x)
    ∂f =
        y -> distance(M, x, y) == 0 ? zero_tangent_vector(M, y) :
            -2 * log(M, y, x) / distance(M, x, y)
    p = SubGradientProblem(M, f, ∂f)
    oR = solve(p, o)
    xHat = get_solver_result(oR)
    @test get_initial_stepsize(p, o) == 0.1
    @test get_stepsize(p, o, 1) == 0.1
    @test get_last_stepsize(p, o, 1) == 0.1
    # Check Fallbacks of Problen
    @test get_cost(p, x) == 0.0
    @test norm(M, x, get_subgradient(p, x)) == 0
    @test_throws ErrorException get_gradient(p, o.x)
    @test_throws ErrorException getProximalMap(p, 1.0, o.x, 1)
end
