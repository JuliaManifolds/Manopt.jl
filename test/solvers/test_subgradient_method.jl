using Manopt, ManifoldsBase, Manifolds, Test

@testset "Subgradient Plan" begin
    M = Euclidean(2)
    x = [4.0, 2.0]
    x0 = [5.0, 2.0]
    o = SubGradientMethodOptions(M, x0, StopAfterIteration(200), ConstantStepsize(M))
    o.∂ = [1.0, 0.0]
    f(M, y) = distance(M, y, x)
    @testset "Allocating Subgradient" begin
        function ∂f(M, y)
            if distance(M, x, y) == 0
                return zero_vector(M, y)
            end
            return -2 * log(M, y, x) / max(10 * eps(Float64), distance(M, x, y))
        end
        p = SubGradientProblem(M, f, ∂f)
        X = zero_vector(M, x)
        Y = get_subgradient(p, x)
        get_subgradient!(p, X, x)
        @test isapprox(M, x, X, Y)
        oR = solve!(p, o)
        xHat = get_solver_result(oR)
        @test get_initial_stepsize(p, o) == 1.0
        @test get_stepsize(p, o, 1) == 1.0
        @test get_last_stepsize(p, o, 1) == 1.0
        # Check Fallbacks of Problen
        @test get_cost(p, x) == 0.0
        @test norm(M, x, get_subgradient(p, x)) == 0
        @test_throws MethodError get_gradient(p, o.x)
        @test_throws MethodError get_proximal_map(p, 1.0, o.x, 1)
        o2 = subgradient_method(M, f, ∂f, x0; return_options=true)
        xhat2 = get_solver_result(o2)
        @test f(M, xhat2) <= f(M, x0)
    end
    @testset "Mutating Subgradient" begin
        function ∂f!(M, X, y)
            d = distance(M, x, y)
            if d == 0
                return zero_vector!(M, X, y)
            end
            log!(M, X, y, x)
            X .*= -2 / max(10 * eps(Float64), d)
            return X
        end
        p = SubGradientProblem(M, f, ∂f!; evaluation=InplaceEvaluation())
        X = zero_vector(M, x)
        Y = get_subgradient(p, x)
        get_subgradient!(p, X, x)
        @test isapprox(M, x, X, Y)
        oR = solve!(p, o)
        xHat = get_solver_result(oR)
        # Check Fallbacks of Problen
        @test get_cost(p, x) == 0.0
        @test norm(M, x, get_subgradient(p, x)) == 0
        @test_throws MethodError get_gradient(p, o.x)
        @test_throws MethodError get_proximal_map(p, 1.0, o.x, 1)
        o2 = subgradient_method(
            M, f, ∂f!, copy(x0); evaluation=InplaceEvaluation(), return_options=true
        )
        xhat2 = get_solver_result(o2)
        @test f(M, xhat2) <= f(M, x0)
    end
end
