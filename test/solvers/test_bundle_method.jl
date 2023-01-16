using Manopt, ManifoldsBase, Manifolds, Test, QuadraticModels, RipQP

@testset "Subgradient Plan for Bundle Method" begin
    M = Hyperbolic(4)
    p = [0.0, 0.0, 0.0, 0.0, 1.0]
    p0 = [0.0, 0.0, 0.0, 0.0, -1.0]
    bms = BundleMethodState(
        M, p0; stopping_criterion=StopAfterIteration(200),
    )
    @test get_iterate(bms) == p0
    bms.X = [1.0, 0.0, 0.0, 0.0, 0.0]
    f(M, q) = distance(M, q, p)
    @testset "Allocating Subgradient" begin
        function ∂f(M, q)
            if distance(M, p, q) == 0
                return zero_vector(M, q)
            end
            return -2 * log(M, q, p) / max(10 * eps(Float64), distance(M, p, q))
        end
        mp = DefaultManoptProblem(M, ManifoldSubgradientObjective(f, ∂f))
        X = zero_vector(M, p)
        Y = get_subgradient(mp, p)
        get_subgradient!(mp, X, p)
        @test isapprox(M, p, X, Y)
        oR = solve!(mp, bms)
        xHat = get_solver_result(oR)
        # Check Fallbacks of Problen
        @test get_cost(mp, p) == 0.0
        @test norm(M, p, get_subgradient(mp, p)) == 0
        @test_throws MethodError get_gradient(mp, bms.p)
        @test_throws MethodError get_proximal_map(mp, 1.0, bms.p, 1)
        bms2 = bundle_method(M, f, ∂f, p0; return_state=true)
        p_star2 = get_solver_result(bms2)
        @test get_subgradient(bms2) == -∂f(M, p_star2)
        @test f(M, p_star2) <= f(M, p0)
        set_iterate!(bms2, M, p)
        @test get_iterate(bms2) == p
    end

    @testset "Mutating Subgradient" begin
        function ∂f!(M, X, q)
            d = distance(M, p, q)
            if d == 0
                zero_vector!(M, X, q)
                return X
            end
            log!(M, X, q, p)
            X .*= -2 / max(10 * eps(Float64), d)
            return X
        end
        bmom = ManifoldSubgradientObjective(f, ∂f!; evaluation=InplaceEvaluation())
        mp = DefaultManoptProblem(M, bmom)
        X = zero_vector(M, p)
        Y = get_subgradient(mp, p)
        get_subgradient!(mp, X, p)
        @test isapprox(M, p, X, Y)
        sr = solve!(mp, bms)
        xHat = get_solver_result(sr)
        # Check Fallbacks of Problen
        @test get_cost(mp, p) == 0.0
        @test norm(M, p, get_subgradient(mp, p)) == 0
        @test_throws MethodError get_gradient(mp, bms.p)
        @test_throws MethodError get_proximal_map(mp, 1.0, bms.p, 1)
        s2 = bundle_method(
            M, f, ∂f!, copy(p0); evaluation=InplaceEvaluation(), return_state=true
        )
        p_star2 = get_solver_result(s2)
        @test f(M, p_star2) <= f(M, p0)
    end
end
