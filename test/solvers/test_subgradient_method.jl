using Manopt, ManifoldsBase, Manifolds, Test

@testset "Subgradient Plan" begin
    M = Euclidean(2)
    p = [4.0, 2.0]
    p0 = [5.0, 2.0]
    sgs = SubGradientMethodState(
        M, p0; stopping_criterion=StopAfterIteration(200), stepsize=ConstantStepsize(M)
    )
    @test get_iterate(sgs) == p0
    sgs.X = [1.0, 0.0]
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
        oR = solve!(mp, sgs)
        xHat = get_solver_result(oR)
        @test get_initial_stepsize(mp, sgs) == 1.0
        @test get_stepsize(mp, sgs, 1) == 1.0
        @test get_last_stepsize(mp, sgs, 1) == 1.0
        # Check Fallbacks of Problen
        @test get_cost(mp, p) == 0.0
        @test norm(M, p, get_subgradient(mp, p)) == 0
        @test_throws MethodError get_gradient(mp, sgs.p)
        @test_throws MethodError get_proximal_map(mp, 1.0, sgs.p, 1)
        sgs2 = subgradient_method(M, f, ∂f, p0; return_state=true)
        p_star2 = get_solver_result(sgs2)
        @test get_subgradient(sgs2) == -∂f(M, p_star2)
        @test f(M, p_star2) <= f(M, p0)
        set_iterate!(sgs2, M, p)
        @test get_iterate(sgs2) == p
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
        sgom = ManifoldSubgradientObjective(f, ∂f!; evaluation=InplaceEvaluation())
        mp = DefaultManoptProblem(M, sgom)
        X = zero_vector(M, p)
        Y = get_subgradient(mp, p)
        get_subgradient!(mp, X, p)
        @test isapprox(M, p, X, Y)
        sr = solve!(mp, sgs)
        xHat = get_solver_result(sr)
        # Check Fallbacks of Problen
        @test get_cost(mp, p) == 0.0
        @test norm(M, p, get_subgradient(mp, p)) == 0
        @test_throws MethodError get_gradient(mp, sgs.p)
        @test_throws MethodError get_proximal_map(mp, 1.0, sgs.p, 1)
        s2 = subgradient_method(
            M, f, ∂f!, copy(p0); evaluation=InplaceEvaluation(), return_state=true
        )
        p_star2 = get_solver_result(s2)
        @test f(M, p_star2) <= f(M, p0)
    end
end
