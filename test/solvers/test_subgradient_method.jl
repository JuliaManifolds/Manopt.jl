using Manopt, ManifoldsBase, Manifolds, Test

include("../utils/example_tasks.jl")

@testset "Subgradient Plan" begin
    M = Euclidean(2)
    p = [4.0, 2.0]
    p0 = [5.0, 2.0]
    sgs = SubGradientMethodState(
        M, p0; stopping_criterion=StopAfterIteration(200), stepsize=ConstantStepsize(M)
    )
    @test startswith(repr(sgs), "# Solver state for `Manopt.jl`s Subgradient Method\n")
    @test get_iterate(sgs) == p0
    sgs.X = [1.0, 0.0]
    f(M, q) = distance(M, q, p)
    @testset "Allocating Subgradient" begin
        function ∂f(M, q)
            if distance(M, p, q) == 0
                return zero_vector(M, q)
            end
            return -log(M, q, p) / max(10 * eps(Float64), distance(M, p, q))
        end
        mp = DefaultManoptProblem(M, ManifoldSubgradientObjective(f, ∂f))
        X = zero_vector(M, p)
        Y = get_subgradient(mp, p)
        get_subgradient!(mp, X, p)
        @test isapprox(M, p, X, Y)
        oR = solve!(mp, sgs)
        q1 = get_solver_result(oR)
        @test get_initial_stepsize(mp, sgs) == 1.0
        @test get_stepsize(mp, sgs, 1) == 1.0
        @test get_last_stepsize(mp, sgs, 1) == 1.0
        # Check Fallbacks of Problen
        @test get_cost(mp, p) == 0.0
        @test norm(M, p, get_subgradient(mp, p)) == 0
        @test_throws MethodError get_gradient(mp, sgs.p)
        @test_throws MethodError get_proximal_map(mp, 1.0, sgs.p, 1)
        @test_throws MethodError get_proximal_map!(mp, 1.0, sgs.p, 1)
        @test_throws MethodError get_proximal_map(mp, 1.0, sgs.p)
        @test_throws MethodError get_proximal_map!(mp, 1.0, sgs.p)
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
            X ./= -max(10 * eps(Float64), d)
            return X
        end
        sgom = ManifoldSubgradientObjective(f, ∂f!; evaluation=InplaceEvaluation())
        mp = DefaultManoptProblem(M, sgom)
        X = zero_vector(M, p)
        Y = get_subgradient(mp, p)
        get_subgradient!(mp, X, p)
        @test isapprox(M, p, X, Y)
        @test isapprox(M, p, Y, zero_vector(M, p))
        sr = solve!(mp, sgs)
        q1 = get_solver_result(sr)
        q2 = subgradient_method(M, sgom, p0)
        q3 = copy(M, p0)
        q2 = subgradient_method!(M, sgom, p0)
        # Check Fallbacks of Problen
        @test get_cost(mp, q1) == 0.0
        @test norm(M, q1, get_subgradient(mp, q1)) == 0
        @test_throws MethodError get_gradient(mp, sgs.p)
        @test_throws MethodError get_proximal_map(mp, 1.0, sgs.p, 1)
        s2 = subgradient_method(
            M, f, ∂f!, p0; evaluation=InplaceEvaluation(), return_state=true
        )
        p_star2 = get_solver_result(s2)
        @test f(M, p_star2) <= f(M, p0)
    end

    @testset "Circle" begin
        Mc, fc, ∂fc, pc, pcs = Circle_mean_task()
        q4 = subgradient_method(Mc, fc, ∂fc, pc)
        q5 = subgradient_method(Mc, fc, ∂fc, pc; evaluation=InplaceEvaluation())
        @test isapprox(q4, 0.0; atol=1e-8)
        @test isapprox(q5, 0.0; atol=1e-8)
    end
end
