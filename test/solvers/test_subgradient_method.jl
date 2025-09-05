s = joinpath(@__DIR__, "..", "ManoptTestSuite.jl")
!(s in LOAD_PATH) && (push!(LOAD_PATH, s))

using Manifolds, ManifoldsBase, Manopt, ManoptTestSuite, Random, Test

@testset "Subgradient Plan" begin
    M = Euclidean(2)
    p = [4.0, 2.0]
    p0 = [5.0, 2.0]
    q0 = [10.0, 5.0]
    sgs = SubGradientMethodState(
        M;
        p = p0,
        stopping_criterion = StopAfterIteration(200),
        stepsize = Manopt.ConstantStepsize(M),
    )
    sgs_ac = SubGradientMethodState(
        M;
        p = q0,
        stopping_criterion = StopAfterIteration(200),
        stepsize = Manopt.ConstantStepsize(M, 1.0; type = :absolute),
    )
    sgs_ad = SubGradientMethodState(
        M;
        p = q0,
        stopping_criterion = StopAfterIteration(200),
        stepsize = Manopt.DecreasingStepsize(M; length = 1.0, type = :absolute),
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
        solve!(mp, sgs_ac)
        solve!(mp, sgs_ad)
        q1 = get_solver_result(oR)
        @test get_initial_stepsize(mp, sgs) == 1.0
        @test get_stepsize(mp, sgs, 1) == 1.0
        @test get_last_stepsize(mp, sgs, 1) == 1.0
        # Check absolute constant stepsize
        @test get_initial_stepsize(mp, sgs_ac) == 1.0
        @test get_stepsize(mp, sgs_ac, 1) ==
            1.0 / norm(get_manifold(mp), get_iterate(sgs_ac), get_subgradient(sgs_ac))
        @test get_last_stepsize(mp, sgs_ac, 1) ==
            1.0 / norm(get_manifold(mp), get_iterate(sgs_ac), get_subgradient(sgs_ac))
        # Check absolute decreasing stepsize
        @test get_initial_stepsize(mp, sgs_ad) == 1.0
        @test get_stepsize(mp, sgs_ad, 1) ==
            1.0 / norm(get_manifold(mp), get_iterate(sgs_ad), get_subgradient(sgs_ad))
        @test get_stepsize(mp, sgs_ad, 200) ==
            0.005 / norm(get_manifold(mp), get_iterate(sgs_ad), get_subgradient(sgs_ad))
        # Check Fallbacks of Problem
        @test get_cost(mp, p) == 0.0
        @test norm(M, p, get_subgradient(mp, p)) == 0
        @test_throws MethodError get_gradient(mp, sgs.p)
        @test_throws MethodError get_proximal_map(mp, 1.0, sgs.p, 1)
        @test_throws MethodError get_proximal_map!(mp, 1.0, sgs.p, 1)
        @test_throws MethodError get_proximal_map(mp, 1.0, sgs.p)
        @test_throws MethodError get_proximal_map!(mp, 1.0, sgs.p)
        sgs2 = subgradient_method(M, f, ∂f, p0; return_state = true)
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
        sgom = ManifoldSubgradientObjective(f, ∂f!; evaluation = InplaceEvaluation())
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
        subgradient_method!(M, sgom, q3)
        @test isapprox(M, q1, p)
        @test isapprox(M, q2, p)
        @test isapprox(M, q3, p)
        Random.seed!(23)
        q4 = subgradient_method(M, f, ∂f!; evaluation = InplaceEvaluation())
        @test isapprox(M, q4, p; atol = 0.5) # random point -> not that close
        # in-place
        q5 = copy(M, p0)
        subgradient_method!(M, f, ∂f!, q5; evaluation = InplaceEvaluation())
        @test isapprox(M, q3, q5)
        # Check Fallbacks of Problem
        @test get_cost(mp, q1) == 0.0
        @test norm(M, q1, get_subgradient(mp, q1)) == 0
        @test_throws MethodError get_gradient(mp, sgs.p)
        @test_throws MethodError get_proximal_map(mp, 1.0, sgs.p, 1)
        s2 = subgradient_method(
            M, f, ∂f!, p0; evaluation = InplaceEvaluation(), return_state = true
        )
        p_star2 = get_solver_result(s2)
        @test f(M, p_star2) <= f(M, p0)
    end

    @testset "Circle" begin
        Mc, fc, ∂fc, pc, pcs = ManoptTestSuite.Circle_mean_task()
        q4 = subgradient_method(Mc, fc, ∂fc, pc)
        q5 = subgradient_method(Mc, fc, ∂fc, pc; evaluation = InplaceEvaluation())
        s3 = subgradient_method(Mc, fc, ∂fc, pc; return_state = true)
        q6 = get_solver_result(s3)[]
        @test isapprox(q4, 0.0; atol = 1.0e-8)
        @test isapprox(q5, 0.0; atol = 1.0e-8)
        @test isapprox(q6, 0.0; atol = 1.0e-8)
    end
end
