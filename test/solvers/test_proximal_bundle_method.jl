using Manopt, Manifolds, Test, QuadraticModels, RipQP, ManifoldDiff
import Manopt: proximal_bundle_method_subsolver, proximal_bundle_method_subsolver!

@testset "The Proximal Bundle Method" begin
    M = Hyperbolic(4)
    p = [0.0, 0.0, 0.0, 0.0, 1.0]
    p0 = [0.0, 0.0, 0.0, 0.0, -1.0]
    pbms = ProximalBundleMethodState(M; p = p0, stopping_criterion = StopAfterIteration(200))
    @test get_iterate(pbms) == p0

    pbms.X = [1.0, 0.0, 0.0, 0.0, 0.0]
    @testset "Special Stopping Criteria" begin
        sc1 = StopWhenLagrangeMultiplierLess(1.0e-8)
        @test startswith(repr(sc1), "StopWhenLagrangeMultiplierLess([1.0e-8]; mode=:estimate)")
        @test get_reason(sc1) == ""
        # Trigger manually
        sc1.at_iteration = 2
        @test length(get_reason(sc1)) > 0
        sc2 = StopWhenLagrangeMultiplierLess([1.0e-8, 1.0e-8]; mode = :both)
        @test startswith(repr(sc2), "StopWhenLagrangeMultiplierLess([1.0e-8, 1.0e-8]; mode=:both)")
        @test get_reason(sc2) == ""
        # Trigger manually
        sc2.at_iteration = 2
        @test length(get_reason(sc2)) > 0
        sc3 = StopWhenLagrangeMultiplierLess([1.0e-8, 1.0e-8]; mode = :both, names = ["a", "b"])
        # Trigger manually
        sc3.at_iteration = 2
        @test length(get_reason(sc3)) > 0
    end
    @testset "Allocating Subgradient" begin
        f(M, q) = distance(M, q, p)
        ∂f(M, q) = (distance(M, p, q) == 0) ? zero_vector(M, q) : (-log(M, q, p) / max(10 * eps(Float64), distance(M, p, q)))
        mp = DefaultManoptProblem(M, ManifoldSubgradientObjective(f, ∂f))
        X = zero_vector(M, p)
        Y = get_subgradient(mp, p)
        get_subgradient!(mp, X, p)
        @test isapprox(M, p, X, Y)
        oR = solve!(mp, pbms)
        xHat = get_solver_result(oR)
        # Check Fallbacks of Problem
        @test get_cost(mp, p) == 0.0
        @test norm(M, p, get_subgradient(mp, p)) == 0
        @test_throws MethodError get_gradient(mp, pbms.p)
        @test_throws MethodError get_proximal_map(mp, 1.0, pbms.p, 1)
        pbms2 = proximal_bundle_method(
            M,
            f,
            ∂f,
            p0;
            stopping_criterion = StopAfterIteration(200),
            return_state = true,
            debug = [],
        )
        p_star2 = get_solver_result(pbms2)
        @test get_subgradient(pbms2) == -∂f(M, p_star2)
        @test f(M, p_star2) <= f(M, p0)
        set_iterate!(pbms2, M, p)
        @test get_iterate(pbms2) == p
        # Test warnings
        dw1 = DebugWarnIfLagrangeMultiplierIncreases(:Once; tol = 0.0)
        dw1(mp, pbms, 1) #do one normal run.
        @test repr(dw1) == "DebugWarnIfLagrangeMultiplierIncreases(; tol=\"0.0\")"
        pbms.ν = 101.0
        @test_logs (:warn,) dw1(mp, pbms, 2)
        dw2 = DebugWarnIfLagrangeMultiplierIncreases(:Once; tol = 1.0e1)
        dw2.old_value = -101.0
        @test repr(dw2) == "DebugWarnIfLagrangeMultiplierIncreases(; tol=\"10.0\")"
        pbms.ν = -1.0
        @test_logs (:warn,) (:warn,) dw2(mp, pbms, 1)
    end
    @testset "Mutating Subgradient" begin
        f(M, q) = distance(M, q, p)
        function ∂f!(M, X, q)
            d = distance(M, p, q)
            if d == 0
                zero_vector!(M, X, q)
                return X
            end
            log!(M, X, q, p)
            X .*= -1 / max(10 * eps(Float64), d)
            return X
        end
        bmom = ManifoldSubgradientObjective(f, ∂f!; evaluation = InplaceEvaluation())
        mp = DefaultManoptProblem(M, bmom)
        X = zero_vector(M, p)
        Y = get_subgradient(mp, p)
        get_subgradient!(mp, X, p)
        @test isapprox(M, p, X, Y)
        sr = solve!(mp, pbms)
        xHat = get_solver_result(sr)
        # Test Fallbacks of Problem
        @test get_cost(mp, p) == 0.0
        @test norm(M, p, get_subgradient(mp, p)) == 0
        @test_throws MethodError get_gradient(mp, pbms.p)
        @test_throws MethodError get_proximal_map(mp, 1.0, pbms.p, 1)
        s2 = proximal_bundle_method(
            M,
            f,
            ∂f!,
            copy(p0);
            stopping_criterion = StopAfterIteration(200),
            evaluation = InplaceEvaluation(),
            sub_state = AllocatingEvaluation(), # keep the default allocating subsolver here
            return_state = true,
            debug = [],
        )
        p_star2 = get_solver_result(s2)
        @test f(M, p_star2) <= f(M, p0)
    end
    @testset "A simple median run" begin
        M = Sphere(2)
        p1 = [1.0, 0.0, 0.0]
        p2 = 1 / sqrt(2) .* [1.0, 1.0, 0.0]
        p3 = 1 / sqrt(2) .* [1.0, 0.0, 1.0]
        data = [p1, p2, p3]
        f(M, p) = sum(1 / length(data) * distance.(Ref(M), Ref(p), data))
        function ∂f(M, p)
            return sum(
                1 / length(data) *
                    ManifoldDiff.subgrad_distance.(Ref(M), data, Ref(p), 1; atol = 1.0e-8),
            )
        end
        p0 = p1
        pbm_s = proximal_bundle_method(M, f, ∂f, p0; return_state = true)
        @test startswith(
            Manopt.status_summary(pbm_s; inline = false),
            "# Solver state for `Manopt.jl`s Proximal Bundle Method\n"
        )
        q = get_solver_result(pbm_s)
        # with default parameters for both median and proximal bundle, this is not very precise
        m = median(M, data)
        @test distance(M, q, m) < 2 * 1.0e-3
        # test access functions
        @test get_iterate(pbm_s) == q
        @test norm(M, q, get_subgradient(pbm_s)) < 1.0e-4
        # test the other stopping criterion mode
        q2 = proximal_bundle_method(
            M, f, ∂f, p0;
            stopping_criterion = StopWhenLagrangeMultiplierLess([1.0e-8, 1.0e-8]; mode = :both),
        )
        @test distance(M, q2, m) < 2 * 1.0e-3
        # Test bundle size and in-place
        p_size = copy(p0)
        function ∂f!(M, X, p)
            X = sum(
                1 / length(data) *
                    ManifoldDiff.subgrad_distance!.(Ref(M), Ref(X), data, Ref(p), 1; atol = 1.0e-8),
            )
            return X
        end
        proximal_bundle_method!(
            M, f, ∂f!, p_size; bundle_size = 2, stopping_criterion = StopAfterIteration(200),
            evaluation = InplaceEvaluation(), sub_problem = (proximal_bundle_method_subsolver!),
        )
    end
    @testset "Trigger the case where the bundle is not transported" begin
        M = Hyperbolic(4)
        p = [0.0, 0.0, 0.0, 0.0, 1.0]
        p0 = [0.0, 0.0, 0.0, 0.0, -1.0]
        pbms = ProximalBundleMethodState(M; p = p0, stopping_criterion = StopAfterIteration(200))
        f(M, q) = distance(M, q, p)
        ∂f(M, q) = (distance(M, p, q) == 0) ? zero_vector(M, q) : (-log(M, q, p) / max(10 * eps(Float64), distance(M, p, q)))
        mp = DefaultManoptProblem(M, ManifoldSubgradientObjective(f, ∂f))
        pbms.p_last_serious = p0
        Manopt.step_solver!(mp, pbms, 1)
        # test bundle base point still p0
        @test pbms.bundle[1][1] == p0
    end
end
