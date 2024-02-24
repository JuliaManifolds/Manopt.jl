using Manopt, Manifolds, Test, QuadraticModels, RipQP, ManifoldDiff
using Manopt: bundle_method_subsolver

@testset "The Convex Bundle Method" begin
    M = Hyperbolic(4)
    p = [0.0, 0.0, 0.0, 0.0, 1.0]
    p0 = [0.0, 0.0, 0.0, 0.0, -1.0]
    diam = floatmax()
    Ω = 0.0
    cbms = ConvexBundleMethodState(
        M,
        p0;
        diam=diam,
        domain=(M, q) -> distance(M, q, p0) < diam / 2 ? true : false,
        k_max=Ω,
        stopping_criterion=StopAfterIteration(200),
        sub_problem=bundle_method_subsolver,
    )
    @test get_iterate(cbms) == p0
    cbms.X = [1.0, 0.0, 0.0, 0.0, 0.0]
    @testset "Special Stopping Criteria" begin
        sc1 = StopWhenLagrangeMultiplierLess(1e-8)
        @test startswith(
            repr(sc1), "StopWhenLagrangeMultiplierLess([1.0e-8]; mode=:estimate)\n"
        )
        sc2 = StopWhenLagrangeMultiplierLess([1e-8, 1e-8]; mode=:both)
        @test startswith(
            repr(sc2), "StopWhenLagrangeMultiplierLess([1.0e-8, 1.0e-8]; mode=:both)\n"
        )
    end
    @testset "Allocating Subgradient" begin
        f(M, q) = distance(M, q, p)
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
        oR = solve!(mp, cbms)
        xHat = get_solver_result(oR)
        # Check Fallbacks of Problem
        @test get_cost(mp, p) == 0.0
        @test norm(M, p, get_subgradient(mp, p)) == 0
        @test_throws MethodError get_gradient(mp, cbms.p)
        @test_throws MethodError get_proximal_map(mp, 1.0, cbms.p, 1)
        bms2 = convex_bundle_method(
            M,
            f,
            ∂f,
            p0;
            diam=diam,
            domain=(M, q) -> distance(M, q, p0) < diam / 2 ? true : false,
            k_max=Ω,
            stopping_criterion=StopAfterIteration(200),
            sub_problem=bundle_method_subsolver,
            return_state=true,
            debug=[],
        )
        p_star2 = get_solver_result(bms2)
        @test get_subgradient(bms2) == -∂f(M, p_star2)
        @test f(M, p_star2) <= f(M, p0)
        set_iterate!(bms2, M, p)
        @test get_iterate(bms2) == p
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
        bmom = ManifoldSubgradientObjective(f, ∂f!; evaluation=InplaceEvaluation())
        mp = DefaultManoptProblem(M, bmom)
        X = zero_vector(M, p)
        Y = get_subgradient(mp, p)
        get_subgradient!(mp, X, p)
        @test isapprox(M, p, X, Y)
        sr = solve!(mp, cbms)
        xHat = get_solver_result(sr)
        # Check Fallbacks of Problem
        @test get_cost(mp, p) == 0.0
        @test norm(M, p, get_subgradient(mp, p)) == 0
        @test_throws MethodError get_gradient(mp, cbms.p)
        @test_throws MethodError get_proximal_map(mp, 1.0, cbms.p, 1)
        s2 = convex_bundle_method(
            M,
            f,
            ∂f!,
            copy(p0);
            diam=diam,
            domain=(M, q) -> distance(M, q, p0) < diam / 2 ? true : false,
            k_max=Ω,
            stopping_criterion=StopAfterIteration(200),
            sub_problem=bundle_method_subsolver,
            evaluation=InplaceEvaluation(),
            return_state=true,
            debug=[],
        )
        p_star2 = get_solver_result(s2)
        @test f(M, p_star2) <= f(M, p0)
    end
    @testset "A simple median rum" begin
        M = Sphere(2)
        p1 = [1.0, 0.0, 0.0]
        p2 = 1 / sqrt(2) .* [1.0, 1.0, 0.0]
        p3 = 1 / sqrt(2) .* [1.0, 0.0, 1.0]
        data = [p1, p2, p3]
        f(M, p) = sum(1 / length(data) * distance.(Ref(M), Ref(p), data))
        function ∂f(M, p)
            return sum(
                1 / length(data) *
                ManifoldDiff.subgrad_distance.(Ref(M), data, Ref(p), 1; atol=1e-8),
            )
        end
        p0 = p1
        cbm_s = convex_bundle_method(M, f, ∂f, p0; return_state=true)
        @test startswith(
            repr(cbm_s), "# Solver state for `Manopt.jl`s Convex Bundle Method\n"
        )
        q = get_solver_result(cbm_s)
        m = median(M, data)
        @test distance(M, q, m) < 2 * 1e-3 #with default params this is not very precise
        # tst the other stopping criterion mode
        q2 = convex_bundle_method(
            M,
            f,
            ∂f,
            p0;
            stopping_criterion=StopWhenLagrangeMultiplierLess([1e-8, 1e-8]; mode=:both),
        )
        @test distance(M, q2, m) < 2 * 1e-3
    end
end
