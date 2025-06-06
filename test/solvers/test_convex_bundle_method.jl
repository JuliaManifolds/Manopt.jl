using Manopt, Manifolds, Test, QuadraticModels, RipQP, ManifoldDiff, LinearAlgebra, Random

using Manopt: ζ_1, ζ_2, close_point, _domain_condition, _null_condition
using Manopt: estimate_sectional_curvature

@testset "Convex Bundle Method Tests" begin
    M = Hyperbolic(4)
    p = [0.0, 0.0, 0.0, 0.0, 1.0]
    q0 = [1.0, 0.0, 0.0, 0.0, √2]# [0.0, 0.0, 0.0, 0.0, -1.0]
    p0 = exp(M, p, 4log(M, p, q0))
    diameter = floatmax()
    Ω = 0.0
    ω = -1.0

    @testset "Curvature Bound Functions" begin
        @test ζ_1(0.5, 1.0) == 1.0
        @test ζ_1(-0.5, 1.0) ≈ sqrt(0.5) * coth(sqrt(0.5))
        @test ζ_2(-0.5, 1.0) == 1.0
        @test ζ_2(0.5, 1.0) ≈ sqrt(0.5) * cot(sqrt(0.5))
    end

    @testset "Estimate Sectional Curvature" begin
        curv = estimate_sectional_curvature(M, p)
        @test isfinite(curv)
        curvature_cbms = ConvexBundleMethodState(M; p=p0)
        @test ω ≤ curvature_cbms.k_min
        @test Ω ≥ curvature_cbms.k_max
    end

    @testset "Close Point Function" begin
        tol = 0.1
        q = close_point(M, p, tol)
        @test distance(M, p, q) <= tol
    end

    cbms = ConvexBundleMethodState(
        M;
        p=p0,
        atol_λ=1e0,
        diameter=diameter,
        domain=(M, q) -> distance(M, q, p0) < diameter / 2 ? true : false,
        k_max=Ω,
        k_min=ω,
        stepsize=Manopt.DomainBackTrackingStepsize(M; contraction_factor=0.975),
        stopping_criterion=StopAfterIteration(200),
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

        # Reset the serious iterate
        p0 = [0.0, 0.0, 0.0, 0.0, -1.0]
        set_iterate!(cbms, M, p0)
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

        @testset "Domain and Null Conditions" begin
            @test _domain_condition(M, p, p0, 1.0, 1.0, cbms.domain)
            @test !_null_condition(
                mp,
                M,
                p,
                p0,
                cbms.X,
                cbms.g,
                cbms.vector_transport_method,
                cbms.inverse_retraction_method,
                cbms.m,
                1.0,
                cbms.ξ,
                cbms.ϱ,
            )
        end

        @testset "Stepsize and Debugging" begin
            io = IOBuffer()
            ds = DebugStepsize(; io=io)
            bms2 = convex_bundle_method(
                M,
                f,
                ∂f,
                p0;
                diameter=diameter,
                domain=(M, q) -> distance(M, q, p0) < diameter / 2 ? true : false,
                k_max=Ω,
                k_min=ω,
                stopping_criterion=StopAfterIteration(200),
                return_state=true,
                debug=[],
            )
            p_star2 = get_solver_result(bms2)
            @test get_subgradient(bms2) == -∂f(M, p_star2)
            @test f(M, p_star2) <= f(M, p0)
            set_iterate!(bms2, M, p)
            @test get_iterate(bms2) == p
            io = IOBuffer()
            ds = DebugStepsize(; io=io)
            # reset stepsize
            bms2.stepsize(mp, bms2, 0)
            bms2.stepsize(mp, bms2, 1)
            ds(mp, bms2, 1)
            s = String(take!(io))
            @test s == "s:1.0"
        end

        @testset "Warnings" begin
            dw1 = DebugWarnIfLagrangeMultiplierIncreases(:Once; tol=0.0)
            @test repr(dw1) == "DebugWarnIfLagrangeMultiplierIncreases(; tol=\"0.0\")"
            cbms.ξ = 101.0
            @test_logs (:warn,) dw1(mp, cbms, 1)
            dw2 = DebugWarnIfLagrangeMultiplierIncreases(:Once; tol=1e1)
            dw2.old_value = -101.0
            @test repr(dw2) == "DebugWarnIfLagrangeMultiplierIncreases(; tol=\"10.0\")"
            cbms.ξ = -1.0
            @test_logs (:warn,) (:warn,) dw2(mp, cbms, 1)
        end
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
            diameter=diameter,
            domain=(M, q) -> distance(M, q, p0) < diameter / 2 ? true : false,
            k_max=Ω,
            stopping_criterion=StopAfterIteration(200),
            evaluation=InplaceEvaluation(),
            sub_problem=convex_bundle_method_subsolver!,
            return_state=true,
            debug=[],
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
                ManifoldDiff.subgrad_distance.(Ref(M), data, Ref(p), 1; atol=1e-8),
            )
        end
        p0 = p1
        cbm_s = convex_bundle_method(M, f, ∂f, p0; k_max=1.0, k_min=1.0, return_state=true)
        @test startswith(
            repr(cbm_s), "# Solver state for `Manopt.jl`s Convex Bundle Method\n"
        )
        q = get_solver_result(cbm_s)
        m = median(M, data)
        @test distance(M, q, m) < 2e-2 #with default parameters this is not very precise
        # test the other stopping criterion mode
        q2 = convex_bundle_method(
            M,
            f,
            ∂f,
            p0;
            k_max=1.0,
            stopping_criterion=StopWhenLagrangeMultiplierLess([1e-6, 1e-6]; mode=:both),
        )
        @test distance(M, q2, m) < 2e-2
        # try to force entering the backtracking loop
        diam = π / 4
        domf(M, p) = distance(M, p, p0) < diam / 2 ? true : false
        q2 = convex_bundle_method(
            M,
            f,
            ∂f,
            p0;
            k_max=1.0,
            diameter=diam,
            domain=domf,
            stopping_criterion=StopAfterIteration(3),
        )
    end

    @testset "Null Step Backtracking Stepsize Loop" begin
        M = Sphere(2)
        p = [1.0, 0.0, 0.0]
        q = [0.0, 1.0, 0.0]
        f(M, q) = distance(M, q, p)
        function ∂f(M, q)
            if distance(M, p, q) == 0
                return zero_vector(M, q)
            end
            return -log(M, q, p) / max(10 * eps(Float64), distance(M, p, q))
        end
        cbms = ConvexBundleMethodState(
            M,
            convex_bundle_method_subsolver;
            p=q,
            k_max=1.0,
            k_min=1.0,
            stepsize=DomainBackTrackingStepsize(M; contraction_factor=0.975),
            stopping_criterion=StopAfterIteration(20),
        )
        mp = DefaultManoptProblem(M, ManifoldSubgradientObjective(f, ∂f))

        # Manually set cbms parameters to ensure _null_condition is true
        cbms.ξ = -1.0
        cbms.ϱ = -10.0
        cbms.m = 1.0
        cbms.g .= [0.1, 0.1, 0.1]

        nsbt = NullStepBackTrackingStepsize(M; initial_stepsize=cbms.last_stepsize)
        @test get_initial_stepsize(nsbt) == 1
        @test nsbt(mp, cbms, 1) < 1e-15 # Expected value?

        # nsbt show/status
        @test startswith(repr(nsbt), "NullStepBackTracking(;\n")
        @test startswith(Manopt.status_summary(nsbt), "NullStepBackTracking(;\n")
        @test endswith(Manopt.status_summary(nsbt), "e-16")
        # Test show/summary on domainbt
        dbt = DomainBackTrackingStepsize(M; contraction_factor=0.975)
        @test startswith(repr(dbt), "DomainBackTracking(;\n")
        @test startswith(Manopt.status_summary(dbt), "DomainBackTracking(;\n")
        @test endswith(Manopt.status_summary(dbt), "of 1.0")
    end

    @testset "Bundle Cap Condition" begin
        M = Sphere(2)
        p = [1.0, 0.0, 0.0]
        q = [0.0, 1.0, 0.0]
        f(M, q) = distance(M, q, p)
        function ∂f(M, q)
            d = distance(M, p, q)
            return d == 0 ? zero_vector(M, q) : -log(M, q, p) / d
        end
        diam = π / 2
        domf(M, p) = distance(M, p, p) < diam / 2 ? true : false
        cbms = ConvexBundleMethodState(
            M,
            convex_bundle_method_subsolver;
            diameter=diam,
            domain=domf,
            bundle_cap=3,
            p=q,
            k_max=1.0,
            k_min=1.0,
            stepsize=DomainBackTrackingStepsize(M; contraction_factor=0.975),
            stopping_criterion=StopAfterIteration(20),
        )
        mp = DefaultManoptProblem(M, ManifoldSubgradientObjective(f, ∂f))
        initialize_solver!(mp, cbms)

        # Manually populate the bundle to reach the bundle_cap
        for i in 1:2
            push!(cbms.bundle, (copy(M, p), copy(M, p, cbms.X)))
            push!(cbms.linearization_errors, 0.0)
            push!(cbms.λ, 0.0)
            push!(cbms.transported_subgradients, zero_vector(M, p))
        end

        # Ensure the first element in the bundle is not equal to p_last_serious
        cbms.p_last_serious .= [0.0, 1.0, 0.0]

        step_solver!(mp, cbms, 1)

        @test length(cbms.bundle) == cbms.bundle_cap
        @test cbms.bundle[1][1] ≠ cbms.p_last_serious
        @test length(cbms.linearization_errors) == length(cbms.bundle)
        @test length(cbms.λ) == length(cbms.bundle)

        # step_solver!(mp, cbms, 2)
    end
end
