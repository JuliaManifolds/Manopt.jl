using Manopt, Manifolds, Test
using LinearAlgebra: I, eigen, eigvecs, eigvals, tr, Diagonal, dot, norm, Symmetric

using RecursiveArrayTools

@testset "Riemannian quasi-Newton Methods with box-like domains" begin
    @testset "get_stepsize_bound - basic" begin
        M = Hyperrectangle([0.0, 0.0], [2.0, 2.0])

        # d[i] > 0
        p = [0.0, 1.0]; d = [1.0, 1.0]
        @test Manopt.get_stepsize_bound(M, p, d, 1) ≈ (2.0 - 0.0) / 1.0  # = 2.0
        @test Manopt.get_stepsize_bound(M, p, d, 2) ≈ (2.0 - 1.0) / 1.0  # = 1.0

        # d[i] < 0
        p = [0.0, 1.0]; d = [-1.0, -1.0]
        @test Manopt.get_stepsize_bound(M, p, d, 1) ≈ (0.0 - 0.0) / -1.0  # = 0.0
        @test Manopt.get_stepsize_bound(M, p, d, 2) ≈ (0.0 - 1.0) / -1.0  # = 1.0

        # d[i] = 0
        p = [0.0, 1.0]; d = [0.0, 0.0]
        @test Manopt.get_stepsize_bound(M, p, d, 1) ≈ Inf
        @test Manopt.get_stepsize_bound(M, p, d, 2) ≈ Inf
    end

    @testset "set_stepsize_bound!" begin
        M = Hyperrectangle([0.0, 0.0], [2.0, 2.0])
        # p[1] is at its lower bound and d[1] points outward: bound 0, entry is zeroed;
        # d[2] reaches the upper bound at t = 0.5
        p = [0.0, 1.0]; d = [-1.0, 2.0]
        # t_current beyond the bound of entry 2: it is cut to the distance to the bound
        d_out = copy(d)
        @test Manopt.set_stepsize_bound!(M, d_out, p, d, 1.0) === d_out
        @test d_out == [0.0, 1.0]
        # t_current within the bound of entry 2: it is kept
        d_out = copy(d)
        Manopt.set_stepsize_bound!(M, d_out, p, d, 0.25)
        @test d_out == [0.0, 2.0]
    end

    @testset "segment Hessian updater - basic d = -g" begin
        M = Hyperrectangle([0.0, 1.0], [3.0, 3.0])

        d = [-1.0, -4.0]
        p = [0.0, 0.0]

        dt = 0.25
        db = -4.0 # for d = -g the entry of d is the negated gradient entry
        ha = QuasiNewtonMatrixDirectionUpdate(M, BFGS(), DefaultOrthonormalBasis(), [2.0 0.0; 0.0 2.0])
        b = 2
        z = [-0.25, -1.0]

        # optimized formula
        upd = Manopt.GenericSegmentHessianUpdater(similar(d), similar(d))
        Manopt.init_updater!(M, upd, p, d, ha)
        hv_eb_dz, hv_eb_d = upd(M, p, 0 + dt, dt, b, db, ha)
        @test hv_eb_dz ≈ -2.0
        @test hv_eb_d ≈ -8.0

        # original formula

        original_hv_eb_dz = dot([0, 1], ha.matrix, z)
        original_hv_eb_d = dot([0, 1], ha.matrix, d)

        @test hv_eb_dz == original_hv_eb_dz
        @test hv_eb_d == original_hv_eb_d
    end

    @testset "segment Hessian updater - basic d = [-2.0, -1.0]" begin
        M = Hyperrectangle([0.0, 1.0], [3.0, 3.0])

        d = [-2.0, -1.0]
        p = [0.0, 0.0]

        dt = 0.25
        db = -2.0
        ha = QuasiNewtonMatrixDirectionUpdate(M, BFGS(), DefaultOrthonormalBasis(), [2.0 0.0; 0.0 2.0])
        b = 1
        z = [-0.5, -0.25]

        # optimized formula
        upd = Manopt.GenericSegmentHessianUpdater(similar(d), similar(d))
        Manopt.init_updater!(M, upd, p, d, ha)
        hv_eb_dz, hv_eb_d = upd(M, p, 0 + dt, dt, b, db, ha)
        @test hv_eb_dz == -1.0
        @test hv_eb_d == -4.0

        # original formula

        original_hv_eb_dz = dot([1, 0], ha.matrix, z)
        original_hv_eb_d = dot([1, 0], ha.matrix, d)

        @test hv_eb_dz == original_hv_eb_dz
        @test hv_eb_d == original_hv_eb_d
    end

    @testset "segment Hessian updater - limited memory update" begin
        M = Hyperrectangle([1.0, 4.0], [2.0, 10.0])

        p = [2.0, 5.0]
        ha = QuasiNewtonLimitedMemoryBoxDirectionUpdate(QuasiNewtonLimitedMemoryDirectionUpdate(M, p, InverseBFGS(), 2))
        st = QuasiNewtonState(M)

        @test startswith(repr(ha), "QuasiNewtonLimitedMemoryBoxDirectionUpdate with internal state:")
        @test startswith(
            Manopt.status_summary(ha),
            "A limited memory quasi Newton direction update with support for box constraints",
        )
        @test startswith(
            Manopt.status_summary(ha; context = :inline),
            "A limited memory direction update with support for box constraints; internally: ",
        )

        f(M, p) = sum(p .^ 2)
        grad_f(M, p) = 2 * p
        gmp = ManifoldGradientObjective(f, grad_f)
        mp = DefaultManoptProblem(M, gmp)

        st.yk = [2.0, 4.0]
        st.sk = [4.0, 2.0]
        grad = grad_f(M, p)
        st.p = p
        st.X = grad
        update_hessian!(ha, mp, st, p, 1)

        d = similar(grad)
        ha(d, mp, st)

        d2 = ha(mp, st)
        @test d ≈ d2

        b = 1

        dt = 0.25
        db = d[b]

        t_current = 0 + dt

        # compare the generic and limited memory updater
        gupd = Manopt.GenericSegmentHessianUpdater(similar(d), similar(d))
        Manopt.init_updater!(M, gupd, p, d, ha)
        hv_eb_dz, hv_eb_d = gupd(M, p, t_current, dt, b, db, ha)

        @test hv_eb_dz ≈ -0.125
        @test hv_eb_d ≈ -0.5

        lmupd = Manopt.get_default_hessian_segment_updater(M, p, ha)
        @test lmupd isa Manopt.LimitedMemorySegmentHessianUpdater

        Manopt.init_updater!(M, lmupd, p, d, ha)
        hv_eb_dz_limited, hv_eb_d_limited = lmupd(M, p, t_current, dt, b, db, ha)

        @test hv_eb_dz ≈ hv_eb_dz_limited
        @test hv_eb_d ≈ hv_eb_d_limited

        ha.last_gcd_result = :found_unlimited
        ha.last_gcd_stepsize = Inf
        @test Manopt.get_parameter(ha, Val(:max_stepsize)) == Inf

        @testset "No memory tests" begin
            ha2 = QuasiNewtonLimitedMemoryBoxDirectionUpdate(QuasiNewtonLimitedMemoryDirectionUpdate(M, p, InverseBFGS(), 2))
            @test Manopt.hessian_value(ha2, M, p, Manopt.UnitVector(b), grad) ≈ 4.0
            Manopt.update_current_scale!(M, p, ha2)
            @test ha2.current_scale == ha2.qn_du.initial_scale
            @test ha2.M_11 == fill(0.0, 0, 0)
            @test ha2.M_21 == fill(0.0, 0, 0)
            @test ha2.M_22 == fill(0.0, 0, 0)
            # resetting the memory also resets the scale, so that the limited memory
            # segment updater agrees with the generic one
            ha2.current_scale = 5.0
            Manopt.initialize_update!(ha2)
            @test ha2.current_scale == inv(ha2.qn_du.initial_scale)
            gupd2 = Manopt.GenericSegmentHessianUpdater(similar(d), similar(d))
            Manopt.init_updater!(M, gupd2, p, d, ha2)
            lmupd2 = Manopt.get_default_hessian_segment_updater(M, p, ha2)
            Manopt.init_updater!(M, lmupd2, p, d, ha2)
            @test all(
                lmupd2(M, p, t_current, dt, b, db, ha2) .≈
                    gupd2(M, p, t_current, dt, b, db, ha2)
            )
        end
    end

    @testset "GeneralizedCauchyDirectionSubsolver" begin
        M = Hyperrectangle([-1.0, -2.0, -Inf], [2.0, Inf, 2.0])
        ha = QuasiNewtonMatrixDirectionUpdate(M, BFGS())

        p = [0.0, 0.0, 0.0]
        gf = Manopt.GeneralizedCauchyDirectionSubsolver(M, p, ha)

        X1 = [-5.0, 0.0, 0.0]

        d = -X1
        d_out = similar(d)

        @test Manopt.find_generalized_cauchy_direction!(M, gf, d_out, p, d, X1) === (:found_limited, 1.0)
        @test d_out ≈ [2.0, 0.0, 0.0]

        d_out = similar(d)

        @test Manopt.find_generalized_cauchy_direction!(M, gf, d_out, p, 0 * d, X1) === (:not_found, NaN)

        d2 = [0.0, 1.0, 0.0]

        @test Manopt.find_generalized_cauchy_direction!(M, gf, d_out, p, d2, [0.0, -1.0, 0.0]) === (:found_unlimited, Inf)
        @test d_out ≈ d2

        @test Manopt.find_generalized_cauchy_direction!(M, gf, d_out, p, [1.0, 1.0, 0.0], [-10.0, -10.0, -10.0]) === (:found_limited, 1.0)
        @test d_out ≈ [2.0, 10.0, 0.0]

        p2 = [-1.0, -2.0, 2.0]
        gf2 = Manopt.GeneralizedCauchyDirectionSubsolver(M, p2, ha)

        @test Manopt.find_generalized_cauchy_direction!(M, gf2, d_out, p2, [-1.0, -1.0, 1.0], [-10.0, -10.0, -10.0]) === (:not_found, NaN)

        M2 = Hyperrectangle([-10.0], [10.0])

        ha2 = QuasiNewtonMatrixDirectionUpdate(M2, BFGS(), DefaultOrthonormalBasis(), [100.0;;])
        p3 = [1.0]
        gf3 = Manopt.GeneralizedCauchyDirectionSubsolver(M2, p3, ha2)

        d_out = similar(p3)
        @test Manopt.find_generalized_cauchy_direction!(M2, gf3, d_out, p3, [1.0], [-10.0]) === (:found_limited, 90.0)
    end

    @testset "Zero breakpoints in GCD" begin
        # x1 ≥ 0 is active and d points outwards in x1: that element has to be fixed from the
        # start, otherwise the model is minimized along p + t d and the result is an ascent direction
        M = Hyperrectangle([0.0, -Inf], [Inf, Inf])
        p = [0.0, 0.0]
        X = [10.0, 1.0]
        B = [0.12 -0.2; -0.2 1.0]
        ha = QuasiNewtonMatrixDirectionUpdate(M, BFGS(), DefaultOrthonormalBasis(), inv(B))
        gf = Manopt.GeneralizedCauchyDirectionSubsolver(M, p, ha)
        d = -B * X # descent direction, ⟨X, d⟩ = -9, but its projection (0, 1) is not
        d_out = similar(d)
        @test Manopt.find_generalized_cauchy_direction!(M, gf, d_out, p, d, X) === (:not_found, NaN)
        @test d_out == [0.0, 0.0]
        # aliasing d_out and d, as done by the quasi-Newton direction update
        d_alias = copy(d)
        @test Manopt.find_generalized_cauchy_direction!(M, gf, d_alias, p, d_alias, X) === (:not_found, NaN)
        @test d_alias == [0.0, 0.0]
        # projected gradient direction after a reset
        ha_id = QuasiNewtonMatrixDirectionUpdate(M, BFGS(), DefaultOrthonormalBasis(), [1.0 0.0; 0.0 1.0])
        gf_id = Manopt.GeneralizedCauchyDirectionSubsolver(M, p, ha_id)
        @test Manopt.find_generalized_cauchy_direction!(M, gf_id, d_out, p, -X, X) === (:found_unlimited, Inf)
        @test d_out ≈ [0.0, -1.0]

        # a projected direction that is still a descent direction: the zero-breakpoint element
        # must not influence the step along the remaining elements
        ha2 = QuasiNewtonMatrixDirectionUpdate(M, BFGS(), DefaultOrthonormalBasis(), [1.0 0.5; 0.5 2.0])
        gf2 = Manopt.GeneralizedCauchyDirectionSubsolver(M, p, ha2)
        @test Manopt.find_generalized_cauchy_direction!(M, gf2, d_out, p, [-1.0, -1.0], [1.0, 1.0]) === (:found_unlimited, Inf)
        @test d_out ≈ [0.0, -0.5] # minimizer of t ↦ -t + t^2 along (0, -1)

        # every bounded variable at an active bound with d pointing outwards
        M3 = Hyperrectangle([0.0, 0.0], [1.0, 1.0])
        gf3 = Manopt.GeneralizedCauchyDirectionSubsolver(M3, p, ha_id)
        @test Manopt.find_generalized_cauchy_direction!(M3, gf3, d_out, p, [-1.0, -0.05], [1.0, -0.1]) === (:not_found, NaN)
        @test d_out == [0.0, 0.0]

        # the same with an unbounded manifold factor without an isotropic stepsize limit
        M4 = Hyperrectangle([0.0], [1.0]) × Euclidean(1)
        p4 = ArrayPartition([0.0], [0.0])
        ha4 = QuasiNewtonLimitedMemoryBoxDirectionUpdate(
            QuasiNewtonLimitedMemoryDirectionUpdate(M4, p4, InverseBFGS(), 2)
        )
        gf4 = Manopt.GeneralizedCauchyDirectionSubsolver(M4, p4, ha4)
        d_out4 = zero_vector(M4, p4)
        X4 = ArrayPartition([1.0], [1.0])
        @test Manopt.find_generalized_cauchy_direction!(M4, gf4, d_out4, p4, -X4, X4) === (:found_unlimited, Inf)
        @test d_out4 ≈ ArrayPartition([0.0], [-1.0])
    end

    @testset "Hitting multiple bounds at the same time in GCD" begin
        M = Hyperrectangle([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
        ha = QuasiNewtonMatrixDirectionUpdate(M, BFGS(), DefaultOrthonormalBasis(), [1.0 0 0; 0 1 0; 0 0 1])

        p = [0.0, 0.0, 0.0]
        gf = Manopt.GeneralizedCauchyDirectionSubsolver(M, p, ha)

        d = [-2.0, -2.0, -1.0]
        d_out = similar(d)
        X = [10.0, 10.0, 10.0]

        @test Manopt.find_generalized_cauchy_direction!(M, gf, d_out, p, d, X) === (:found_limited, 1.0)
        @test d_out ≈ [-1.0, -1.0, -1.0]
    end

    @testset "Pure Hyperrectangle" begin
        M = Hyperrectangle([-1.0, 2.0, -Inf], [2.0, Inf, 2.0])
        f(M, p) = sum(p .^ 2)
        function grad_f(M, p)
            return project(M, p, 2 .* p)
        end
        p0 = [0.0, 4.0, 1.0]
        p_opt = quasi_Newton(M, f, grad_f, p0; stopping_criterion = StopWhenProjectedNegativeGradientNormLess(1.0e-6) | StopAfterIteration(10))
        @test p_opt ≈ [0, 2, 0]
        # a preconditioner deactivates `initial_scale`, which the box update has to treat as 1
        for kwargs in ((; preconditioner = (M, p, X) -> X), (; initial_scale = nothing))
            p_opt_p = quasi_Newton(
                M, f, grad_f, p0;
                stopping_criterion = StopWhenProjectedNegativeGradientNormLess(1.0e-6) | StopAfterIteration(10),
                kwargs...,
            )
            @test isapprox(p_opt_p, [0, 2, 0]; atol = 1.0e-3)
        end
        @test Manopt.QuasiNewtonLimitedMemoryBoxDirectionUpdate(
            Manopt.QuasiNewtonLimitedMemoryDirectionUpdate(
                M, p0, InverseBFGS(), 2; initial_scale = nothing
            )
        ).current_scale == 1.0
        # with a non-unit `initial_scale` the two Hessian accessors must agree
        u_s = Manopt.QuasiNewtonLimitedMemoryBoxDirectionUpdate(
            Manopt.QuasiNewtonLimitedMemoryDirectionUpdate(
                M, p0, InverseBFGS(), 2; initial_scale = 2.0
            )
        )
        e1 = Manopt.UnitVector(1)
        @test Manopt.hessian_value_diag(u_s, M, p0, e1) ≈
            Manopt.hessian_value(u_s, M, p0, e1, [1.0, 0.0, 0.0])
        # `cautious_update` wraps the box update; the cautious function `x -> 1e3 * x`
        # rejects every pair, so the memory stays empty
        for (cautious_function, memory_length) in ((x -> 1.0e-4 * x, 2), (x -> 1.0e3 * x, 0))
            st_c = quasi_Newton(
                M, f, grad_f, p0;
                cautious_update = true, cautious_function = cautious_function, return_state = true,
                stopping_criterion = StopWhenProjectedNegativeGradientNormLess(1.0e-6) | StopAfterIteration(10),
            )
            du_c = Manopt.get_state(st_c).direction_update
            @test du_c isa QuasiNewtonCautiousDirectionUpdate{<:QuasiNewtonLimitedMemoryBoxDirectionUpdate}
            @test length(du_c.update.qn_du.memory_s) == memory_length
            @test get_solver_result(st_c) ≈ [0, 2, 0]
        end


        f2(M, p) = sum(p .^ 4)
        function grad_f2(M, p)
            return project(M, p, 4 .* (p .^ 3))
        end
        p0 = [0.0, 4.0, 1.0]
        p_opt = quasi_Newton(M, f2, grad_f2, p0; stopping_criterion = StopWhenProjectedNegativeGradientNormLess(1.0e-6) | StopAfterIteration(100))
        @test f2(M, p_opt) < 16.1

        for stepsize in [ArmijoLinesearch(), CubicBracketingLinesearch(), NonmonotoneLinesearch()]
            p_opt = quasi_Newton(
                M, f2, grad_f2, p0;
                stopping_criterion = StopWhenProjectedNegativeGradientNormLess(1.0e-6) | StopAfterIteration(100),
                stepsize = stepsize
            )
            @test f2(M, p_opt) < 64.0
        end

        MInf = Hyperrectangle([-Inf, -Inf, -Inf], [Inf, Inf, Inf])

        f3(M, p) = sum(p .^ 4) - sum(p .^ 2)
        function grad_f3(M, p)
            return project(MInf, p, 4 .* (p .^ 3) - 2 .* p)
        end
        p0 = [0.0, 4.0, 1.0]
        p_opt = quasi_Newton(MInf, f3, grad_f3, p0; stopping_criterion = StopWhenProjectedNegativeGradientNormLess(1.0e-6) | StopAfterIteration(100))
        @test f3(MInf, p_opt) < 16.1
    end

    @testset "project_truncate" begin
        M = Hyperrectangle([1.0, 1.0], [2.0, 2.0])
        # at the lower bound with X pointing out of the box, and in the interior
        @test Manopt.project_truncate(M, [1.0, 1.5], [-2.0, -0.1]) ≈ [0.0, -0.1]
        # the step to P(p + X) is cut at the bounds
        @test Manopt.project_truncate(M, [1.5, 1.5], [1.0, -0.2]) ≈ [0.5, -0.2]
        # close to the bound, the result is small although X points into the box
        Y = Manopt.project_truncate(M, [1.0 + 1.0e-9, 1.5], [-2.0, 0.0])
        @test isapprox(Y, [-1.0e-9, 0.0]; atol = 1.0e-15)
        # in place of X
        Y = [-2.0, -0.1]
        Manopt.project_truncate!(M, Y, [1.0, 1.5], Y)
        @test Y ≈ [0.0, -0.1]
        N = ProductManifold(M, Sphere(2))
        p = ArrayPartition([1.0, 1.5], [1.0, 0.0, 0.0])
        X = ArrayPartition([-2.0, -0.1], [0.0, -1.0, 2.0])
        Y = Manopt.project_truncate(N, p, X)
        @test Y.x[1] ≈ [0.0, -0.1]
        @test Y.x[2] ≈ [0.0, -1.0, 2.0]
        # without box constraints, X is only projected onto the tangent space
        @test Manopt.project_truncate(Sphere(2), [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) ≈
            [0.0, 1.0, 0.0]
        @test Manopt.project_truncate(Sphere(2), [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]) ≈
            [0.0, 1.0, 0.0]
        @test Manopt.isotropic_max_stepsize(M, [1.5, 1.5]) == Inf
        @test Manopt.isotropic_max_stepsize(N, p) ≈ Manopt.max_stepsize(Sphere(2), p.x[2])
    end

    @testset "zero_at_active_bounds!" begin
        M = Hyperrectangle([1.0, 1.0, 1.0, 2.0], [2.0, 2.0, 2.0, 2.0])
        p = [1.0, 2.0, 1.5, 2.0]
        # held at the lower bound with X > 0, at the upper bound with X < 0 and the fixed
        # variable; the gradient points into the box at the bounds or p is in the interior
        X = [3.0, -1.0, 4.0, -5.0]
        Y = [1.0, 2.0, 3.0, 4.0]
        @test Manopt.zero_at_active_bounds!(M, Y, p, X) === Y
        @test Y == [0.0, 0.0, 3.0, 0.0]
        X = [-3.0, 1.0, -4.0, 5.0]
        Y = [1.0, 2.0, 3.0, 4.0]
        Manopt.zero_at_active_bounds!(M, Y, p, X)
        @test Y == [1.0, 2.0, 3.0, 0.0]
        # in place of X
        X = [3.0, -1.0, 4.0, -5.0]
        Manopt.zero_at_active_bounds!(M, X, p, X)
        @test X == [0.0, 0.0, 4.0, 0.0]
        N = ProductManifold(M, Sphere(2))
        pN = ArrayPartition(p, [1.0, 0.0, 0.0])
        XN = ArrayPartition([3.0, -1.0, 4.0, -5.0], [0.0, -1.0, 2.0])
        YN = ArrayPartition([1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 1.0])
        Manopt.zero_at_active_bounds!(N, YN, pN, XN)
        @test YN.x[1] == [0.0, 0.0, 3.0, 0.0]
        @test YN.x[2] == [0.0, 1.0, 1.0]
        # without box constraints, Y is not modified
        Y = [0.0, 1.0, 0.0]
        @test Manopt.zero_at_active_bounds!(Sphere(2), Y, [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]) == [0.0, 1.0, 0.0]
    end

    @testset "Large gradient of a fixed variable" begin
        # the last variable is fixed and its gradient element is large at the minimizer;
        # it must not leak into the direction of the free variables
        M = Hyperrectangle([-Inf, -Inf, -Inf, 2.0], [Inf, Inf, Inf, 2.0])
        A = [4.0 1.0 0.5 2.0; 1.0 3.0 1.0 -1.5; 0.5 1.0 2.0 1.0; 2.0 -1.5 1.0 5.0]
        b = [1.0, -2.0, 0.5, -20.0]
        f(M, p) = 0.5 * dot(p, A * p) - dot(b, p)
        grad_f(M, p) = A * p - b
        p_opt = quasi_Newton(
            M, f, grad_f, [0.0, 0.0, 0.0, 2.0];
            memory_size = 5,
            stopping_criterion = StopWhenProjectedNegativeGradientNormLess(1.0e-10) |
                StopAfterIteration(30),
        )
        p_star = [A[1:3, 1:3] \ (b[1:3] - 2.0 * A[1:3, 4]); 2.0]
        @test p_opt ≈ p_star atol = 1.0e-8
    end

    @testset "Sufficient model decrease of the generalized Cauchy direction" begin
        M = Hyperrectangle([-1.0, -1.0, -1.0], [1.0, 2.0, 2.0])
        A = [4.0 1.0 0.0; 1.0 3.0 1.0; 0.0 1.0 2.0]
        b = [1.0, -8.0, 3.0]
        f(M, p) = 0.5 * dot(p, A * p) - dot(b, p)
        grad_f(M, p) = A * p - b
        p0 = [0.5, 0.5, 0.5]
        pg_norm(p) = norm(Manopt.project_truncate(M, p, -grad_f(M, p)))

        @test_throws DomainError Manopt.QuasiNewtonLimitedMemoryBoxDirectionUpdate(
            Manopt.QuasiNewtonLimitedMemoryDirectionUpdate(M, p0, InverseBFGS(), 2);
            model_decrease_factor = -1.0,
        )

        # the bound on the norm of the Hessian approximation and the reference value
        s = quasi_Newton(
            M, f, grad_f, p0; memory_size = 3, nonpositive_curvature_behavior = :byrd,
            stopping_criterion = StopAfterIteration(2), return_state = true,
        )
        qns = get_state(s)
        du = qns.direction_update
        p = get_iterate(qns)
        X = grad_f(M, p)
        @test any(!iszero, du.qn_du.ρ)
        E = [[i == j ? 1.0 : 0.0 for i in 1:3] for j in 1:3]
        H = [Manopt.hessian_value(du, M, p, Manopt.UnitVector(i), E[j]) for i in 1:3, j in 1:3]
        λ = Manopt.hessian_norm_bound(du, M, p)
        @test maximum(eigvals(Symmetric(H))) <= λ * (1 + 1.0e-12)
        h = pg_norm(p)
        @test Manopt.model_decrease_reference(du, M, p, X, zero_vector(M, p)) ≈
            h^2 / norm(X) * h / max(1.0, λ)

        # a huge factor rejects every direction computed with curvature information:
        # the memory is then reset in each iteration and the solver still converges
        for (κ, rejected) in ((1.0e8, true), (0.0, false))
            s = quasi_Newton(
                M, f, grad_f, p0; memory_size = 3, nonpositive_curvature_behavior = :byrd,
                model_decrease_factor = κ, return_state = true,
                stopping_criterion = StopWhenProjectedNegativeGradientNormLess(1.0e-9) |
                    StopAfterIteration(200),
            )
            qns = get_state(s)
            @test pg_norm(get_iterate(qns)) < 1.0e-9
            @test (qns.direction_update.insufficient_decrease_count > 0) == rejected
            @test qns.direction_update.model_decrease_factor == κ
        end
        @test contains(
            Manopt.status_summary(
                Manopt.QuasiNewtonLimitedMemoryBoxDirectionUpdate(
                    Manopt.QuasiNewtonLimitedMemoryDirectionUpdate(M, p0, InverseBFGS(), 2)
                )
            ),
            "model decrease factor",
        )
    end

    @testset "Quasi-Newton direction restricted to the variables not held at their bounds" begin
        # D = [0, ∞) and the Euclidean ℝ² at p = 0 with gradient (1; 1, 0), so the first
        # variable is held at its bound. The first entry of -B g points out of the box, and
        # projecting only -B g onto the tangent cone would leave a descent direction with a
        # model decrease of only 3/2 η² + O(η³), although the projected gradient has norm 1,
        # even for the exact inverse Hessian B of a quadratic. Restricting the gradient and
        # -B g to the free variables avoids the leak of the gradient entry of the held variable,
        # and the model decrease is of order 1 for all η. A large `model_decrease_factor` still
        # rejects the direction.
        M = Hyperrectangle([0.0], [Inf]) × Euclidean(2)
        p = ArrayPartition([0.0], [0.0, 0.0])
        g = [1.0, 1.0, 0.0]
        to_tangent(v) = ArrayPartition(v[1:1], v[2:3])
        for (η, κ, rejected) in ((1.0e-3, 1.0e-4, false), (1.0e-3, 0.0, false), (1.0e-1, 1.0e-4, false), (1.0e-3, 10.0, true))
            B = [2.0 -1.0 -0.5; -1.0 (1 + η) 0.0; -0.5 0.0 1.0]
            H = inv(B)
            f(M, q) = dot(g, vcat(q.x...)) + dot(vcat(q.x...), H * vcat(q.x...)) / 2
            grad_f(M, q) = to_tangent(g + H * vcat(q.x...))
            mp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
            ha = QuasiNewtonLimitedMemoryBoxDirectionUpdate(
                QuasiNewtonLimitedMemoryDirectionUpdate(M, p, InverseBFGS(), 3);
                model_decrease_factor = κ,
            )
            st = QuasiNewtonState(
                M; p = copy(M, p), X = to_tangent(g), direction_update = ha,
                stepsize = Manopt.ArmijoLinesearchStepsize(
                    M; initial_guess = Manopt.ConstantInitialGuess(1.0), stop_increasing_at_step = 0,
                ),
            )
            # the eigenpairs of H are H-conjugate, so the limited memory reproduces H exactly
            F = eigen(Symmetric(H))
            for i in 1:3
                st.sk = to_tangent(F.vectors[:, i])
                st.yk = to_tangent(H * F.vectors[:, i])
                update_hessian!(ha, mp, st, p, i)
            end
            r = zero_vector(M, p)
            ha.qn_du(r, mp, st)
            @test r ≈ to_tangent(-B * g)
            d = ha(mp, st)
            @test ha.last_gcd_result === (rejected ? :insufficient_decrease : :found_unlimited)
            @test ha.insufficient_decrease_count == rejected
            if !rejected
                # the direction stays on the bound, only moves the second variable, is a
                # descent direction and its model decrease is of order 1
                @test d.x[1] == [0.0]
                @test isapprox(d.x[2][2], 0.0; atol = 1.0e-12)
                @test inner(M, p, to_tangent(g), d) < 0
                model_decrease = -(inner(M, p, to_tangent(g), d) + Manopt.hessian_value_diag(ha, M, p, d) / 2)
                @test model_decrease > 0.2
            end
            # one solver step: a rejected direction resets the memory, and the step follows
            # the generalized Cauchy point (0; -1, 0) for the identity
            step_solver!(mp, st, 1)
            if rejected
                @test ha.insufficient_decrease_count == 2
                @test length(ha.qn_du.memory_s) == 1
                @test st.p.x[1] == [0.0] && st.p.x[2][1] < 0 && st.p.x[2][2] == 0
            end
            @test f(M, p) - f(M, st.p) > 0.1
        end
    end

    @testset "Default stopping criterion with box constraints" begin
        M = Hyperrectangle([0.0, 0.0], [1.0, 1.0])
        f(M, p) = sum((p .- 2) .^ 2)
        grad_f(M, p) = 2 .* (p .- 2)
        # the gradient (-2, -2) does not vanish at the solution (1, 1)
        s = quasi_Newton(M, f, grad_f, [0.5, 0.5]; return_state = true)
        sc = get_state(s).stop
        @test any(c -> c isa StopWhenProjectedNegativeGradientNormLess, sc.criteria)
        @test !any(c -> c isa StopWhenGradientNormLess, sc.criteria)
        @test get_count(get_state(s), :Iterations) < 10
        @test get_solver_result(s) ≈ [1.0, 1.0]
        s = quasi_Newton(M, f, grad_f, [0.5, 0.5]; linesearch_failure_behavior = :restart, return_state = true)
        @test any(c -> c isa StopWhenProjectedNegativeGradientNormLess, get_state(s).stop.criteria)
        @test any(c -> c isa StopWhenProjectedNegativeGradientNormLess, QuasiNewtonState(M; p = [0.5, 0.5]).stop.criteria)
        @test any(c -> c isa StopWhenGradientNormLess, QuasiNewtonState(Euclidean(2); p = [0.5, 0.5]).stop.criteria)
    end

    @testset "has_anisotropic_max_stepsize" begin
        @test !Manopt.has_anisotropic_max_stepsize(Sphere(2))
        @test Manopt.has_anisotropic_max_stepsize(Hyperrectangle([1], [2]))
        @test Manopt.has_anisotropic_max_stepsize(ProductManifold(Hyperrectangle([1], [2]), Sphere(2)))
    end

    @testset "Hyperrectangle × Sphere" begin
        S2 = Sphere(2)
        px = [0.0, 1.0, 0.0]
        Mbox = Hyperrectangle([-1.0, 2.0, -Inf], [2.0, Inf, 2.0])
        M = Mbox × S2
        f(M, p) = sum(p.x[1] .^ 4) + 0.5 * distance(S2, p.x[2], px)^2
        grad_f(M, p) = ArrayPartition(project(Mbox, p.x[1], 4 .* (p.x[1] .^ 3)), -log(S2, p.x[2], px))
        p0 = ArrayPartition([0.0, 4.0, 1.0], [1.0, 0.0, 0.0])

        @testset "Hessian updater" begin
            d = -grad_f(M, p0)
            ha = QuasiNewtonMatrixDirectionUpdate(M, BFGS(), DefaultOrthonormalBasis())
            gupd = Manopt.GenericSegmentHessianUpdater(similar(d), similar(d))
            Manopt.init_updater!(M, gupd, p0, d, ha)
            b = (1, 2)
            dt = 0.25
            t_current = 0 + dt
            db = d.x[b[1]][b[2]]
            hv_eb_dz, hv_eb_d = gupd(M, p0, t_current, dt, b, db, ha)
            @test hv_eb_dz ≈ -64.0
            @test hv_eb_d ≈ -256.0
        end

        @testset "GCD check" begin
            d = -grad_f(M, p0)
            ha = QuasiNewtonLimitedMemoryBoxDirectionUpdate(QuasiNewtonLimitedMemoryDirectionUpdate(M, p0, InverseBFGS(), 2))
            gf = Manopt.GeneralizedCauchyDirectionSubsolver(M, p0, ha)
            d_out = similar(d)
            X = grad_f(M, p0)
            @test Manopt.find_generalized_cauchy_direction!(M, gf, d_out, p0, d, X) === (:found_limited, 1.0)
        end

        p_opt = quasi_Newton(M, f, grad_f, p0; stopping_criterion = StopWhenProjectedNegativeGradientNormLess(1.0e-6) | StopAfterIteration(100))
        @test distance(M, p_opt, ArrayPartition([0, 2, 0], px)) < 0.1
    end

    @testset "Isotropic limit in the generalized Cauchy direction" begin
        S2 = Sphere(2)
        M = Hyperrectangle([-100.0], [100.0]) × S2
        p = ArrayPartition([0.0], [1.0, 0.0, 0.0])
        d = ArrayPartition([-1.0], [0.0, 10.0, 0.0])
        X = -d
        ha = QuasiNewtonLimitedMemoryBoxDirectionUpdate(
            QuasiNewtonLimitedMemoryDirectionUpdate(M, p, InverseBFGS(), 2)
        )
        gf = Manopt.GeneralizedCauchyDirectionSubsolver(M, p, ha)
        d_out = similar(d)
        @test Manopt.find_generalized_cauchy_direction!(M, gf, d_out, p, d, X) ===
            (:found_limited, 1.0)
        # the direction may not leave the isotropic maximal step size of the sphere factor
        @test norm(S2, p.x[2], d_out.x[2]) <= Manopt.max_stepsize(S2, p.x[2]) + 1.0e-12
    end

    @testset "Sphere × Hyperrectangle" begin
        S2 = Sphere(2)
        px = [0.0, 1.0, 0.0]
        Mbox = Hyperrectangle([-1.0 2.0; -Inf -Inf], [2.0 Inf; 2.0 Inf])
        M = S2 × Mbox
        f(M, p) = sum(p.x[2] .^ 4) + 0.5 * distance(S2, p.x[1], px)^2
        grad_f(M, p) = ArrayPartition(-log(S2, p.x[1], px), project(Mbox, p.x[2], 4 .* (p.x[2] .^ 3)))
        p0 = ArrayPartition([1.0, 0.0, 0.0], [0.0 4.0; 1.0 1.0])

        p_opt = quasi_Newton(M, f, grad_f, p0; stopping_criterion = StopWhenProjectedNegativeGradientNormLess(1.0e-6) | StopAfterIteration(100))
        @test distance(M, p_opt, ArrayPartition(px, [0 2; 0 0])) < 0.1
        p_opt_c = quasi_Newton(
            M, f, grad_f, p0; cautious_update = true,
            stopping_criterion = StopWhenProjectedNegativeGradientNormLess(1.0e-6) | StopAfterIteration(100),
        )
        @test is_point(M, p_opt_c)
        @test distance(M, p_opt_c, ArrayPartition(px, [0 2; 0 0])) < 0.1
    end

    @testset "Cautious update with box constraints" begin
        M = Hyperrectangle([-1.0, 2.0, -Inf], [2.0, Inf, 2.0])
        f(M, p) = sum(p .^ 2)
        grad_f(M, p) = project(M, p, 2 .* p)
        p0 = [0.0, 4.0, 1.0]
        mp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
        ha = QuasiNewtonLimitedMemoryBoxDirectionUpdate(
            QuasiNewtonLimitedMemoryDirectionUpdate(M, p0, InverseBFGS(), 2)
        )
        cha = QuasiNewtonCautiousDirectionUpdate(ha; θ = identity)
        @test startswith(
            repr(cha),
            "QuasiNewtonCautiousDirectionUpdate with θ = identity and internal state:\nQuasiNewtonLimitedMemoryBoxDirectionUpdate",
        )
        st = QuasiNewtonState(M; p = copy(M, p0), direction_update = cha)
        # with θ = identity the bound is the projected gradient norm at p0 with gradient
        # (0, 8, 2): ‖P(p0 - X) - p0‖ = ‖(0, -2, -2)‖ ≈ 2.83 (the gradient norm is ≈ 8.25)
        @test Manopt._projected_gradient_norm(M, p0, grad_f(M, p0)) ≈ sqrt(8)
        # ⟨s, y⟩ / ‖s‖² = 10 exceeds the bound, so the pair is added
        st.sk = [0.0, 1.0, 0.0]
        st.yk = [0.0, 10.0, 0.0]
        update_hessian!(cha, mp, st, p0, 1)
        @test length(ha.qn_du.memory_s) == 1
        @test ha.current_scale ≈ 10.0
        # ⟨s, y⟩ / ‖s‖² = 1 is below the bound, so the pair is skipped, but the compact
        # representation is recomputed from the stored pairs, here a modified one
        st.sk = [1.0, 0.0, 0.0]
        st.yk = [1.0, 0.0, 0.0]
        ha.qn_du.memory_y[1] .*= 2
        update_hessian!(cha, mp, st, p0, 2)
        @test length(ha.qn_du.memory_s) == 1
        @test ha.qn_du.ρ[1] ≈ 1 / 20
        @test ha.current_scale ≈ 20.0
        # the parallel transport on the `Hyperrectangle` is isometric, so the re-test is off by
        # default; with it, a stored pair that violates the decision rule after the transport,
        # here with ⟨s, y⟩ / ‖s‖² = 2, is removed, both when the new pair is added …
        @test !cha.retest
        cha.retest = true
        ha.qn_du.memory_y[1] .= [0.0, 2.0, 0.0]
        st.sk = [0.0, 1.0, 0.0]
        st.yk = [0.0, 10.0, 0.0]
        update_hessian!(cha, mp, st, p0, 3)
        @test length(ha.qn_du.memory_s) == 1
        @test ha.qn_du.memory_y[1] == [0.0, 10.0, 0.0]
        @test ha.current_scale ≈ 10.0
        # … and when it is skipped
        ha.qn_du.memory_y[1] .= [0.0, 2.0, 0.0]
        st.sk = [1.0, 0.0, 0.0]
        st.yk = [1.0, 0.0, 0.0]
        update_hessian!(cha, mp, st, p0, 4)
        @test length(ha.qn_du.memory_s) == 0
        @test ha.current_scale ≈ inv(Manopt._box_initial_scale(ha))
        # the direction is the one of the box update, and so is its maximal step size
        st.X = grad_f(M, p0)
        η = cha(mp, st)
        @test η ≈ ha(mp, st)
        @test Manopt.get_parameter(cha, :max_stepsize) == Manopt.get_parameter(ha, :max_stepsize)
        ha.last_gcd_result = :found_limited
        ha.last_gcd_stepsize = 0.5
        @test Manopt.get_parameter(cha, :max_stepsize) == 0.5
        # at a stationary point with an active bound, the gradient (0, 4, 0) at (0, 2, 0) does
        # not vanish, but the projected gradient does, so a pair with a small curvature is added
        Manopt.initialize_update!(cha)
        q = [0.0, 2.0, 0.0]
        st.p = copy(M, q)
        st.X = grad_f(M, q)
        cha(mp, st)
        @test cha.projected_gradient_norm == 0.0
        st.sk = [1.0, 0.0, 0.0]
        st.yk = [0.5, 0.0, 0.0]
        update_hessian!(cha, mp, st, q, 3)
        @test length(ha.qn_du.memory_s) == 1
        # without a recorded norm, it is computed from the gradient at the previous iterate
        Manopt.initialize_update!(cha)
        update_hessian!(cha, mp, st, q, 4)
        @test length(ha.qn_du.memory_s) == 1
        # on a product with box constraints, only the box component is projected
        N = M × Euclidean(2)
        pN = ArrayPartition(q, [0.0, 0.0])
        XN = ArrayPartition(grad_f(M, q), [3.0, 4.0])
        @test Manopt._projected_gradient_norm(N, pN, XN) ≈ 5.0
        @test Manopt._projected_gradient_norm(Euclidean(2), [0.0, 0.0], [3.0, 4.0]) == 5.0
    end
end

@testset "MaxStepsizeInDirection" begin
    @testset "found_limited" begin
        M = Hyperrectangle([-1.0, -2.0, -Inf], [2.0, Inf, 2.0])
        p = [0.0, 0.0, 0.0]
        d = [2.0, 1.0, 1.0]
        d_before = copy(d)

        sdf = Manopt.MaxStepsizeInDirectionSubsolver(M, p)
        @test Manopt.find_max_stepsize_in_direction(M, sdf, p, d) === (:found_limited, 1.0)
        @test d == d_before
    end

    @testset "found_unlimited" begin
        M = Hyperrectangle([-Inf], [Inf])
        p = [0.0]
        d = [1.0]
        d_before = copy(d)

        sdf = Manopt.MaxStepsizeInDirectionSubsolver(M, p)
        @test Manopt.find_max_stepsize_in_direction(M, sdf, p, d) === (:found_unlimited, Inf)
        @test d == d_before
    end

    @testset "not_found" begin
        M = Hyperrectangle([0.0], [1.0])
        p = [0.0]
        d = [-1.0]
        d_before = copy(d)

        sdf = Manopt.MaxStepsizeInDirectionSubsolver(M, p)
        @test Manopt.find_max_stepsize_in_direction(M, sdf, p, d) === (:not_found, NaN)
        @test d == d_before
    end

    @testset "Vanishing step at the minimizer" begin
        # the minimizer is reached before the stopping criterion fires, so the step becomes zero
        N = Hyperrectangle(-ones(3), ones(3))
        c = [0.3, 0.5, 0.2]
        fq(M, p) = 0.5 * sum(abs2, p - c)
        grad_fq(M, p) = p - c
        for memory_size in [3, -1]
            q = quasi_Newton(
                N, fq, grad_fq, zeros(3);
                memory_size = memory_size, stopping_criterion = StopAfterIteration(6),
            )
            @test isapprox(N, q, c)
        end
    end
end
