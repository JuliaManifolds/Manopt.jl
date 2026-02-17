using Manopt, Manifolds, Test
using LinearAlgebra: I, eigvecs, tr, Diagonal, dot

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


    @testset "update_fp_fpp - basic d = -g" begin
        M = Hyperrectangle([0.0, 1.0], [3.0, 3.0])

        grad = [1.0, 4.0]
        d = [-1.0, -4.0]
        p = [0.0, 0.0]

        # values taken from loop iteration found in test case: "find_gcp! - with bounds, single variable is held fixed"
        old_f_prime = -17.0
        old_f_double_prime = 34.0
        dt = 0.25
        gb = 4.0
        db = -4.0 # in case of d = -g, db = -gb
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


    @testset "update_fp_fpp - basic d = [-2.0, -1.0]" begin
        M = Hyperrectangle([0.0, 1.0], [3.0, 3.0])

        grad = [1.0, 4.0]
        d = [-2.0, -1.0]
        p = [0.0, 0.0]

        old_f_prime = -6.0
        old_f_double_prime = 10.0
        dt = 0.25
        gb = 1.0
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

    @testset "update_fp_fpp - basic d = [-2.0, -1.0] with limited memory update" begin
        M = Hyperrectangle([1.0, 4.0], [2.0, 10.0])

        p = [2.0, 5.0]
        ha = QuasiNewtonLimitedMemoryBoxDirectionUpdate(QuasiNewtonLimitedMemoryDirectionUpdate(M, p, InverseBFGS(), 2))
        st = QuasiNewtonState(M)

        @test startswith(repr(ha), "QuasiNewtonLimitedMemoryBoxDirectionUpdate with internal state:")

        f(M, p) = sum(p .^ 2)
        grad_f(M, p) = 2 * p
        gmp = ManifoldGradientObjective(f, grad_f)
        mp = DefaultManoptProblem(M, gmp)

        st.yk = [2.0, 4.0]
        st.sk = [4.0, 2.0]
        update_hessian!(ha, mp, st, p, 1)
        grad = grad_f(M, p)
        st.p = p
        st.X = grad

        d = similar(grad)
        ha(d, mp, st)

        d2 = ha(mp, st)
        @test d ≈ d2

        b = 1

        old_f_prime = -6.0
        old_f_double_prime = 10.0
        dt = 0.25
        db = d[b]
        gb = grad[b]

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
            idx = Manopt.get_bounds_index(M)
            @test Manopt.hessian_value(ha2, M, p, Manopt.UnitVector(b), grad) ≈ 4.0
            Manopt.set_M_current_scale!(M, p, ha2)
            @test ha2.current_scale == ha2.qn_du.initial_scale
            @test ha2.M_11 == fill(0.0, 0, 0)
            @test ha2.M_21 == fill(0.0, 0, 0)
            @test ha2.M_22 == fill(0.0, 0, 0)
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

        p_opt = quasi_Newton(
            MInf, f3, grad_f3, p0;
            stopping_criterion = StopWhenProjectedNegativeGradientNormLess(1.0e-6) | StopAfterIteration(100),
        )
        @test f3(MInf, p_opt) < 64.0
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
end
