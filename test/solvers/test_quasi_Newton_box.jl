using Manopt, Manifolds, Test
using LinearAlgebra: I, eigvecs, tr, Diagonal, dot

using RecursiveArrayTools

@testset "Riemannian quasi-Newton Methods with box-like domains" begin
    @testset "get_bound_t - basic" begin
        M = Hyperrectangle([0.0, 0.0], [2.0, 2.0])

        # d[i] > 0
        p = [0.0, 1.0]; d = [1.0, 1.0]
        @test Manopt.get_bound_t(M, p, d, 1) ≈ (2.0 - 0.0) / 1.0  # = 2.0
        @test Manopt.get_bound_t(M, p, d, 2) ≈ (2.0 - 1.0) / 1.0  # = 1.0

        # d[i] < 0
        p = [0.0, 1.0]; d = [-1.0, -1.0]
        @test Manopt.get_bound_t(M, p, d, 1) ≈ (0.0 - 0.0) / -1.0  # = 0.0
        @test Manopt.get_bound_t(M, p, d, 2) ≈ (0.0 - 1.0) / -1.0  # = 1.0

        # d[i] = 0
        p = [0.0, 1.0]; d = [0.0, 0.0]
        @test Manopt.get_bound_t(M, p, d, 1) ≈ Inf
        @test Manopt.get_bound_t(M, p, d, 2) ≈ Inf
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
        d_old = [-1.0, -4.0]

        d[2] = 0.0

        # optimized formula
        f_prime, f_double_prime = Manopt.GenericFPFPPUpdater()(M, p, old_f_prime, old_f_double_prime, dt, db, gb, ha, b, z, d_old)
        @test f_prime ≈ -0.5
        @test f_double_prime ≈ 2.0

        # original formula
        f_original_prime = dot(grad, d) + dot(d, ha.matrix, z)
        f_original_double_prime = Manopt.hess_val(ha, M, p, d)

        @test f_prime == f_original_prime
        @test f_double_prime == f_original_double_prime
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
        d_old = [-2.0, -1.0]

        d[1] = 0.0

        # optimized formula
        f_prime, f_double_prime = Manopt.GenericFPFPPUpdater()(M, p, old_f_prime, old_f_double_prime, dt, db, gb, ha, b, z, d_old)
        @test f_prime == -3.5
        @test f_double_prime == 2

        # original formula
        f_original_prime = dot(grad, d) + dot(d, ha.matrix, z)
        f_original_double_prime = Manopt.hess_val(ha, M, p, d)

        @test f_prime == f_original_prime
        @test f_double_prime == f_original_double_prime
    end

    @testset "update_fp_fpp - basic d = [-2.0, -1.0] with limited memory update" begin
        M = Hyperrectangle([1.0, 4.0], [2.0, 10.0])

        p = [2.0, 5.0]
        ha = QuasiNewtonLimitedMemoryBoxDirectionUpdate(QuasiNewtonLimitedMemoryDirectionUpdate(M, p, InverseBFGS(), 2))
        st = QuasiNewtonState(M)

        f(M, p) = sum(p .^ 2)
        grad_f(M, p) = 2 * p
        gmp = ManifoldGradientObjective(f, grad_f)
        mp = DefaultManoptProblem(M, gmp)

        st.yk = [2.0, 4.0]
        st.sk = [4.0, 2.0]
        update_hessian!(ha, mp, st, p, 1)
        grad = grad_f(M, p)

        d = similar(grad)
        ha(d, mp, st)

        d2 = ha(mp, st)
        @test d ≈ d2

        b = 1
        z = [-0.5, -0.25]
        d_old = [-2.0, -1.0]

        d[1] = 0.0

        old_f_prime = -6.0
        old_f_double_prime = 10.0
        dt = 0.25
        db = d[b]
        gb = grad[b]

        # compare the generic and limited memory updater
        f_prime, f_double_prime = Manopt.GenericFPFPPUpdater()(M, p, old_f_prime, old_f_double_prime, dt, db, gb, ha, b, z, d_old)
        @test f_prime == -3.5
        @test f_double_prime == 10

        lmupd = Manopt.get_default_fpfpp_updater(ha)
        @test lmupd isa Manopt.LimitedMemoryFPFPPUpdater

        Manopt.init_updater!(M, lmupd, p, d, ha)
        f_prime_limited, f_double_prime_limited = lmupd(M, p, old_f_prime, old_f_double_prime, dt, db, gb, ha, b, z, d_old)

        @test f_prime ≈ f_prime_limited
        @test f_double_prime ≈ f_double_prime_limited

        ha.last_gcp_result = :found_unlimited
        @test Manopt.get_parameter(ha, Val(:max_stepsize)) == Inf

        @testset "No memory tests" begin
            ha2 = QuasiNewtonLimitedMemoryBoxDirectionUpdate(QuasiNewtonLimitedMemoryDirectionUpdate(M, p, InverseBFGS(), 2))
            @test Manopt.hess_val_eb(ha2, M, p, b, grad) ≈ 4.0
            Manopt.set_M_current_scale!(M, p, ha2)
            @test ha2.current_scale == ha2.qn_du.initial_scale
            @test ha2.M_11 == fill(0.0, 0, 0)
            @test ha2.M_21 == fill(0.0, 0, 0)
            @test ha2.M_22 == fill(0.0, 0, 0)
        end
    end

    @testset "GeneralizedCauchyPointFinder" begin
        M = Hyperrectangle([-1.0, -2.0, -Inf], [2.0, Inf, 2.0])
        ha = QuasiNewtonMatrixDirectionUpdate(M, BFGS())

        p = [0.0, 0.0, 0.0]
        gf = Manopt.GeneralizedCauchyPointFinder(M, p, ha)

        X1 = [-5.0, 0.0, 0.0]

        d = -X1
        d_out = similar(d)

        @test Manopt.find_generalized_cauchy_point_direction!(gf, d_out, p, d, X1) === :found_limited
        @test d_out ≈ [2.0, 0.0, 0.0]

        d2 = [0.0, 1.0, 0.0]

        @test Manopt.find_generalized_cauchy_point_direction!(gf, d_out, p, d2, [0.0, -1.0, 0.0]) === :found_unlimited
        @test d_out ≈ d2

        @test Manopt.find_generalized_cauchy_point_direction!(gf, d_out, p, [1.0, 1.0, 0.0], [-10.0, -10.0, -10.0]) === :found_limited
        @test d_out ≈ [2.0, 10.0, 0.0]

        p2 = [-1.0, -2.0, 2.0]
        gf2 = Manopt.GeneralizedCauchyPointFinder(M, p2, ha)

        @test Manopt.find_generalized_cauchy_point_direction!(gf2, d_out, p2, [-1.0, -1.0, 1.0], [-10.0, -10.0, -10.0]) === :not_found

        M2 = Hyperrectangle([-10.0], [10.0])

        ha2 = QuasiNewtonMatrixDirectionUpdate(M2, BFGS(), DefaultOrthonormalBasis(), [100.0;;])
        p3 = [1.0]
        gf3 = Manopt.GeneralizedCauchyPointFinder(M2, p3, ha2)

        d_out = similar(p3)
        @test Manopt.find_generalized_cauchy_point_direction!(gf3, d_out, p3, [1.0], [-10.0]) === :found_limited
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

    @testset "requires_gcp" begin
        @test !Manopt.requires_gcp(Sphere(2))
        @test Manopt.requires_gcp(Hyperrectangle([1], [2]))
        @test Manopt.requires_gcp(ProductManifold(Hyperrectangle([1], [2]), Sphere(2)))
    end

    @testset "Hyperrectangle × Sphere" begin
        S2 = Sphere(2)
        px = [0.0, 1.0, 0.0]
        Mbox = Hyperrectangle([-1.0, 2.0, -Inf], [2.0, Inf, 2.0])
        M = Mbox × S2
        f(M, p) = sum(p.x[1] .^ 4) + 0.5 * distance(S2, p.x[2], px)^2
        grad_f(M, p) = ArrayPartition(project(Mbox, p.x[1], 4 .* (p.x[1] .^ 3)), -log(S2, p.x[2], px))
        p0 = ArrayPartition([0.0, 4.0, 1.0], [1.0, 0.0, 0.0])

        p_opt = quasi_Newton(M, f, grad_f, p0; stopping_criterion = StopWhenProjectedNegativeGradientNormLess(1.0e-6) | StopAfterIteration(100))
        @test distance(M, p_opt, ArrayPartition([0, 2, 0], px)) < 0.1
    end
end
