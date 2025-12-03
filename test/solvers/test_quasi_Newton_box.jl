using Manopt, Manifolds, Test
using LinearAlgebra: I, eigvecs, tr, Diagonal, dot

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

    @testset "GCPFinder" begin
        M = Hyperrectangle([-1.0, -2.0, -Inf], [2.0, Inf, 2.0])
        ha = QuasiNewtonMatrixDirectionUpdate(M, BFGS())

        p = [0.0, 0.0, 0.0]
        gf = Manopt.GCPFinder(M, p, ha)

        X1 = [-5.0, 0.0, 0.0]

        d = -X1
        d_out = similar(d)

        @test Manopt.find_gcp_direction!(gf, d_out, p, d, X1) === :found_limited
        @test d_out ≈ [2.0, 0.0, 0.0]

        d2 = [0.0, 1.0, 0.0]

        @test Manopt.find_gcp_direction!(gf, d_out, p, d2, [0.0, -1.0, 0.0]) === :found_unlimited
        @test d_out ≈ d2

        @test Manopt.find_gcp_direction!(gf, d_out, p, [1.0, 1.0, 0.0], [-10.0, -10.0, -10.0]) === :found_limited
        @test d_out ≈ [2.0, 10.0, 0.0]
    end
    @testset "Pure Hyperrectangle" begin
        M = Hyperrectangle([-1.0, 2.0, -Inf], [2.0, Inf, 2.0])
        f(M, p) = sum(p .^ 2)
        grad_f(M, p) = 2 .* p
        p0 = [0.0, 4.0, 10.0]
        # p_opt = quasi_Newton(M, f, grad_f, p0)
    end

    @testset "requires_gcp" begin
        @test !Manopt.requires_gcp(Sphere(2))
        @test Manopt.requires_gcp(Hyperrectangle([1], [2]))
        @test Manopt.requires_gcp(ProductManifold(Hyperrectangle([1], [2]), Sphere(2)))
    end

    @testset "Hyperrectangle × Sphere" begin

    end
end
