using Manifolds, Manopt, Test, ManifoldsBase

@testset "gradients" begin
    @testset "Circle (Allocating)" begin
        M = Circle()
        N = PowerManifold(M, 4)
        x = [0.1, 0.2, 0.3, 0.5]
        tvTestξ = [-1.0, 0.0, 0.0, 1.0]
        @test grad_TV(N, x) == tvTestξ
        @test grad_TV(M, (x[1], x[1])) ==
              (zero_tangent_vector(M, x[1]), zero_tangent_vector(M, x[1]))
        @test norm(N, x, grad_TV(N, x, 2) - tvTestξ) ≈ 0
        tv2Testξ = [0.0, 0.5, -1.0, 0.5]
        @test grad_TV2(N, x) == tv2Testξ
        @test norm(N, x, forward_logs(N, x) - [0.1, 0.1, 0.2, 0.0]) ≈ 0 atol = 10^(-16)
        @test norm(
            N,
            x,
            grad_intrinsic_infimal_convolution_TV12(N, x, x, x, 1.0, 1.0)[1] -
            [-1.0, 0.0, 0.0, 1.0],
        ) ≈ 0
        @test norm(N, x, grad_intrinsic_infimal_convolution_TV12(N, x, x, x, 1.0, 1.0)[2]) ≈
              0
        x2 = [0.1, 0.2, 0.3]
        N2 = PowerManifold(M, size(x2)...)
        @test grad_TV2(N2, x2) == zeros(3)
        @test grad_TV2(N2, x2, 2) == zeros(3)
        @test grad_TV(M, (0.0, 0.0), 2) == (0.0, 0.0)
        # 2d forward logs
        N3 = PowerManifold(M, 2, 2)
        N3C = PowerManifold(M, 2, 2, 2)
        x3 = [0.1 0.2; 0.3 0.5]
        x3C = cat(x3, x3; dims=3)
        tC = cat([0.2 0.3; 0.0 0.0], [0.1 0.0; 0.2 0.0]; dims=3)
        @test norm(N3C, x3C, forward_logs(N3, x3) - tC) ≈ 0 atol = 10^(-16)

        M = Circle()
        p = 0
        q = π / 4
        @test grad_distance(M, p, q) == q - p
    end
    @testset "Sphere (Mutating)" begin
        M = Sphere(2)
        p = [0.0, 0.0, 1.0]
        q = [0.0, 1.0, 0.0]
        r = [1.0, 0.0, 0.0]
        @testset "Gradient of the distance function" begin
            X = zero_tangent_vector(M, p)
            grad_distance!(M, X, p, q)
            Y = grad_distance(M, p, q)
            Z = [0.0, 0.0, -π / 2] # known solution
            @test X == Y
            @test X == Z
        end
        @testset "Grad of second order total variation" begin
            N = PowerManifold(M, NestedPowerRepresentation(), 3)
            s = [p, q, r]
            X = zero_tangent_vector(N, s)
            grad_TV2!(M, X, s)
            Y = grad_TV2(M, s)
            Z = -1 / sqrt(2) .* [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
            @test Y == X
            @test Y ≈ Z
        end
    end
end
