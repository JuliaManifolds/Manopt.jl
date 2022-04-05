using Manifolds, Manopt, Test, ManifoldsBase

@testset "Differentials" begin
    p = [1.0, 0.0, 0.0]
    q = [0.0, 1.0, 0.0]
    M = Sphere(2)
    X = log(M, p, q)
    Y = similar(X)
    @testset "Differentials on Sn(2)" begin
        @test differential_log_basepoint(M, p, p, X) == -X
        differential_log_basepoint!(M, Y, p, p, X)
        @test Y == -X
        @test differential_log_basepoint(M, p, q, X) == -X
        differential_log_basepoint!(M, Y, p, q, X) == -X
        @test Y == -X
        @test differential_log_argument(M, p, p, X) == X
        differential_log_argument!(M, Y, p, p, X) == X
        @test Y == X
        @test differential_log_argument(M, p, q, X) == zero_vector(M, q)
        differential_log_argument!(M, Y, p, q, X)
        @test Y == zero_vector(M, q)
        @test differential_exp_basepoint(M, p, zero_vector(M, p), X) == X
        differential_exp_basepoint!(M, Y, p, zero_vector(M, p), X)
        @test Y == X
        @test norm(M, q, differential_exp_basepoint(M, p, X, X) - [-π / 2, 0.0, 0.0]) ≈ 0 atol =
            6 * 10^(-16)
        differential_exp_basepoint!(M, Y, p, X, X)
        @test norm(M, q, Y - [-π / 2, 0.0, 0.0]) ≈ 0 atol = 6 * 10^(-16)
        @test differential_exp_argument(M, p, zero_vector(M, p), X) == X
        differential_exp_argument!(M, Y, p, zero_vector(M, p), X) == X
        @test Y == X
        @test norm(M, q, differential_exp_argument(M, p, X, zero_vector(M, p))) ≈ 0
        differential_exp_argument!(M, Y, p, X, zero_vector(M, p))
        @test norm(M, q, Y) ≈ 0
        for t in [0, 0.15, 0.33, 0.66, 0.9]
            @test differential_geodesic_startpoint(M, p, p, t, X) == (1 - t) * X
            differential_geodesic_startpoint!(M, Y, p, p, t, X)
            @test Y == (1 - t) * X
            @test norm(M, p, differential_geodesic_endpoint(M, p, p, t, X) - t * X) ≈ 0 atol =
                10.0^(-16)
            differential_geodesic_endpoint!(M, Y, p, p, t, X)
            @test norm(M, p, Y - t * X) ≈ 0 atol = 10.0^(-16)
        end
    end
    @testset "Differentials on Power of Sn(2)" begin
        N = PowerManifold(M, NestedPowerRepresentation(), 3)
        x = [p, q, p]
        y = [p, p, q]
        V = [X, zero_vector(M, p), -X]
        W = similar.(V)
        @test norm(
            N,
            x,
            differential_forward_logs(N, x, V) - [-X, [π / 2, 0.0, 0.0], zero_vector(M, p)],
        ) ≈ 0 atol = 8 * 10.0^(-16)
        differential_forward_logs!(N, W, x, V)
        @test norm(N, x, W - [-X, [π / 2, 0.0, 0.0], zero_vector(M, p)]) ≈ 0 atol =
            8 * 10.0^(-16)
        @test differential_log_argument(N, x, y, V) == [V[1], V[2], V[2]]
        differential_log_argument!(N, W, x, y, V)
        @test W == [V[1], V[2], V[2]]
    end
    @testset "Differentials on SPD(2)" begin
        #
        # Single differentials on Hn
        M2 = SymmetricPositiveDefinite(2)
        p2 = [1.0 0.0; 0.0 1.0]
        X2 = [0.5 1.0; 1.0 0.5]
        q2 = exp(M2, p2, X2)
        # Text differentials (1) Dx of Log_xy
        @test norm(M2, p2, differential_log_basepoint(M2, p2, p2, X2) + X2) ≈ 0 atol =
            4 * 10^(-16)
        @test norm(M2, q2, differential_log_argument(M2, p2, q2, zero_vector(M2, p2))) ≈ 0 atol =
            4 * 10^(-16)
        @test norm(
            M2, p2, differential_exp_basepoint(M2, p2, zero_vector(M2, p2), X2) - X2
        ) ≈ 0 atol = 4 * 10^(-16)
        @test norm(
            M2, p2, differential_exp_argument(M2, p2, zero_vector(M2, p2), X2) - X2
        ) ≈ 0 atol = 4 * 10^(-16)
        for t in [0, 0.15, 0.33, 0.66, 0.9]
            @test norm(
                M2, p2, differential_geodesic_startpoint(M2, p2, p2, t, X2) - (1 - t) * X2
            ) ≈ 0 atol = 4 * 10^(-16)
            @test norm(M2, p2, differential_geodesic_endpoint(M2, p2, p2, t, X2) - t * X2) ≈
                0 atol = 4 * 10.0^(-16)
        end
        @test norm(M2, q2, differential_geodesic_startpoint(M2, p2, q2, 1.0, X2)) ≈ 0 atol =
            4 * 10.0^(-16)
        @test norm(M2, q2, differential_exp_basepoint(M2, p2, X2, zero_vector(M2, p2))) ≈ 0 atol =
            4 * 10.0^(-16)
        @test norm(M2, q2, differential_exp_argument(M2, p2, X2, zero_vector(M2, p2))) ≈ 0 atol =
            4 * 10.0^(-16)
        # test coeff of log_basepoint, since it is not always expicitly used.
        @test βdifferential_log_basepoint(-1.0, 1.0, 2.0) ≈ -2 * cosh(2.0) / sinh(2.0)
    end
    @testset "Differentials on Euclidean(2)" begin
        M3 = Euclidean(2)
        x3 = [1.0, 2.0]
        ξ3 = [1.0, 0.0]
        @test norm(M3, x3, differential_exp_basepoint(M3, x3, ξ3, ξ3) - ξ3) ≈ 0 atol =
            4 * 10.0^(-16)
    end
    @testset "Differentials on the Circle" begin
        M = Circle()
        p = 0
        q = π / 4
        X = π / 8
        @test differential_log_argument(M, p, q, X) == X
    end
    @testset "forward logs on a multivariate power manifold" begin
        S = Sphere(2)
        M = PowerManifold(S, NestedPowerRepresentation(), 2, 2)
        p = [zeros(3) for i in [1, 2], j in [1, 2]]
        p[1, 1] = [1.0, 0.0, 0.0]
        p[1, 2] = 1 / sqrt(2) .* [1.0, 1.0, 0.0]
        p[2, 1] = 1 / sqrt(2) .* [1.0, 0.0, 1.0]
        p[2, 2] = [0.0, 1.0, 0.0]
        t1 = forward_logs(M, p)
        @test t1[1, 1, 1] ≈ log(S, p[1, 1], p[2, 1])
        @test t1[1, 1, 2] ≈ log(S, p[1, 1], p[1, 2])
        @test t1[1, 2, 1] ≈ log(S, p[1, 2], p[2, 2])
        @test t1[1, 2, 2] ≈ log(S, p[1, 2], p[1, 2]) atol = 1e-15
        @test t1[2, 1, 1] ≈ log(S, p[2, 1], p[2, 1]) atol = 1e-15
        @test t1[2, 1, 2] ≈ log(S, p[2, 1], p[2, 2])
        @test t1[2, 2, 1] ≈ log(S, p[2, 2], p[2, 2])
        @test t1[2, 2, 2] ≈ log(S, p[2, 2], p[2, 2])
        t1a = zero.(t1)
        forward_logs!(M, t1a, p)
        @test all(t1 .== t1a)
        X = zero_vector(M, p)
        X[1, 1] .= [0.0, 0.5, 0.5]
        t2 = differential_forward_logs(M, p, X)
        a =
            differential_log_basepoint(S, p[1, 1], p[2, 1], X[1, 1]) +
            differential_log_argument(S, p[1, 1], p[2, 1], X[2, 1])
        @test t2[1, 1, 1] ≈ a
        @test t2[1, 2, 1] ≈ zero_vector(S, p[1, 2]) atol = 1e-17
        @test t2[2, 1, 1] ≈ zero_vector(S, p[2, 1]) atol = 1e-17
        @test t2[2, 2, 1] ≈ zero_vector(S, p[2, 2]) atol = 1e-17
        b =
            differential_log_basepoint(S, p[1, 1], p[1, 2], X[1, 1]) +
            differential_log_argument(S, p[1, 1], p[1, 2], X[1, 2])
        @test t2[1, 1, 2] ≈ b
        @test t2[1, 2, 2] ≈ zero_vector(S, p[1, 2]) atol = 1e-17
        @test t2[2, 1, 2] ≈ zero_vector(S, p[2, 1]) atol = 1e-17
        @test t2[2, 2, 2] ≈ zero_vector(S, p[2, 2]) atol = 1e-17
    end
end
