using Manifolds, Manopt, Test, ManifoldsBase

@testset "Differentials" begin
    p = [1.0, 0.0, 0.0]
    q = [0.0, 1.0, 0.0]
    M = Sphere(2)
    X = log(M, p, q)
    Y = similar(X)
    @testset "forward logs" begin
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
        @test isapprox(
            N, x, Manopt.differential_log_argument(N, x, y, V), [V[1], V[2], V[2]]
        )
        Manopt.differential_log_argument!(N, W, x, y, V)
        @test isapprox(N, x, W, [V[1], V[2], V[2]])
    end
    @testset "forward logs on a multivariate power manifold" begin
        S = Sphere(2)X
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
            Manopt.differential_log_basepoint(S, p[1, 1], p[2, 1], X[1, 1]) +
            Manopt.differential_log_argument(S, p[1, 1], p[2, 1], X[2, 1])
        @test t2[1, 1, 1] ≈ a
        @test t2[1, 2, 1] ≈ zero_vector(S, p[1, 2]) atol = 1e-17
        @test t2[2, 1, 1] ≈ zero_vector(S, p[2, 1]) atol = 1e-17
        @test t2[2, 2, 1] ≈ zero_vector(S, p[2, 2]) atol = 1e-17
        b =
            Manopt.differential_log_basepoint(S, p[1, 1], p[1, 2], X[1, 1]) +
            Manopt.differential_log_argument(S, p[1, 1], p[1, 2], X[1, 2])
        @test t2[1, 1, 2] ≈ b
        @test t2[1, 2, 2] ≈ zero_vector(S, p[1, 2]) atol = 1e-17
        @test t2[2, 1, 2] ≈ zero_vector(S, p[2, 1]) atol = 1e-17
        @test t2[2, 2, 2] ≈ zero_vector(S, p[2, 2]) atol = 1e-17
    end
end
