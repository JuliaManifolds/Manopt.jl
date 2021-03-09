using Manifolds, Manopt, Test, ManifoldsBase

@testset "Differentials (on Sphere(2))" begin
    # The Adjoint Differentials test using the same variables as the differentials
    # test
    p = [1.0, 0.0, 0.0]
    q = [0.0, 1.0, 0.0]
    M = Sphere(2)
    X = log(M, p, q)
    # Tept differentials (1) Dp of Log_pq
    Y = similar(X)
    @test adjoint_differential_log_basepoint(M, p, p, X) == -X
    adjoint_differential_log_basepoint!(M, Y, p, p, X)
    @test Y == -X
    @test adjoint_differential_log_argument(M, p, p, X) == X
    adjoint_differential_log_argument!(M, Y, p, p, X)
    @test Y == X
    @test adjoint_differential_exp_basepoint(M, p, zero_tangent_vector(M, p), X) == X
    adjoint_differential_exp_basepoint!(M, Y, p, zero_tangent_vector(M, p), X)
    @test Y == X
    @test adjoint_differential_exp_argument(M, p, zero_tangent_vector(M, p), X) == X
    adjoint_differential_exp_argument!(M, Y, p, zero_tangent_vector(M, p), X)
    @test Y == X
    for t in [0, 0.15, 0.33, 0.66, 0.9]
        @test adjoint_differential_geodesic_startpoint(M, p, p, t, X) == (1 - t) * X
        adjoint_differential_geodesic_startpoint!(M, Y, p, p, t, X)
        @test Y == (1 - t) * X
        @test norm(M, p, adjoint_differential_geodesic_endpoint(M, p, p, t, X) - t * X) ≈ 0 atol =
            10.0^(-16)
        adjoint_differential_geodesic_endpoint!(M, Y, p, p, t, X)
        @test norm(M, p, Y - t * X) ≈ 0 atol = 10.0^(-16)
    end
    Mp = PowerManifold(M, NestedPowerRepresentation(), 3)
    pP = [p, q, p]
    qP = [p, p, q]
    XP = [X, zero_tangent_vector(M, p), -X]
    YP = similar.(XP)
    @test norm(
        Mp,
        pP,
        adjoint_differential_forward_logs(Mp, pP, XP) -
        [-X, zero_tangent_vector(M, p), zero_tangent_vector(M, p)],
    ) ≈ 0 atol = 4 * 10.0^(-16)
    adjoint_differential_forward_logs!(Mp, YP, pP, XP)
    @test norm(Mp, pP, YP - [-X, zero_tangent_vector(M, p), zero_tangent_vector(M, p)]) ≈ 0 atol =
        4 * 10.0^(-16)
    ZP = [[0.0, π / 2, 0.0], [0.0, 0.0, 0.0], [π / 2, 0.0, 0.0]]
    @test adjoint_differential_log_argument(Mp, pP, qP, XP) == ZP
    adjoint_differential_log_argument!(Mp, YP, pP, qP, XP)
    @test ZP == YP
end
