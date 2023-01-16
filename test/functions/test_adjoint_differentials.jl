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
    Mp = PowerManifold(M, NestedPowerRepresentation(), 3)
    pP = [p, q, p]
    qP = [p, p, q]
    XP = [X, zero_vector(M, p), -X]
    YP = similar.(XP)
    ZP = adjoint_differential_forward_logs(Mp, pP, XP)
    @test norm(Mp, pP, ZP - [-X, X, zero_vector(M, p)]) ≈ 0 atol = 4 * 10.0^(-16)
    adjoint_differential_forward_logs!(Mp, YP, pP, XP)
    @test isapprox(Mp, pP, YP, ZP)
    ZP = [[0.0, π / 2, 0.0], [0.0, 0.0, 0.0], [π / 2, 0.0, 0.0]]
    @test Manopt.adjoint_differential_log_argument(Mp, pP, qP, XP) == ZP
    Manopt.adjoint_differential_log_argument!(Mp, YP, pP, qP, XP)
    @test ZP == YP
end
