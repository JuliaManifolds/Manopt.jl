using Manifolds, Manopt, Test, Dates

@testset "proximal maps" begin
    #
    # prox_TV
    p = [1.0, 0.0, 0.0]
    q = [0.0, 1.0, 0.0]
    M = Sphere(2)
    N = PowerManifold(M, NestedPowerRepresentation(), 2)
    @test_throws ErrorException prox_distance(M, 1.0, p, q, 3)
    @test_throws ErrorException prox_distance!(M, p, 1.0, p, q, 3)
    @test distance(
        M, prox_distance(M, distance(M, p, q) / 2, p, q, 1), shortest_geodesic(M, p, q, 0.5)
    ) ≈ 0
    t = similar(p)
    prox_distance!(M, t, distance(M, p, q) / 2, p, q, 1)
    @test t == prox_distance(M, distance(M, p, q) / 2, p, q, 1)
    (r, s) = prox_TV(M, π / 4, (p, q))
    X = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    prox_TV!(M, X, π / 4, (p, q))
    @test norm(r - s) < eps(Float64)
    @test norm(X[1] - s) < eps(Float64)
    @test norm(X[2] - r) < eps(Float64)
    # i.e. they are moved together
    @test distance(M, r, s) < eps(Float64)
    (t, u) = prox_TV(M, π / 8, (p, q))
    @test_throws ErrorException prox_TV(M, π, (p, q), 3)
    @test_throws ErrorException prox_TV!(M, [p, q], π, (p, q), 3)
    # they cross correlate
    @test (
        abs(t[1] - u[2]) < eps(Float64) &&
        abs(t[2] - u[1]) < eps(Float64) &&
        abs(t[3] - u[3]) < eps(Float64)
    )
    @test distance(M, t, u) == π / 4 # and have moved half their distance
    #
    (v, w) = prox_TV(M, 1.0, (p, q), 2)
    vC, wC = shortest_geodesic(M, p, q, [1 / 3, 2 / 3])
    @test distance(M, v, vC) ≈ 0
    @test distance(M, w, wC) ≈ 0
    P = [similar(p), similar(q)]
    prox_TV!(M, P, 1.0, (p, q), 2)
    @test P == [v, w]
    # prox_TV on Power
    T = prox_TV(N, π / 8, [p, q])
    @test distance(N, T, [t, u]) ≈ 0
    # parallelprox_TV
    N2 = PowerManifold(M, NestedPowerRepresentation(), 3)
    r = geodesic(M, p, q, 0.5)
    s, t = prox_TV(M, π / 16, (r, q))
    u, v = prox_TV(M, π / 16, (p, r))
    y = prox_parallel_TV(N2, π / 16, [[p, r, q], [p, r, q]])
    @test distance(N2, y[1], [p, s, t]) ≈ 0 # even indices in first comp
    @test distance(N2, y[2], [u, v, q]) ≈ 0 # odd in second
    # dimensions of x have to fit, here they don't
    @test_throws ErrorException prox_parallel_TV(N2, π / 16, [[p, r, q]])
    # prox_TV2
    p2, r2, q2 = prox_TV2(M, 1.0, (p, r, q))
    sum(distance.(Ref(M), [p, r, q], [p2, r2, q2])) ≈ 0
    @test_throws ErrorException prox_TV2(M, 1.0, (p, r, q), 2) # since prox_TV is only defined for p=1
    distance(
        PowerManifold(M, NestedPowerRepresentation(), 3),
        [p2, r2, q2],
        prox_TV2(PowerManifold(M, NestedPowerRepresentation(), 3), 1.0, [p, r, q]),
    ) ≈ 0
    # Circle
    M2 = Circle()
    N2 = PowerManifold(M2, 3)
    pS, rS, qS = [-0.5, 0.1, 0.5]
    d = dot([pS, rS, qS], [1.0, -2.0, 1.0])
    m = min(0.3, abs(Manopt.sym_rem(d) / 6))
    s = sign(Manopt.sym_rem(d))
    pSc, rSc, qSc = Manopt.sym_rem.([pS, rS, qS] .- m .* s .* [1.0, -2.0, 1.0])
    pSr, rSr, qSr = prox_TV2(M2, 0.3, (pS, rS, qS))
    @test sum(distance.(Ref(M2), [pSc, rSc, qSc], [pSr, rSr, qSr])) ≈ 0
    # p=2
    t = 0.3 * Manopt.sym_rem(d) / (1 + 0.3 * 6.0)
    @test sum(
        distance.(
            Ref(M2),
            [prox_TV2(M2, 0.3, (pS, rS, qS), 2)...],
            [pS, rS, qS] .- t .* [1.0, -2.0, 1.0],
        ),
    ) ≈ 0
    # others fail
    @test_throws ErrorException prox_TV2(M2, 0.3, (pS, rS, qS), 3)
    # Rn
    M3 = Euclidean(1)
    pR, rR, qR = [pS, rS, qS]
    m = min.(Ref(0.3), abs.([pR, rR, qR] .* [1.0, -2.0, 1.0]) / 6)
    s = sign(d)  # we can reuse d
    pRc, rRc, qRc = [pR, rR, qR] .- m .* s .* [1.0, -2.0, 1.0]
    pRr, rRr, qRr = prox_TV2(M3, 0.3, (pR, rR, qR))
    @test sum(distance.(Ref(M3), [pRc, rRc, qRc], [pRr, rRr, qRr])) ≈ 0
    # p=2
    t = 0.3 * d / (1 + 0.3 * 6.0)
    @test sum(
        distance.(
            Ref(M3),
            [prox_TV2(M3, 0.3, (pR, rR, qR), 2)...],
            [pR, rR, qR] .- t .* [1.0, -2.0, 1.0],
        ),
    ) ≈ 0
    # others fail
    @test_throws ErrorException prox_TV2(M3, 0.3, (pR, rR, qR), 3)
    #
    # collaborative integer tests
    #
    @test_throws ErrorException prox_TV2(M3, 0.3, (pS, rS, qS), 3)
    ξR, ηR, νR = [pS, rS, qS]
    N3 = PowerManifold(M3, 3)
    P = [pR rR qR]
    Ξ = [ξR ηR νR]
    @test project_collaborative_TV(N3, 0.0, P, Ξ, 1, 1) == Ξ
    @test project_collaborative_TV(N3, 0.0, P, Ξ, 1.0, 1) == Ξ
    @test project_collaborative_TV(N3, 0.0, P, Ξ, 1, 1.0) == Ξ
    @test project_collaborative_TV(N3, 0.0, P, Ξ, 1.0, 1.0) == Ξ

    @test project_collaborative_TV(N3, 0.0, P, Ξ, 2, 1) == Ξ
    @test norm(N3, P, project_collaborative_TV(N3, 0.0, P, Ξ, 2, Inf)) ≈ norm(Ξ)
    @test sum(abs.(project_collaborative_TV(N3, 0.0, P, Ξ, 1, Inf))) ≈ 1.0
    @test norm(N3, P, project_collaborative_TV(N3, 0.0, P, Ξ, Inf, Inf)) ≈ norm(Ξ)
    @test_throws ErrorException project_collaborative_TV(N3, 0.0, P, Ξ, 3, 3)
    @test_throws ErrorException project_collaborative_TV(N3, 0.0, P, Ξ, 3, 1)
    @test_throws ErrorException project_collaborative_TV(N3, 0.0, P, Ξ, 3, Inf)

    @testset "Multivariate project collaborative TV" begin
        S = Sphere(2)
        M = PowerManifold(S, NestedPowerRepresentation(), 2, 2, 2)
        p = [zeros(3) for i in [1, 2], j in [1, 2], k in [1, 2]]
        p[1, 1, 1] = [1.0, 0.0, 0.0]
        p[1, 2, 1] = 1 / sqrt(2) .* [1.0, 1.0, 0.0]
        p[2, 1, 1] = 1 / sqrt(2) .* [1.0, 0.0, 1.0]
        p[2, 2, 1] = [0.0, 1.0, 0.0]
        p[:, :, 2] = deepcopy(p[:, :, 1])
        X = zero_tangent_vector(M, p)
        X[1, 1, 1] .= [0.0, 0.5, 0.5]
        norm(project_collaborative_TV(M, 1, p, X, 2, 1)) ≈ 0
        @test norm(project_collaborative_TV(M, 0.5, p, X, 2, 1)) ≈ (norm(X[1, 1, 1]) - 0.5)
        Nf = PowerManifold(S, NestedPowerRepresentation(), 2, 2, 1)
        @test_throws ErrorException project_collaborative_TV(Nf, 1, p, X, 2, 1)
    end
end
