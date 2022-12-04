using Manifolds, Manopt, Test
@testset "DouglasRachford" begin
    # Though this seems a strange way, it is a way to compute the mid point
    M = Sphere(2)
    p = [1.0, 0.0, 0.0]
    q = [0.0, 1.0, 0.0]
    r = [0.0, 0.0, 1.0]
    start = [0.0, 0.0, 1.0]
    result = result = geodesic(M, p, q, distance(M, p, q) / 2)
    F(M, x) = distance(M, x, p)^2 + distance(M, x, q)^2
    prox1 = (M, η, x) -> prox_distance(M, η, p, x)
    prox2 = (M, η, x) -> prox_distance(M, η, q, x)
    @test_throws ErrorException DouglasRachford(M, F, Array{Function,1}([prox1]), start) # we need more than one prox
    xHat = DouglasRachford(M, F, [prox1, prox2], start)
    @test F(M, start) > F(M, xHat)
    @test distance(M, xHat, result) ≈ 0
    # but we can also compute the riemannian center of mass (locally) on Sn
    # though also this is not that useful, but easy to test that DR works
    F2(M, x) = distance(M, x, p)^2 + distance(M, x, q)^2 + distance(M, x, r)^2
    prox1 = (M, η, x) -> prox_distance(M.manifold, η, p, x)
    prox2 = (M, η, x) -> prox_distance(M.manifold, η, q, x)
    prox3 = (M, η, x) -> prox_distance(M.manifold, η, r, x)
    o = DouglasRachford(
        M,
        F2,
        [prox1, prox2, prox3],
        start;
        debug=[DebugCost(), DebugIterate(), DebugProximalParameter(), 100],
        record=[RecordCost(), RecordProximalParameter()],
        return_options=true,
    )
    xHat2 = get_solver_result(o)
    drec2 = get_record(o)
    result2 = mean(M, [p, q, r])
    # since the default does not run that long -> rough estimate
    @test distance(M, xHat2, result2) ≈ 0
    #test getter/set
    O = DouglasRachfordOptions(M, p)
    set_iterate!(O, q)
    @test get_iterate(O) == q
end
