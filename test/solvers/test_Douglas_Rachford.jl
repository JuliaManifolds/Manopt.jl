using Manifolds, Manopt, Test
@testset "DouglasRachford" begin
    # Though this seems a strange way, it is a way to compute the mid point
    M = Sphere(2)
    d1 = [1.0, 0.0, 0.0]
    d2 = [0.0, 1.0, 0.0]
    d3 = [0.0, 0.0, 1.0]
    start = [0.0, 0.0, 1.0]
    result = result = geodesic(M, d1, d2, distance(M, d1, d2) / 2)
    f(M, p) = distance(M, p, d1)^2 + distance(M, p, d2)^2
    prox1 = (M, η, p) -> prox_distance(M, η, d1, p)
    prox2 = (M, η, p) -> prox_distance(M, η, d2, p)
    @test_throws ErrorException DouglasRachford(M, f, Array{Function,1}([prox1]), start) # we need more than one prox
    xHat = DouglasRachford(M, f, [prox1, prox2], start)
    @test f(M, start) > f(M, xHat)
    @test distance(M, xHat, result) ≈ 0
    # but we can also compute the riemannian center of mass (locally) on Sn
    # though also this is not that useful, but easy to test that DR works
    F2(M, p) = distance(M, p, d1)^2 + distance(M, p, d2)^2 + distance(M, p, d3)^2
    prox1 = (M, η, p) -> prox_distance(M, η, d1, p)
    prox2 = (M, η, p) -> prox_distance(M, η, d2, p)
    prox3 = (M, η, p) -> prox_distance(M, η, d3, p)
    xHat2 = DouglasRachford(M, F2, [prox1, prox2, prox3], start;)
    result2 = mean(M, [d1, d2, d3])
    # since the default does not run that long -> rough estimate
    @test distance(M, xHat2, result2) ≈ 0 atol = 1e-7
    #test getter/set
    s = DouglasRachfordState(M, d1)
    set_iterate!(s, d2)
    @test get_iterate(s) == d2
    @testset "Debug and Record prox parameter" begin
        io = IOBuffer()
        mpo = ManifoldProximalMapObjective(f, [prox1, prox2, prox3])
        p = DefaultManoptProblem(M, mpo)
        ds = DebugSolverState(s, DebugProximalParameter(; io=io))
        step_solver!(p, ds, 1)
        debug = String(take!(io))
        @test startswith(debug, "λ:")
        rs = RecordSolverState(s, RecordProximalParameter())
        step_solver!(p, rs, 1)
        @test get_record(rs) == [1.0]
    end
end
