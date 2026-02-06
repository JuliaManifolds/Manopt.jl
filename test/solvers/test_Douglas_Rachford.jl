using Manifolds, Manopt, Test
using ManifoldDiff: prox_distance, prox_distance!

@testset "DouglasRachford" begin
    # Though this seems a strange way, it is a way to compute the mid point
    M = Sphere(2)
    d1 = [1.0, 0.0, 0.0]
    d2 = [0.0, 1.0, 0.0]
    d3 = [0.0, 0.0, 1.0]
    p = [0.0, 0.0, 1.0]
    p_star = geodesic(M, d1, d2, distance(M, d1, d2) / 2)
    f(M, p) = distance(M, p, d1)^2 + distance(M, p, d2)^2
    prox1a = (M, η, p) -> prox_distance(M, η, d1, p)
    prox2a = (M, η, p) -> prox_distance(M, η, d2, p)
    @test_throws ErrorException DouglasRachford(M, f, Array{Function, 1}([prox1a]), p)
    q1a = DouglasRachford(M, f, [prox1a, prox2a], p)
    @test isapprox(M, q1a, p_star; atol = 1.0e-14)
    q1i = DouglasRachford(
        M, f, [prox1a, prox2a], p; reflection_evaluation = InplaceEvaluation()
    )
    @test isapprox(M, q1i, p_star; atol = 1.0e-14)
    prox1i = (M, q, η, p) -> prox_distance!(M, q, η, d1, p)
    prox2i = (M, q, η, p) -> prox_distance!(M, q, η, d2, p)
    q2 = copy(M, p)
    DouglasRachford!(M, f, [prox1i, prox2i], q2; evaluation = InplaceEvaluation())
    @test isapprox(M, q1a, q2)
    # compute the Riemannian center of mass (locally) on Sn
    # though also this is not that useful, but easy to test that DR works
    F2(M, p) = distance(M, p, d1)^2 + distance(M, p, d2)^2 + distance(M, p, d3)^2
    prox3a = (M, η, p) -> prox_distance(M, η, d3, p)
    q3 = DouglasRachford(M, F2, [prox1a, prox2a, prox3a], p)
    p_star_2 = mean(M, [d1, d2, d3])
    # since the default does not run that long -> rough estimate
    @test isapprox(M, q3, p_star_2; atol = 1.0e-14)
    prox3i = (M, q, η, p) -> prox_distance!(M, q, η, d3, p)
    q4 = DouglasRachford(M, F2, [prox1i, prox2i, prox3i], p; evaluation = InplaceEvaluation())
    # since the default does not run that long -> rough estimate
    @test isapprox(M, q4, p_star_2; atol = 1.0e-14)

    #test getter/set
    s = DouglasRachfordState(M; p = d1)
    sr = "# Solver state for `Manopt.jl`s Douglas Rachford Algorithm\n"
    @test startswith(Manopt.status_summary(s; inline = false), sr)
    set_iterate!(s, d2)
    @test get_iterate(s) == d2
    @testset "Debug and Record prox parameter" begin
        io = IOBuffer()
        mpo = ManifoldProximalMapObjective(f, [prox1a, prox2a, prox3a])
        p = DefaultManoptProblem(M, mpo)
        ds = DebugSolverState(s, DebugProximalParameter(; io = io))
        step_solver!(p, ds, 1)
        debug = String(take!(io))
        @test startswith(debug, "λ:")
        rs = RecordSolverState(s, RecordProximalParameter())
        step_solver!(p, rs, 1)
        @test get_record(rs) == [1.0]
    end
    @testset "Number" begin
        M = Circle()
        data = [-π / 4, π / 4]
        p = π / 8
        p_star = 0
        f3(M, p) = distance(M, p, data[1])^2 + distance(M, p, data[2])^2
        prox1b = (M, η, p) -> prox_distance(M, η, data[1], p)
        prox2b = (M, η, p) -> prox_distance(M, η, data[2], p)
        q1 = DouglasRachford(M, f3, [prox1b, prox2b], p)
        @test q1 == p_star
        q2 = DouglasRachford(M, f3, [prox1b, prox2b], p; evaluation = InplaceEvaluation())
        @test q1 == q2
        s = DouglasRachford(M, f3, [prox1b, prox2b], p; return_state = true)
        q3 = get_solver_result(s)[]
        @test q2 == q3
    end
end
