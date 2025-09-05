using Manopt, Manifolds, ManifoldDiff, Test
using ManifoldDiff: prox_distance, prox_distance!

@testset "Proximal Point" begin
    # Dummy problem
    M = Sphere(2)
    q = [1.0, 0.0, 0.0]
    f(M, p) = 0.5 * distance(M, p, q)^2
    prox_f(M, λ, p) = prox_distance(M, λ, q, p)
    prox_f!(M, r, λ, p) = prox_distance!(M, r, λ, q, p)

    p0 = [0.0, 0.0, 1.0]
    q1 = proximal_point(M, prox_f, p0)
    @test distance(M, q, q1) < 1.0e-12
    q2 = copy(M, p0)
    os2 = proximal_point!(
        M, prox_f!, q2; evaluation = InplaceEvaluation(), return_objective = true
    )
    @test isapprox(M, q1, q2)
    q2a = get_proximal_map(M, os2[1], 1.0, q2)
    @test isapprox(M, q2, q2a)
    os3 = proximal_point(M, prox_f, p0; return_state = true, return_objective = true)
    obj = os3[1]
    # test with get_prox map that these are fix points
    pps = os3[2]
    q3a = get_proximal_map(M, obj, 1.0, get_iterate(pps))
    @test isapprox(M, q2, q3a)
    q3b = rand(M)
    get_proximal_map!(M, q3b, obj, 1.0, get_iterate(pps))
    @test distance(M, q3a, q3b) == 0
    @test startswith(repr(pps), "# Solver state for `Manopt.jl`s Proximal Point Method\n")
end
