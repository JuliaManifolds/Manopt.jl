using Manifolds, Manopt, Test, ManifoldsBase
using LinearAlgebra: I

using Random
Random.seed!(42)
# Test the additional manifold functions
#
@testset "Additional Manifold functions" begin
    @testset "mid point & reflect" begin
        M = Sphere(2)
        p = [1.0, 0.0, 0.0]
        q = [0.0, 1.0, 0.0]

        r = mid_point(M, p, q)
        r2 = similar(r)
        mid_point!(M, r2, p, q)
        r3 = shortest_geodesic(M, p, q, 0.5)
        r4 = mid_point(M, p, q, q)
        @test isapprox(M, r, r2)
        @test isapprox(M, r2, r3)
        @test isapprox(M, r3, r4)
        r5 = similar(r4)
        mid_point!(M, r5, p, q, q)
        @test isapprox(M, r4, r5)

        r4 = mid_point(M, p, -p, q)
        r5 = similar(r4)
        mid_point!(M, r5, p, -p, q)
        @test isapprox(M, r4, q)
        @test isapprox(M, r4, r5)

        @test isapprox(M, reflect(M, p, q), -q)
        qA = similar(q)
        reflect!(M, qA, p, q)
        @test isapprox(M, qA, -q)
        f = x -> x
        @test reflect(M, f, q) == q
        qA = similar(q)
        reflect!(M, qA, f, q)
        @test qA == q

        M2 = Euclidean(2)
        p2 = [1.0, 0.0]
        q2 = [0.0, 1.0]
        s = mid_point(M2, p2, q2)
        s2 = similar(s)
        mid_point!(M2, s2, p2, q2)
        @test s == s2
        @test s == (p2 + q2) / 2
        s = mid_point(M2, p2, q2, s)
        @test s == s2
        s2 = similar(s)
        mid_point!(M2, s2, p2, q2, s)
        @test s == s2

        M = Circle()
        p = 0
        q = π
        @test mid_point(M, p, q, 1.0) ≈ π / 2
        @test mid_point(M, p, q, -1.0) ≈ -π / 2
        @test mid_point(M, 0, π / 2) ≈ π / 4
    end
end
