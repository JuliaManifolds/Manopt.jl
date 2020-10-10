using Manifolds, Manopt, Test, ManifoldsBase

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
        f = x -> x
        @test reflect(M, f, q) == q

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
    @testset "random" begin
        Mc = Circle()
        pc = random_point(Mc)
        @test is_manifold_point(Mc, pc, true)
        Xc = random_tangent(Mc, pc)
        @test is_tangent_vector(Mc, pc, Xc, true)

        Me = Euclidean(3)
        pe = random_point(Me)
        @test is_manifold_point(Me, pe, true)
        Xe = random_tangent(Me, pe)
        @test is_tangent_vector(Me, pe, Xe, true)

        Mg = Grassmann(3, 2)
        pg = random_point(Mg)
        @test is_manifold_point(Mg, pg, true)
        Xg = random_tangent(Mg, pg)
        @test is_tangent_vector(Mg, pg, Xg, true; atol = 10^(-14))

        Mp = ProductManifold(Mg, Me)
        pp = random_point(Mp)
        @test is_manifold_point(Mp, pp, true)
        Xp = random_tangent(Mp, pp)
        @test is_tangent_vector(Mp, pp, Xp, true; atol = 10^(-15))

        Mp2 = PowerManifold(Me, NestedPowerRepresentation(), 4)
        pp2 = random_point(Mp2)
        @test is_manifold_point(Mp2, pp2)
        Xp2 = random_tangent(Mp2, pp2)
        @test is_tangent_vector(Mp2, pp2, Xp2)

        Mp3 = PowerManifold(Me, ArrayPowerRepresentation(), 4)
        pp3 = random_point(Mp3)
        @test is_manifold_point(Mp3, pp3)
        Xp3 = random_tangent(Mp3, pp3)
        @test is_tangent_vector(Mp3, pp3, Xp3)

        Mr = Rotations(3)
        pr = random_point(Mr)
        @test is_manifold_point(Mr, pr)
        Xr = random_tangent(Mr, pr)
        @test is_tangent_vector(Mr, pr, Xr)

        Mspd = SymmetricPositiveDefinite(3)
        pspd = random_point(Mspd)
        @test is_manifold_point(Mspd, pspd; atol = 10^(-14))
        Xspd = random_tangent(Mspd, pspd)
        @test is_tangent_vector(Mspd, pspd, Xspd, true; atol = 10^(-15))
        Xspd2 = random_tangent(Mspd, pspd, Val(:Rician))
        @test is_tangent_vector(Mspd, pspd, Xspd2, true; atol = 10^(-15))

        Mst = Stiefel(3, 2)
        pst = random_point(Mst)
        @test is_manifold_point(Mst, pst, true)

        Msp = Sphere(2)
        psp = random_point(Msp)
        @test is_manifold_point(Msp, psp, true)
        Xsp = random_tangent(Msp, psp)
        @test is_tangent_vector(Msp, psp, Xsp, true; atol = 10^(-15))

        Mh = Hyperbolic(2)
        ph = [0.0, 0.0, 1.0]
        Xh = random_tangent(Mh, ph)
        @test is_tangent_vector(Mh, ph, Xh, true)
    end
end
