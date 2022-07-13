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
    @testset "random" begin
        Mc = Circle()
        pc = random_point(Mc)
        @test is_point(Mc, pc, true)
        Xc = random_tangent(Mc, pc)
        @test is_vector(Mc, pc, Xc, true)

        Me = Euclidean(3)
        pe = random_point(Me)
        @test is_point(Me, pe, true)
        pe2 = random_point(Me, :Gaussian)
        @test is_point(Me, pe2, true)
        pe3 = random_point(Me, :Gaussian, 1.0)
        @test is_point(Me, pe3, true)
        Xe = random_tangent(Me, pe)
        @test is_vector(Me, pe, Xe, true)

        Mg = Grassmann(3, 2)
        pg = random_point(Mg)
        @test is_point(Mg, pg, true)
        Xg = random_tangent(Mg, pg)
        @test is_vector(Mg, pg, Xg, true; atol=1e-14)

        Mp = ProductManifold(Mg, Me)
        pp = random_point(Mp)
        @test is_point(Mp, pp, true)
        Xp = random_tangent(Mp, pp)
        @test is_vector(Mp, pp, Xp, true; atol=1e-14)

        Mp2 = PowerManifold(Me, NestedPowerRepresentation(), 4)
        pp2 = random_point(Mp2)
        @test is_point(Mp2, pp2)
        Xp2 = random_tangent(Mp2, pp2)
        @test is_vector(Mp2, pp2, Xp2)

        Mp3 = PowerManifold(Me, ArrayPowerRepresentation(), 4)
        pp3 = random_point(Mp3)
        @test is_point(Mp3, pp3)
        Xp3 = random_tangent(Mp3, pp3)
        @test is_vector(Mp3, pp3, Xp3)

        Mr = Rotations(3)
        pr = random_point(Mr)
        @test is_point(Mr, pr)
        Xr = random_tangent(Mr, pr)
        @test is_vector(Mr, pr, Xr)

        Mr2 = Rotations(1) # only one point, so there is no randomness, but well
        pr2 = random_point(Mr2)
        @test is_point(Mr2, pr2)
        @test is_vector(Mr2, pr2, random_tangent(Mr2, pr2))

        Mspd = SymmetricPositiveDefinite(3)
        pspd = random_point(Mspd)
        @test is_point(Mspd, pspd; atol=10^(-14))
        Xspd = random_tangent(Mspd, pspd)
        @test is_vector(Mspd, pspd, Xspd, true; atol=10^(-15))
        Xspd2 = random_tangent(Mspd, pspd, Val(:Rician))
        @test is_vector(Mspd, pspd, Xspd2, true; atol=10^(-15))

        Mst = Stiefel(3, 2)
        pst = random_point(Mst)
        @test is_point(Mst, pst, true)
        Xst = random_tangent(Mst, pst)
        @test is_vector(Mst, pst, Xst, true)

        Msp = Sphere(2)
        psp = random_point(Msp)
        @test is_point(Msp, psp, true)
        Xsp = random_tangent(Msp, psp)
        @test is_vector(Msp, psp, Xsp, true; atol=10^(-15))

        Mh = Hyperbolic(2)
        ph = [0.0, 0.0, 1.0]
        Xh = random_tangent(Mh, ph)
        @test is_vector(Mh, ph, Xh, true)

        MT = TangentBundle(Msp)
        pT = random_point(MT)
        @test is_point(MT, pT, true)
        XT = random_tangent(MT, pT)
        @test is_vector(MT, pT, XT, true; atol=10^(-15))

        Mfr = FixedRankMatrices(4, 6, 2)
        pfr = random_point(Mfr)
        @test is_point(Mfr, pfr, true)
        Xfr = random_tangent(Mfr, pfr)
        @test is_vector(Mfr, pfr, Xfr, true; atol=10^(-14))

        Mse1 = Euclidean(3)
        Mse2 = Rotations(3)
        Mse = SpecialEuclidean(3)
        pse = random_point(Mse)
        @test is_point(Mse1, pse.parts[1], true)
        @test is_point(Mse2, pse.parts[2], true)
        Xse = random_tangent(Mse, pse)
        @test is_vector(Mse1, pse.parts[1], Xse.parts[1], true)
        @test is_vector(Mse2, pse.parts[2], Xse.parts[2], true)
        pse2 = random_point(Mse, :Gaussian)
        @test is_point(Mse1, pse2.parts[1], true)
        @test is_point(Mse2, pse2.parts[2], true)
        Xse2 = random_tangent(Mse, pse2, :Gaussian)
        @test is_vector(Mse1, pse2.parts[1], Xse2.parts[1], true)
        @test is_vector(Mse2, pse2.parts[2], Xse2.parts[2], true)
    end
end
