using Manifolds, Manopt, Test, ManifoldsBase, RecursiveArrayTools
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
        # Without being too far away -> classical mid point
        @test mid_point(M, 0, 0.1, π / 2) == mid_point(M, 0, 0.1)
        # the same also works for the 1-element-array representation of `Circle`
        @test mid_point(M, [0.0], [Float64(π)], [1.0]) ≈ [π / 2]
        @test mid_point(M, [0.0], [Float64(π)], [-1.0]) ≈ [-π / 2]
        # the in-place variant has to honour the reference point as well
        y = [0.0]
        @test mid_point!(M, y, [0.0], [Float64(π)], [1.0]) ≈ [π / 2]
        @test mid_point!(M, y, [0.0], [Float64(π)], [-1.0]) ≈ [-π / 2]
    end
    @testset "max_stepsize" begin
        M = Sphere(2)
        TM = TangentBundle(M)
        TTM = TangentBundle(TM)

        R3 = Euclidean(3)
        TR3 = TangentBundle(R3)

        p = [0.0, 1.0, 0.0]
        X = [0.0, 0.0, 0.0]

        @test Manopt.max_stepsize(M) == π
        @test Manopt.max_stepsize(M, p) == π
        @test Manopt.max_stepsize(TM) == π
        @test Manopt.max_stepsize(TM, ArrayPartition(p, X)) == π
        @test Manopt.max_stepsize(
            TTM, ArrayPartition(ArrayPartition(p, X), ArrayPartition(X, X))
        ) == π

        @test Manopt.max_stepsize(R3, p) == Inf
        @test Manopt.max_stepsize(TR3, ArrayPartition(p, X)) == Inf

        S_R3 = ProductManifold(M, R3)
        @test Manopt.max_stepsize(S_R3) ≈ π
        @test Manopt.max_stepsize(S_R3, ArrayPartition(p, [0.0, 0.0, 0.0])) ≈ π

        S_pow = PowerManifold(M, NestedPowerRepresentation(), 3)
        @test Manopt.max_stepsize(S_pow) ≈ π
        @test Manopt.max_stepsize(S_pow, [p, p, p]) ≈ π

        # in a direction
        Y = [1.0, 0.0, 1.0]
        @test Manopt.max_stepsize(M, p, Y) ≈ π / norm(Y)
        @test Manopt.max_stepsize(M, p, X) == Inf
        @test Manopt.max_stepsize(R3, p, Y) == Inf
        # each component is moved by its component of the direction
        @test Manopt.max_stepsize(S_R3, ArrayPartition(p, X), ArrayPartition(Y, [10.0, 0.0, 0.0])) ≈ π / norm(Y)
        @test Manopt.max_stepsize(S_R3, ArrayPartition(p, X), ArrayPartition(X, Y)) == Inf
        @test Manopt.max_stepsize(S_pow, [p, p, p], [Y, 2 .* Y, X]) ≈ π / (2 * norm(Y))
        @test Manopt.max_stepsize(S_pow, [p, p, p], [X, X, X]) == Inf

        Mfr = FixedRankMatrices(5, 4, 2)
        pfr = SVDMPoint(
            [
                -0.42232620708727264 0.18201829740358394
                0.12501774539665936 -0.5154706413711303
                -0.049666735478056216 0.34842538478418505
                -0.6185683735016354 -0.6581006457016838
                -0.6487815663485731 0.38296559722742113
            ],
            [0.9240871604723607, 0.488958057530268],
            [
                -0.29372512377826565 0.9509130294099503 -0.09069249144195571 0.035564506968098104
                -0.87290474583133 -0.29359416630673274 -0.3315532983520648 -0.20472464572606347
            ],
        )
        @test Manopt.max_stepsize(Mfr, pfr) == manifold_dimension(Mfr)

        M = Hyperrectangle([-3, -1.5], [3, 1.5])
        @test Manopt.max_stepsize(M) ≈ 6.0
        @test Manopt.max_stepsize(M, [-1, 0.5]) ≈ 4.0
        # in a direction: the step size up to the first bound met
        q = [-1.0, 0.5]
        @test Manopt.max_stepsize(M, q, [1.0, 1.0]) ≈ 1.0
        @test Manopt.max_stepsize(M, q, [-1.0, 0.0]) ≈ 2.0
        @test Manopt.max_stepsize(M, q, [-1.0, -4.0]) ≈ 0.5
        @test Manopt.max_stepsize(M, q, [0.0, 0.0]) == Inf
        # elements pointing out of the box at their bound do not limit the step
        @test Manopt.max_stepsize(M, [3.0, 0.5], [1.0, -0.5]) ≈ 4.0
        @test Manopt.max_stepsize(M, [3.0, 0.5], [1.0, 0.0]) == Inf
        @test Manopt.max_stepsize(M, [3.0, 1.5], [-1.0, 0.0]) ≈ 6.0
        # infinite bounds
        Mh = Hyperrectangle([0.0, -Inf], [Inf, Inf])
        @test Manopt.max_stepsize(Mh) == Inf
        @test Manopt.max_stepsize(Mh, [1.0, 0.0], [1.0, -1.0]) == Inf
        @test Manopt.max_stepsize(Mh, [1.0, 0.0], [-0.5, 1.0]) ≈ 2.0
        @test Manopt.max_stepsize(Hyperrectangle(Float32[0, 0], Float32[1, 1]), Float32[0.5, 0.5], Float32[1, 0]) isa Float32
        # with a box component
        N = ProductManifold(M, Sphere(2))
        @test Manopt.max_stepsize(N, ArrayPartition(q, p), ArrayPartition([0.1, 0.1], Y)) ≈ π / norm(Y)
        @test Manopt.max_stepsize(N, ArrayPartition(q, p), ArrayPartition([1.0, 1.0], Y)) ≈ 1.0
        @test Manopt.max_stepsize(N, ArrayPartition(q, p), ArrayPartition([0.0, 0.0], X)) == Inf

        M = ProbabilitySimplex(3)
        @test Manopt.max_stepsize(M) == 1.0

        Mspd = SymmetricPositiveDefinite(2)
        pspd = [2.0 0.0; 0.0 0.5]
        @test Manopt.max_stepsize(Mspd, pspd) == log(floatmax(Float64))
        @test Manopt.max_stepsize(Mspd) == Manopt.max_stepsize(Mspd, pspd)
    end
    @testset "Vector space default" begin
        @test Manopt.Rn(Val(:Manopt), 3) isa ManifoldsBase.DefaultManifold
        @test Manopt.Rn(Val(:Manifolds), 3) isa Euclidean
    end
end
