using Manopt, Manifolds, Test

@testset "Bezier Tests" begin
    @testset "General Bezier Tests" begin
        repr(BezierSegment([[0.0,0.0],[0.0,0.0]])) == "BezierSegment([[0.0, 0.0], [0.0, 0.0]])"
    end
    @testset "Spherical Test" begin
        M = Sphere(2)
        pC = [0.0, 1.0, 0.0]
        pT = exp(M, pC, [0.0, 0.0, 0.7])
        pB = exp(M, pC, [0.0, 0.0, -0.7])
        B = [
            BezierSegment(shortest_geodesic(M, pT, pC, [0.0, 1/3, 2/3, 1.0])),
            BezierSegment(shortest_geodesic(M, pC, pB, [0.0, 1/3, 2/3, 1.0])),
        ]
        # this is equispaced, so the pure cost is zero and the gradient is a zero-vector
        t = collect(range(0.0,1.0,length=5))
        pts = shortest_geodesic(M, pT, pB, t)
        pts2 = de_casteljau(M,B,2 .* t)
        @test sum(distance.(Ref(M),pts,pts2)) ≈ 0

        #
        @test sum(
                norm.(
                   ∇acceleration_bezier(M, B[1], collect(range(0.0,1.0,length=20))).pts .-
                   [[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]
                )
            ) ≈ 0 atol=10^(-12)

        # cost and gradient
        T = collect(range(0.0,2.0,length=51))
        degrees = get_bezier_degrees(M,B)
        Bvec = get_bezier_points(M, B, :differentiable)
        Mp = PowerManifold(M, NestedPowerRepresentation(), length(Bvec))
        @test cost_acceleration_bezier(M, Bvec, degrees, T) ≈ 0 atol=10^-10
        z = zero_tangent_vector(Mp,Bvec)
        distance(Mp, ∇acceleration_bezier(M, Bvec, degrees, T), z)
        @test norm(Mp, Bvec, ∇acceleration_bezier(M, Bvec, degrees, T) - z) ≈ 0 atol=10^(-12)

        d = [pT, exp(M,pC, [0.3, 0.0,0.0]), pB]
        λ = 3.0

        # cost and gradient with data term
        @test cost_L2_acceleration_bezier(M, Bvec, degrees, T, λ, [pT, pC, pB]) ≈ 0 atol=10^(-10)
        @test cost_L2_acceleration_bezier(M, Bvec, degrees, T, λ, d) ≈ λ/2*distance(M,d[2],pC).^2
        # when the data are the junctions
        @test norm(Mp, Bvec, ∇L2_acceleration_bezier(M, Bvec, degrees, T, λ, [pT, pC, pB]) - z) ≈ 0 atol=10^(-12)
        z[4][1] = -0.9
        @test norm(Mp, Bvec, ∇L2_acceleration_bezier(M, Bvec, degrees, T, λ, d) - z) ≈ 0 atol=10^(-12)
        # when the data is weighted with zero
        @test cost_L2_acceleration_bezier(M, Bvec, degrees, T, 0.0, d) ≈ 0 atol=10^(-10)
        z[4][1] = 0.0
        @test norm(Mp, Bvec, ∇L2_acceleration_bezier(M, Bvec, degrees, T, 0.0, d) - z) ≈ 0 atol=10^(-12)
    end
end
