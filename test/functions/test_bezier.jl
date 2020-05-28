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
    @testset "Spherical Data" begin
        M = Sphere(2)
        B = artificial_S2_composite_bezier_curve()
        @test de_casteljau(M, B, [0.0,1.0,2.0,3.0]) ≈ [B[1].pts[1], B[2].pts[1], B[3].pts[1], B[3].pts[4]]
        @test get_bezier_junction_tangent_vectors(M,B) ≈ [
            log(M,B[1].pts[1], B[1].pts[2]), log(M,B[1].pts[4], B[1].pts[3]),
            log(M,B[2].pts[1], B[2].pts[2]), log(M,B[2].pts[4], B[2].pts[3]),
            log(M,B[3].pts[1], B[3].pts[2]), log(M,B[3].pts[4], B[3].pts[3]),
        ]
        @test get_bezier_junction_tangent_vectors(M,B[1]) ≈ [
            log(M,B[1].pts[1], B[1].pts[2]), log(M,B[1].pts[4], B[1].pts[3]),
        ]
        @test get_bezier_junctions(M, B[1]) == B[1].pts[[1,end]]
        @test get_bezier_inner_points(M,B) == [B[1].pts[2],B[1].pts[3],B[2].pts[2],B[2].pts[3],B[3].pts[2],B[3].pts[3]]
        @test get_bezier_inner_points(M,B[1]) == [B[1].pts[2],B[1].pts[3]]

        @test get_bezier_points(M,B) == cat([[b.pts...] for b in B]...,dims=1)
        @test get_bezier_points(M,B, :continuous) == cat([[b.pts[[1:3]...]...] for b in B]...,[B[3].pts[4]],dims=1)
        @test get_bezier_points(M,B, :differentiable) == cat([B[1].pts[[1,2]]...],[b.pts[[3,4]] for b in B]...,dims=1)
        @test get_bezier_points(M,B[1]) == B[1].pts
        # for segments just check that they
        d = get_bezier_degrees(M,B)
        A = get_bezier_segments(M, get_bezier_points(M,B),d)
        @test [A[i].pts for i ∈ 1:3] == [B[i].pts for i ∈ 1:3]
        A = get_bezier_segments(M, get_bezier_points(M,B,:continuous),d, :continuous)
        @test [A[i].pts for i ∈ 1:3] == [B[i].pts for i ∈ 1:3]
        A = get_bezier_segments(M, get_bezier_points(M,B,:differentiable),d, :differentiable)
        @test [A[i].pts for i ∈ 1:3] == [B[i].pts for i ∈ 1:3]

        @test_throws ErrorException adjoint_differential_bezier_control(M,B,7.0,zero_tangent_vector(M,B[1].pts[1]))
    end
end
