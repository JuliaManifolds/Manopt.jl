using Manifolds, Manopt, Test, ManifoldsBase

@testset "Differentials" begin
    p = [1.,0.,0.]
    q = [0.,1.,0.]
    M = Sphere(2)
    X = log(M,p,q)
    @testset "Differentials on Sn(2)" begin
        @test differential_log_basepoint(M,p,p,X) == -X
        @test differential_log_basepoint(M,p,q,X) == -X
        @test differential_log_argument(M,p,p,X) == X
        @test differential_log_argument(M,p,q,X) == zero_tangent_vector(M,q)
        @test differential_exp_basepoint(M,p,zero_tangent_vector(M,p),X) == X
        @test norm(M,q, differential_exp_basepoint(M,p,X,X) - [-π/2, 0., 0.]) ≈ 0 atol=6*10^(-16)
        @test differential_exp_argument(M,p,zero_tangent_vector(M,p),X) == X
        @test norm(M,q,differential_exp_argument(M,p,X,zero_tangent_vector(M,p))) ≈ 0
        for t in [0,0.15,0.33,0.66,0.9]
            @test differential_geodesic_startpoint(M,p,p,t,X) == (1-t)*X
            @test norm(M,p,differential_geodesic_endpoint(M,p,p,t,X) - t*X)  ≈ 0 atol=10.0^(-16)
        end
    end
    @testset "Differentials on Power of Sn(2)" begin
        N = PowerManifold(M, NestedPowerRepresentation(), 3)
        x = [p,q,p]
        y = [p,p,q]
        V = [X, zero_tangent_vector(M,p), -X]
        @test norm(
                N,
                x,
                differential_forward_logs(N,x,V)
                - [-X, [π/2, 0., 0.],zero_tangent_vector(M,p)]
            ) ≈ 0 atol=8*10.0^(-16)
        @test differential_log_argument(N, x, y, V) == [ V[1], V[2], V[2] ]
    end
    @testset "Differentials on SPD(2)" begin
        #
        # Single differentials on Hn
        M2 = SymmetricPositiveDefinite(2)
        p2 = [1. 0.; 0. 1.]
        X2 = [0.5 1.;1. 0.5]
        q2 = exp(M2,p2,X2)
        # Text differentials (1) Dx of Log_xy
        @test norm(M2, p2, differential_log_basepoint(M2, p2, p2, X2) + X2) ≈ 0 atol=4*10^(-16)
        @test norm(M2, q2, differential_log_argument(M2, p2, q2, zero_tangent_vector(M2,p2))) ≈ 0 atol=4*10^(-16)
        @test norm(M2, p2, differential_exp_basepoint(M2, p2, zero_tangent_vector(M2,p2), X2) - X2) ≈ 0 atol=4*10^(-16)
        @test norm(M2, p2, differential_exp_argument(M2,p2,zero_tangent_vector(M2,p2),X2) - X2) ≈ 0 atol=4*10^(-16)
        for t in [0,0.15,0.33,0.66,0.9]
            @test norm(M2, p2, differential_geodesic_startpoint(M2, p2, p2, t, X2) - (1-t)*X2 ) ≈ 0 atol=4*10^(-16)
            @test norm(M2, p2, differential_geodesic_endpoint(M2, p2, p2, t, X2) - t*X2) ≈ 0 atol=4*10.0^(-16)
        end
        @test norm(M2, q2, differential_geodesic_startpoint(M2, p2, q2, 1., X2)) ≈ 0 atol=4*10.0^(-16)
        @test norm(M2, q2, differential_exp_basepoint(M2, p2, X2, zero_tangent_vector(M2,p2) )) ≈ 0 atol=4*10.0^(-16)
        @test norm(M2,q2,differential_exp_argument(M2,p2,X2,zero_tangent_vector(M2,p2))) ≈ 0 atol=4*10.0^(-16)
    end
    @testset "Differentials on Euclidean(2)" begin
        M3 = Euclidean(2)
        x3 = [1., 2.]
        ξ3 = [1.,0.]
        @test norm(M3, x3, differential_exp_basepoint(M3, x3, ξ3, ξ3) - ξ3) ≈ 0 atol=4*10.0^(-16)
    end
    @testset "Differentials on the Circle" begin
        M = Circle()
        p = 0
        q = π/4
        X = π/8
        @test differential_log_argument(M,p,q,X) == X
    end
end