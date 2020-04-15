@testset "gradient TV." begin
    M = Circle()
    N = PowerManifold(M, 4)
    x = [0.1,0.2,0.3,0.5]
    tvTestξ =  [-1.0,0.,0.,1.]
    @test ∇TV(N,x) == tvTestξ
    @test ∇TV(M, (x[1],x[1])) == (zero_tangent_vector(M,x[1]), zero_tangent_vector(M,x[1]))
    @test norm(N,x, ∇TV(N,x,2) - tvTestξ) ≈ 0
    tv2Testξ = [0.,.5,-1.,0.5]
    @test ∇TV2(N,x) == tv2Testξ
    @test norm(N,x, forward_logs(N,x) - [0.1, 0.1, 0.2, 0.]) ≈ 0 atol=10^(-16)
    @test norm(N, x, ∇intrinsic_infimal_convolution_TV12(N,x,x,x,1.,1.)[1] - [-1., 0., 0., 1.] ) ≈ 0
    @test norm(N, x, ∇intrinsic_infimal_convolution_TV12(N,x,x,x,1.,1.)[2]) ≈ 0
    x2 = [0.1,0.2,0.3]
    N2 = PowerManifold(M,size(x2)...)
    @test ∇TV2(N2,x2) == zeros(3)
    @test ∇TV2(N2,x2,2) == zeros(3)
    @test ∇TV(M, (0., 0.),2) == (0., 0.)
    # 2d forward logs
    N3 = PowerManifold(M,2,2)
    N3C = PowerManifold(M,2,2,2)
    x3 = [0.1 0.2;0.3 0.5]
    x3C = cat(x3, x3; dims=3)
    tC = cat([.2 .3; 0. 0.], [.1 0.; .2 0.] ; dims=3)
    @test norm(N3C, x3C, forward_logs(N3,x3)-tC) ≈ 0 atol=10^(-16)
end
