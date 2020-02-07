@testset "gradient TV." begin
    M = Circle()
    N = PowerManifold(M, 4)
    x = [0.1,0.2,0.3,0.5]
    tvTestξ =  [-1.0,0.,0.,1.]
    @test gradTV(N,x) == tvTestξ
    @test gradTV(M, (x[1],x[1])) == (zero_tangent_vector(M,x[1]), zero_tangent_vector(M,x[1]))
    @test norm(N,x, gradTV(N,x,2) - tvTestξ) ≈ 0
    tv2Testξ = [0.,.5,-1.,0.5]
    @test gradTV2(N,x) == tv2Testξ
    @test norm(N,x, forwardLogs(N,x) - [0.1, 0.1, 0.2, 0.]) ≈ 0 atol=10^(-16)
    x2 = [0.1,0.2,0.3]
    N2 = PowerManifold(M,size(x2)...)
    @test gradTV2(N2,x2) == zeros(3)
    @test gradTV2(N2,x2,2) == zeros(3)
    @test norm(N, x, gradIntrICTV12(N,x,x,x,1.,1.)[1] - [-1., 0., 0., 1.] ) ≈ 0
    @test norm(N, x, gradIntrICTV12(N,x,x,x,1.,1.)[2] - zero_tangent_vector(N,x)) ≈ 0
    @test gradTV(M, (0., 0.),2) == (0., 0.)
    # 2d forward forwardLogs
    N3 = PowerManifold(M,2,2)
    N3C = PowerManifold(M,2,2,2)
    x3 = [0.1 0.2;0.3 0.5]
    x3C = cat(x3, x3; dims=3)
    tC = cat([.2 .3; 0. 0.], [.1 0.; .2 0.] ; dims=3)
    @test norm(N3C, x3C, forwardLogs(N3,x3)-tC) ≈ 0 atol=10^(-16)
end
