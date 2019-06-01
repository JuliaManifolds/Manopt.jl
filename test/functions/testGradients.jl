@testset "gradient TV." begin
    M = Circle()
    N = Power(M,(4,))
    x = PowPoint(S1Point.([0.1,0.2,0.3,0.5]))
    tvTestξ = PowTVector( S1TVector.([-1.0,0.,0.,1.]) )
    @test gradTV(N,x) == tvTestξ
    @test gradTV(M, (x[1],x[1])) == (zeroTVector(M,x[1]), zeroTVector(M,x[1]))
    @test norm(N,x, gradTV(N,x,2) - tvTestξ) ≈ 0
    tv2Testξ = PowTVector(S1TVector.([0.,.5,-1.,0.5]))
    @test gradTV2(N,x) == tv2Testξ
    @test norm(N,x, forwardLogs(N,x)- PowTVector(S1TVector.([0.1, 0.1, 0.2, 0.]))) ≈ 0 atol=10^(-16)
    x2 = PowPoint(S1Point.([0.1,0.2,0.3]))
    N2 = Power(M,size(getValue(x2)))
    @test gradTV2(N2,x2) == PowTVector(S1TVector.(zeros(3)))
    @test gradTV2(N2,x2,2) == PowTVector(S1TVector.(zeros(3)))
    @test norm(N, x,
        gradIntrICTV12(N,x,x,x,1.,1.)[1]
        - PowTVector([S1TVector(-1.),S1TVector(0.), S1TVector(0.), S1TVector(1.)])#
        ) ≈ 0
    @test norm(N, x, gradIntrICTV12(N,x,x,x,1.,1.)[2] - zeroTVector(N,x)) ≈ 0
    @test gradTV(M, (S1Point(0.), S1Point(0.)),2) == (S1TVector(0.), S1TVector(0.))
    # 2d forward forwardLogs
    N3 = Power(M,(2,2))
    N3C = Power(M, (2,2,2) )
    x3 = PowPoint(S1Point.( [0.1 0.2;0.3 0.5] ))
    x3C = PowPoint( cat(getValue(x3), getValue(x3); dims=3) )
    tC = PowTVector(S1TVector.(cat([.2 .3; 0. 0.], [.1 0.; .2 0.] ; dims=3)))
    @test norm(N3C, x3C, forwardLogs(N3,x3)-tC) ≈ 0 atol=10^(-16)
end
