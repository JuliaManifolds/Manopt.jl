@testset "gradient descent..." begin
    M = Circle()
    N = Power(M,(4,))
    x = PowPoint(S1Point.([0.1,0.2,0.3,0.5]))
    gradTV2(N,x)
    tvTestξ = PowTVector( S1TVector.([-1.0,0.,0.,1.]) )
    @test gradTV(N,x) == tvTestξ
    tv2Testξ = PowTVector(S1TVector.([0.,-.5,-1.,-0.5]))
    @test gradTV2(N,x) == tv2Testξ
    x2 = PowPoint(S1Point.([0.1,0.2,0.3]))
    N2 = Power(M,size(getValue(x2)))
    @test gradTV2(N2,x2) == PowTVector(S1TVector.(zeros(3)))
end
