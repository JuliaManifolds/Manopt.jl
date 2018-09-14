@testset "gradient descent..." begin
    M = Circle()
    N = Power(M,(4,))
    x = PowPoint(S1Point.([0.1,0.2,0.3,0.5]))
    tvTestx = PowTVector( S1TVector.([-1.0,0.,0.,1.]) )
    @test gradTV(N,x) == tvTestx
    gradTV2(N,x)
end
