@testset "Testting the Power manifold based on Circle" begin
    x = PowPoint( S1Point.([0.1,0.2,0.]) )
    Mi = Circle()
    M = Power(Mi,size(x.value))
    y = proxTV(M,0.05,x)
    (a,b) = proxTV(Circle(),0.05,(x[1],x[2]) )
    # b gets reused and hence changed
    (b2,c) = proxTV(Circle(),0.05,(b,x[3]) )
    @test y == PowPoint([a,b2,c])
    @test y != PowPoint([a,b,c])
end
