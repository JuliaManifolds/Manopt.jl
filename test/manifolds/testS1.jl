@testset "Test Circle" begin
    M = Circle()
    x = S1Point(0.)
    y = opposite(M,x)
    z = randomMPoint(M)
    η = randomTVector(M,x)
    @test getValue(x) == 0.
    ξ = S1TVector(2*π/3)
    @test getValue(ξ) == 2*π/3
    @test distance(M, x, exp(M,x,ξ)) ≈ 2*π/3
    @test dot(M,x,ξ, S1TVector(0.5)) ≈ π/3
    @test embed(M,x) == SnPoint([1., 0.])
    @test exp(M,x,ξ) == S1Point(2*π/3)
    @test distance(M, exp(M,x,2*ξ), S1Point(-2*π/3) ) ≈ 0 atol=10^(-15)
    @test log(M,x,exp(M,x,ξ)) == ξ
    @test manifoldDimension(x) == 1
    @test manifoldDimension(M) == 1
    @test norm(M,x,ξ) ≈ 2*π/3
    @test distance(M, opposite(M,S1Point(π/2)), S1Point(-π/2) ) ≈ 0 atol=10^(-15)
    @test parallelTransport(M,x,S1Point(π/2),ξ) == ξ
    @test validateMPoint(M,randomMPoint(M))
    @test validateTVector(M,x,randomTVector(M,x))
    @test typeofTVector(typeof(x)) == S1TVector
    @test typeofMPoint(typeof(ξ)) == S1Point
    @test typeofTVector(x) == S1TVector
    @test typeofMPoint(ξ) == S1Point
    @test typicalDistance(M) == π/2
    @test zeroTVector(M,x) == S1TVector(0.)
    @test "$M" == "The manifold S1 consisting of angles"
    @test "$x" == "S1(0.0)"
    @test "$(S1TVector(0.))" == "S1T(0.0)"
    @test_throws ErrorException validateMPoint(M,S1Point(2*π))
    @test_throws ErrorException validateMPoint(M,S1Point(-2*π))

    # Test manifoldDimension
    @test manifoldDimension(M) == 1
    @test manifoldDimension(x) == 1
    # Test norm
    @test norm(M,x,ξ) == 2*π/3
    # Test opposite
    @test getValue(y) - pi ≈ 0 atol = 10.0^(-16)
    # Test parallelTransport
    @test getValue(parallelTransport(M,x,y,ξ)) - getValue(ξ) ≈ 0 atol = 10.0^(-16)
    # Test typicalDistance
    @test typicalDistance(M) == pi/2
    # Test validateMPoint and validateTVector
    xnot = S1Point(3.15)
    @test_throws ErrorException validateMPoint(M,xnot)
end
