@testset "The Hyperbolic Manifold" begin
    M = Hyperbolic(2)
    @test typicalDistance(M) == sqrt(2)
    @test getValue(HnPoint(1.)) == 1.
    @test getValue(HnTVector(0.)) == 0.
    x = HnPoint([0.,0.,1])
    @test getValue(x) == [0., 0., 1.]
    ξ = HnTVector([1.,1.,0,])
    @test getValue(ξ) == [1., 1., 0.]
    @test_throws ErrorException validateMPoint(M, HnPoint([1.,0.]))
    @test_throws ErrorException validateMPoint(M, HnPoint([0., 0., 0.]))
    @test validateTVector(M, x, ξ)
    @test_throws ErrorException validateTVector(M,x, HnTVector([0.,0.])) # Dimensions don't agree
    @test_throws ErrorException validateTVector(M,x, HnTVector(getValue(x))) # Not orthogonal
    y = exp(M, x, ξ)
    @test distance(M, x, y) ≈ norm(M, x, ξ)
    @test norm(M, x, ξ) ≈ sqrt(dot(M,x,ξ,ξ))
    @test distance(M, exp(M,x,log(M,x,y)), y) ≈ 0
    @test distance(M, exp(M,x,zeroTVector(M,x)), x) ≈ 0
    @test norm(M, x, log(M,x,x)) ≈ 0
    #
    @test manifoldDimension(x) == 2
    @test manifoldDimension(M) == 2
    @test parallelTransport(M,x,x,ξ) == ξ
    @test norm(M, y, parallelTransport(M,x,y,ξ)+log(M,y,x)) ≈ 0 atol=5*10^(-15)
    #
    @test typeofTVector(typeof(x)) == HnTVector{Float64}
    @test typeofMPoint(typeof(ξ)) == HnPoint{Float64}
    @test typeofTVector(x) == HnTVector{Float64}
    @test typeofMPoint(ξ) == HnPoint{Float64}

    @test "$(M)" == "The 2-Hyperbolic Space."
    @test "$(x)" == "Hn([0.0, 0.0, 1.0])"
    @test "$(ξ)" == "HnT([1.0, 1.0, 0.0])"
end