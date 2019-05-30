@testset "The Extended Decorator" begin
    M = Euclidean(2)
    x = RnPoint([1.,0.])
    ξ = RnTVector([0.,1.])
    xE = MPointE(x)
    ξE = TVectorE(ξ,xE)
    # check stripping for double decorate - check that we never decorate double
    @test MPointE(xE).base == x
    @test ξE.base == x
    @test getBasePoint(ξE) == xE #access methods promote
    ξEE = TVectorE(ξE,x)
    @test ξEE.base == x
    @test ξEE.vector == ξ
    ξEE2 = TVectorE(ξE,xE)
    @test ξEE.base == x
    @test ξEE.vector == ξ
    #
    promote_rule(MPointE{RnPoint{Float64}}, RnPoint{Float64}) == MPointE{RnPoint{Float64}}
    # convert MPoints
    @test convert(MPointE{RnPoint{Float64}},x) == xE
    @test_throws ErrorException convert(MPointE{SnPoint{Float64}},x)
    @test convert(MPointE{RnPoint{Float64}},xE) == xE
    @test convert(RnPoint{Float64},xE) == x
    @test_throws ErrorException convert(SnPoint{Float64},xE)
    @test convert(RnPoint{Float64},x) == x
    # convert TVecs
    @test_throws ErrorException convert(TVectorE{SnTVector{Float64},SnPoint{Float64}},ξE)
    @test convert(RnTVector{Float64}, ξE) == ξ
    @test_throws ErrorException convert(SnTVector{Float64},ξE)
    @test convert(TVectorE{RnTVector{Float64}, RnPoint{Float64}},ξE) == ξE
    @test convert(RnTVector{Float64},ξ) == ξ
    @test_throws ErrorException convert(TVectorE{RnTVector{Float64},RnPoint{Float64}},ξ)
    @test_throws ErrorException convert(TVectorE{SnTVector{Float64},SnPoint{Float64}},ξE)
    @test strip(ξE) == ξ
    @test getValue(ξE+ξE) == getValue(ξE)+getValue(ξE)
    @test_throws DomainError checkBasePoint(ξE, TVectorE(ξ,RnPoint([0., 1.]) ) )
    # promote
    @test promote(RnPoint(1.), MPointE(RnPoint(1.))) == (MPointE(RnPoint(1.)), MPointE(RnPoint(1.)))
    # getValue should be transparend
    @test getValue(xE) == getValue(x)
    @test getValue(ξE) == getValue(ξ)
    @test "$(xE)" == "$(x)E"
    @test "$(ξE)" == "$(ξ)E_$x"
    @test strip(x) == x
    @test strip(ξ) == ξ
    @test checkBasePoint(ξE,ξE)
    @test checkBasePoint(ξ,ξE)
    @test checkBasePoint(ξE,ξ)
    @test checkBasePoint(ξE,x)
    @test_throws DomainError checkBasePoint(ξE,RnPoint([0., 1.]))
    @test checkBasePoint(ξ,x)
    @test getValue(2. *ξE) == 2*getValue(ξ)
    @test getValue(ξE*2.) == getValue(ξ)*2
    @test getValue(ξE/2) == getValue(ξ)/2
    @test getValue(ξE/2) == getValue(ξ)/2
    @test getValue(-ξE) == -getValue(ξ)
    @test getValue(ξE-ξE) == getValue(TVectorE(ξ-ξ,x))
    @test getValue(+ξE) == getValue(ξ)
    @test ξE == ξE #check wether == for these types works
    # for the default functinos make sure they check and promote
    @test distance(M,xE,xE) == 0
    @test distance(M,x,xE) == 0
    @test distance(M,xE,x) == 0
    @test distance(M,x,MPointE(x,true)) == 0 # validate works
    @test_throws ErrorException distance( Sphere(2), SnPoint([1., 0., 0.]), MPointE( SnPoint([2., 0., 0.]),true) )
    # dot – all combinations
    @test dot(M,xE,ξE,ξE) == 1
    @test dot(M,xE,ξE,ξ) == 1
    @test dot(M,xE,ξ,ξE) == 1
    @test dot(M,xE,ξ,ξ) == 1
    @test dot(M,x,ξE,ξE) == 1
    @test dot(M,x,ξE,ξ) == 1
    @test dot(M,x,ξ,ξE) == 1
    @test dot(M,x,TVectorE(ξ,x,true),ξE) == 1
    # exp - all combinations
    y = RnPoint([1.,1.])
    yE = MPointE(y)
    @test exp(M,xE,ξE) == yE
    @test exp(M,MPointE(x,true),ξ) == yE
    @test exp(M,x,ξE) == yE
    # log
    @test log(M,x,yE) == ξE
    @test log(M,xE,y) == ξE
    @test log(M,xE,MPointE(y,true)) == ξE
    #
    @test manifoldDimension(xE) == 2
    #
    @test norm(M,x,ξE) == 1
    @test norm(M,xE,ξ) == 1
    @test norm(M,MPointE(x,true),ξE) == 1
    #
    ξEy = TVectorE(ξ,y)
    @test parallelTransport(M,xE,yE,ξE) == ξEy 
    @test parallelTransport(M,xE,y,ξE) == ξEy
    @test parallelTransport(M,x,yE,ξE) == ξEy
    @test parallelTransport(M,MPointE(x,true),yE,ξ) == ξEy
    @test parallelTransport(M,xE,y,ξ) == ξEy
    @test parallelTransport(M,x,yE,ξ) == ξEy
    #
    @test validateTVector(M,MPointE(x,true),randomTVector(M,MPointE(x,true)))
    #
    #    @test tangentONB(M,xE,y)
    #
    validateMPoint(M,xE)
    validateTVector(M,xE,ξE)
    validateTVector(M,x,ξE)
    validateTVector(M,xE,ξ)
    #
    A,b = tangentONB(M,x,y)
    AE = TVectorE.(A,Ref(x))
    AE1,b1 = tangentONB(M,xE,yE)
    @test all( [ (AE1 .== AE)... ,b1==b] )
    AE2,b2 = tangentONB(M,x,yE)
    @test all( [ (AE2 .== AE)... ,b2==b] )
    AE3,b3 = tangentONB(M,xE,y)
    @test all( [ (AE3 .== AE)... ,b3==b] )
    AE4, b4 = tangentONB(M,xE,ξE)
    @test all( [ (AE4 .== AE)... ,b4==b] )
    AE5, b5 = tangentONB(M,xE,ξ)
    @test all( [ (AE5 .== AE)... ,b5==b] )
    AE6, b6 = tangentONB(M,x,ξE)
    @test all( [ (AE6 .== AE)... ,b6==b] )
    @test zeroTVector(M,MPointE(x,true)) == TVectorE(RnTVector([0.,0.]),x) 
end