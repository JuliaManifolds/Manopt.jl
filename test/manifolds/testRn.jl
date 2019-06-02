@testset "The Euclidean Space" begin
    #
    # Despite checking both cases of Eucliean Manifolds, this can serve as a
    # prototype testcase suite for your manifold
    #
    M = Euclidean(3)
    N = Euclidean(1)
    v1 = [1., 0., 0.]
    v2 = [0., 2., 0.]
    d1 = [0., 1., 1.]
    d2 = [1., 0., 1.]
    w1 = 2.
    w2 = 4.
    e1 = -1.
    e2 =  2.
    
    x1 = RnPoint(v1)
    x2 = RnPoint(v2)
    ξ1 = RnTVector(d1)
    ξ2 = RnTVector(d2)
    y1 = RnPoint(w1)
    y2 = RnPoint(w2)
    η1 = RnTVector(e1)
    η2 = RnTVector(e2)

    @test getValue(x1) == v1
    @test getValue(y1) == w1
    @test getValue(ξ1) == d1
    @test getValue(η1) == e1

    @test distance(M,x1,x2) == norm(v1-v2)
    @test distance(N,y1,y2) == abs(w1-w2)

    @test dot(M,x1,ξ1,ξ2) == dot(d1,d2)
    @test dot(N,y1,η1,η2) == dot(e1,e2)

    @test exp(M,x1, ξ1) == RnPoint(v1 + d1)
    @test exp(M,x2, ξ2) == RnPoint(v2 + d2)
    @test exp(N,y1, η1) == RnPoint(w1 + e1)
    @test exp(N,y2, η2) == RnPoint(w2 + e2)

    @test log(M,x1,x2) == RnTVector(v2-v1)   
    @test log(N,y1,y2) == RnTVector(w2-w1)

    @test manifoldDimension(M) == 3
    @test manifoldDimension(M) == manifoldDimension(x1)
    @test manifoldDimension(N) == 1
    @test manifoldDimension(N) == manifoldDimension(y1)

    @test norm(M,x1,ξ1) == norm(d1)  
    @test norm(N,y1,η1) == abs(e1)
    
    @test parallelTransport(M,x1,x2,ξ1) == ξ1
    @test parallelTransport(N,y1,y2,η1) == η1

    @test validateMPoint(M,randomMPoint(M))
    @test validateMPoint(N,randomMPoint(N))
    @test validateTVector(M,x1,randomTVector(M,x1))
    @test validateTVector(N,y1,randomTVector(N,y1))
    @test !validateMPoint(M,y1)
    @test !validateMPoint(N,x1)
    @test !validateTVector(M,x1,η1)
    @test !validateTVector(N,y1,ξ1)

    @test tangentONB(M,x1,x2) == ([RnTVector([1.,0.,0.]),RnTVector([0.,1.,0.]),RnTVector([0.,0.,1.])], zeros(3))
    @test tangentONB(M,y1,y2) == ([RnTVector(1.),] , [0.,] )

    @test typeofTVector(typeof(x1)) == RnTVector{Float64}
    @test typeofMPoint(typeof(ξ1)) == RnPoint{Float64}
    @test typeofTVector(x1) == RnTVector{Float64}
    @test typeofMPoint(ξ1) == RnPoint{Float64}

    @test typicalDistance(M) == sqrt(3)

    @test zeroTVector(M,x1) == RnTVector(zeros(3))
    @test zeroTVector(N,y1) == RnTVector(zeros(1))
    
    @test "$(M)" == "The 3-dimensional Euclidean space"
    @test "$(x1)" == "Rn([1.0, 0.0, 0.0])"
    @test "$(ξ1)" == "RnT([0.0, 1.0, 1.0])"
end
