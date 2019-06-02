@testset "The Graph Manifold" begin
    M1 = Euclidean(3)
    v1 = [1., 0., 0.]
    v2 = [0., 2., 0.]
    d1 = [0., 1., 1.]
    d2 = [1., 0., 1.]
    
    x1 = RnPoint(v1)
    x2 = RnPoint(v2)
    ξ1 = RnTVector(d1)
    ξ2 = RnTVector(d2)

    A = [0 1;1 0]
    M = Graph(M1,A)
    x = GraphVertexPoint([x1,x2])
    y = GraphVertexPoint([x2,x1])
    ξ = GraphVertexTVector([ξ1,ξ2])
    η = GraphVertexTVector([ξ2,ξ1])

    @test getValue(x) == [x1,x2]
    @test getValue(y) == [x2,x1]
    @test getValue(ξ) == [ξ1,ξ2]
    @test getValue(η) == [ξ2,ξ1]

    @test distance(M,x,y) == sqrt(2)*distance(M1,x1,x2)
    @test dot(M,x,ξ,η) == dot(M1,x1,ξ1,ξ2) + dot(M1,x2,ξ2,ξ1)
    @test exp(M,x,ξ) == GraphVertexPoint([exp(M1,x1,ξ1),exp(M1,x2,ξ2)])
    @test log(M,x,y) == GraphVertexTVector([log(M1,x1,x2),log(M1,x2,x1)])
    
    @test manifoldDimension(M) == 6
    @test manifoldDimension(x) == 6

    @test norm(M,x,ξ) == sqrt(dot(M,x,ξ,ξ))
    
    # parallelTransport
    # random
    # validate
    
    @test typicalDistance(M) ≈ sqrt(6) atol=5*10^(-16)

    @test zeroTVector(M,x) == GraphVertexTVector([
        zeroTVector(M1,x1), zeroTVector(M1,x2)
    ])
    
    @test "$M" == "The manifold on vertices and edges of a graph of The 3-dimensional Euclidean space of (vertex manifold) dimension 6."
    @test "$x" == "GraphVertexV[Rn([1.0, 0.0, 0.0]), Rn([0.0, 2.0, 0.0])]"
    @test "$ξ" == "GraphVertexT[RnT([0.0, 1.0, 1.0]), RnT([1.0, 0.0, 1.0])]"
end
