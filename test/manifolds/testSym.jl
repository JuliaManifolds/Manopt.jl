@testset "The Symmetric Matrices" begin
    #
    # Despite checking both cases of Eucliean Manifolds, this can serve as a
    # prototype testcase suite for your manifold
    #
    M = Symmetric(2)
    v1 = [1. 0.;0. 1.]
    v2 = [2. 1.; 1. 2.]
    d1 = [1. 1.;1. 1.]
    d2 = [1. 0.; 0. 0.]
    
    x1 = SymPoint(v1)
    x2 = SymPoint(v2)
    ξ1 = SymTVector(d1)
    ξ2 = SymTVector(d2)

    @test getValue(x1) == v1
    @test getValue(ξ1) == d1

    @test distance(M,x1,x2) == norm(v1-v2)

    @test dot(M,x1,ξ1,ξ2) == dot(d1,d2)

    @test exp(M,x1, ξ1) == SymPoint(v1 + d1)
    @test exp(M,x2, ξ2) == SymPoint(v2 + d2)

    @test log(M,x1,x2) == SymTVector(v2-v1)   

    @test manifoldDimension(M) == 3
    @test manifoldDimension(M) == manifoldDimension(x1)

    @test norm(M,x1,ξ1) == norm(d1)  
    
    @test parallelTransport(M,x1,x2,ξ1) == ξ1

    @test validateMPoint(M,x1)
    @test_throws DomainError !validateMPoint(M, SymPoint([1. 0.; 1. 1.]))
    @test_throws DomainError !validateMPoint(M, SymPoint([1. 0. 0.; 0. 1. 0.; 0. 0. 1.]))
    @test validateTVector(M,x1,ξ1)
    @test_throws DomainError validateTVector(M,x1,SymTVector([1. 0.; 1. 1.]))
    @test_throws DomainError validateTVector(M,x1,SymTVector([1. 0. 0.; 0. 1. 0.; 0. 0. 1.]))

    @test typeofTVector(typeof(x1)) == SymTVector{Float64}
    @test typeofMPoint(typeof(ξ1)) == SymPoint{Float64}
    @test typeofTVector(x1) == SymTVector{Float64}
    @test typeofMPoint(ξ1) == SymPoint{Float64}

    @test typicalDistance(M) == sqrt(2)

    @test zeroTVector(M,x1) == SymTVector(zeros(2,2))
    
    @test "$M" == "The Manifold of 2-by-2 symmetric matrices."
    @test "$(x1)" == "Sym([1.0 0.0; 0.0 1.0])"
    @test "$(ξ1)" == "SymT([1.0 1.0; 1.0 1.0])"
end
