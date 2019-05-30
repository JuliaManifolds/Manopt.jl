@testset "Test the 2-by-2 symmetric positive definite matrices" begin
  # explicitly compute an easy exp
  M = SymmetricPositiveDefinite(2)
  x = SPDPoint([1.0 0.0; 0.0 1.0])
  ξ = SPDTVector([1.0 0.0; 0.0 0.0])
  η = SPDTVector([1.0 0.0; 0.0 1.0])
  y = exp(M,x,ξ)
  @test norm( getValue(y) - [ℯ 0;0 1.0]) ≈ 0 atol=10.0^(-16)
  # check that with base the base must mach.
	z = SPDPoint([1.0 0.0; 0.0 0.1])
	ξE = TVectorE(ξ,x);
	@test_throws DomainError exp(M,z,ξE)
  # check that log is the inverse of exp and stores the base point correctly
	# if that's activated
	xT = MPointE(x);
	ξT2 = log(M,x,y)
	@test norm( getValue(ξ) - getValue(ξT2) ) ≈ 0 atol=10.0^(-16)
	@test distance(M, getBasePoint(ξE), x) ≈ 0 atol=10.0^(-16)
	# test parallel transport
	@test norm(getValue( parallelTransport(M,y,x,log(M,y,x)) ) + getValue(ξ) ) ≈ 0 atol=3*10.0^(-16)
	#
	# Test tangent ONB
	#
	n = 4
	M2 = SymmetricPositiveDefinite(n)
	x2 = SPDPoint(one(zeros(n,n)))
	ξ2 = SPDTVector(getValue(x2))
	Ξ,κ = tangentONB(M2,x2,ξ2)
	# test orthogonality
	@test all([ dot(M,x2,Ξ[i],Ξ[j]) for i=1:length(Ξ) for j=i+1:length(Ξ) ] .== 0)
	# test normality
    @test all( [1-dot(M,x2,Ξ[i],Ξ[i]) for i=1:length(Ξ)] .< 3*10^(-16) )
    #
    # Test Matrix trait
    @test getValue(x+y) == getValue(x)+getValue(y)
    @test getValue(x-y) == getValue(x)-getValue(y)
    @test getValue(x*y) == getValue(x)*getValue(y)
    @test getValue(ξ+η) == getValue(ξ)+getValue(η) 
    @test getValue(ξ-η) == getValue(ξ)-getValue(η) 
    @test getValue(ξ*η) == getValue(ξ)*getValue(η)
    @test transpose(x) == SPDPoint(Matrix(transpose(getValue(x))))
    @test transpose(ξ) == SPDTVector(Matrix(transpose(getValue(ξ))))
end