@testset "Test the 2-by-2 symmetric positive definite matrices" begin
  # explicitly compute an easy exp
	M = SymmetricPositiveDefinite(2)
  p = SPDPoint([1.0 0.0; 0.0 1.0])
  ξ = SPDTVector([1.0 0.0; 0.0 0.0],p)
	q = exp(M,p,ξ)
  @test norm(q.value-[e 0;0 1.0]) ≈ 0 atol=10.0^(-16)
  # check that with base the base must mach.
	p2 = SPDPoint([1.0 0.0; 0.0 0.1])
	xi2 = SPDTVector([1.0 0.0; 0.0 0.0],p)
	@test_throws ErrorException exp(M,p2,xi2)
  # check that log is the inverse of exp and stores the base point correctly
	# if that's activated
	ξCheck = log(M,p,q,false)
	@test norm(ξ.value-ξCheck.value) ≈ 0 atol=10.0^(-16)
	@test isnull(ξCheck.base)
	ξCheck = log(M,p,q,true)
	@test norm(get(ξCheck.base).value-p.value) ≈ 0 atol=10.0^(-16)
	# test parallel transport
	@test norm(parallelTransport(M,q,p,log(M,q,p)).value + ξ.value) ≈ 0 atol=3*10.0^(-16)
end
