@testset "Test the 2-by-2 symmetric positive definite matrices" begin
  M = SymmetricPositiveDefinite(2)
  p = SPDPoint([1.0 0.0; 0.0 1.0])
  xi = SPDTVector([1.0 0.0; 0.0 0.0],p)
	q = exp(M,p,xi)
  @test norm(q.value-[e 0;0 1.0]) ≈ 0 atol=10.0^(-16)
	p2 = SPDPoint([1.0 0.0; 0.0 0.1])
	xi2 = SPDTVector([1.0 0.0; 0.0 0.0],p)
  @test_throws ErrorException exp(M,p2,xi2)
end
