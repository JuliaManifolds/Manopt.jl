@testset "The 2-Sphere Sphere(2)" begin
  p = SnPoint([1,0,0])
  q = SnPoint([0,1,0])
  r = SnPoint([0,0,1])
  M = Sphere(2)
  ξ = log(M,p,q)
  q2 = exp(M,p,ξ)
  ν = log(M,p,r)
  # Test unary operator
	@test (-ν).value == -ν.value
	@test norm(q.value-q2.value) ≈ 0 atol=10.0^(-16)
  @test distance(M,p,q) ≈ norm(ξ.value) atol=10.0^(-16)
  @test dot(M,p,ξ,ν) ≈ 0 atol = 10.0^(-16)
  @test norm(mean(M,[p,q,r]).value-1/sqrt(3)*ones(3)) ≈ 0 atol=10.0^(-7)
  @test norm(mean(M,[p,q]).value-[1/sqrt(2),1/sqrt(2),0]) ≈ 0 atol=10.0^(-15)
  @test_throws ErrorException dot(M,log(M,q,r,true),log(M,p,q,true))
	#check that PT(q->p, log_qp) = -log_pq (unitary minus already checked)
	@test parallelTransport(M,q,p,log(M,q,p)) == -ξ
end
