@testset "The 2-Sphere Sphere(2)" begin
  x = SnPoint([1,0,0])
  y = SnPoint([0,1,0])
  z = SnPoint([0,0,1])
  M = Sphere(2)
  ξ = log(M,x,y)
  y2 = exp(M,x,ξ)
  ν = log(M,x,z)
  # Test unary operator
	@test getValue( (-ν) ) == getValue(-ν)
	@test norm( getValue(y) - getValue(y2) ) ≈ 0 atol=10.0^(-16)
  @test norm(M,x,ξ) ≈ norm( getValue(ξ) ) atol=10.0^(-16)
  @test distance(M,x,y) ≈ norm(M,x,ξ) atol=10.0^(-16)
  @test dot(M,x,ξ,ν) ≈ 0 atol = 10.0^(-16)
  @test norm( getValue( mean(M,[x,y,z]) ) - 1/sqrt(3)*ones(3)) ≈ 0 atol=10.0^(-7)
  @test norm( getValue( mean(M,[x,y]) ) - [1/sqrt(2),1/sqrt(2),0] ) ≈ 0 atol=10.0^(-15)
  # Test extended
  xT = MPointE(x); yT = MPointE(y); zT = MPointE(z);
  @test_throws ErrorException dot(M,xT,log(M,xT,zT),log(M,yT,zT) )
  @test dot(M,x,log(M,x,z),log(M,x,y) ) ≈ 0 atol=10.0^(-15)
	#check that PT(q->p, log_qp) = -log_pq (unitary minus already checked)
	@test parallelTransport(M,y,x,log(M,y,x)) == -ξ
  # Text differentials (1) Dx of Log_xy
  @test DxLog(M,x,x,ξ) == -ξ
  @test DyLog(M,x,x,ξ) == ξ
  @test DxExp(M,x,zeroTVector(M,x),ξ) == ξ
  @test DξExp(M,x,zeroTVector(M,x),ξ) == ξ
  for t in [0,0.15,0.33,0.66,0.9]
	  @test DxGeo(M,x,x,t,ξ) == (1-t)*ξ
	  @test DyGeo(M,x,x,t,ξ) == t*ξ
  end
end
