@testset "Differentials (on Sphere(2))" begin
  x = SnPoint([1.,0.,0.])
  y = SnPoint([0.,1.,0.])
  M = Sphere(2)
  ξ = log(M,x,y)
  # Text differentials (1) Dx of Log_xy
  @test DxLog(M,x,x,ξ) == -ξ
  @test DyLog(M,x,x,ξ) == ξ
  @test DxExp(M,x,zeroTVector(M,x),ξ) == ξ
  @test DξExp(M,x,zeroTVector(M,x),ξ) == ξ
  for t in [0,0.15,0.33,0.66,0.9]
        @test DxGeo(M,x,x,t,ξ) == (1-t)*ξ
        @test norm(M,x,DyGeo(M,x,x,t,ξ) - t*ξ)  ≈ 0 atol=10.0^(-16)
  end
end