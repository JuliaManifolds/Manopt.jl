@testset "proximal maps" begin
  #
  # proxTV
  p = SnPoint([1,0,0])
  q = SnPoint([0,1,0])
  M = Sphere(2)
  (r,s) = proxTV(M,π/4,(p,q));
  @test norm( getValue(r) - getValue(s) ) < eps(Float64)
  # i.e. they are moved together
  @test distance(M,r,s) < eps(Float64)
  (t,u) = proxTV(M,π/8,(p,q));
  # they cross correlate
  @test ( abs(t.value[1]-u.value[2])< eps(Float64) && abs(t.value[2]-u.value[1]) < eps(Float64) && abs(t.value[3]-u.value[3])< eps(Float64) )
  @test distance(M,t,u) == π/4 # and have moved half their distance
end
