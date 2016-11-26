@testset "proxTV" begin
  #
  # proxTV
  p = SnPoint([1,0,0])
  q = SnPoint([0,1,0])
  (r1,r2) = proxTV(pi/4,(p,q));
  @test (abs(r1.value[1]-r2.value[1])<10.0^(-15) && abs(r1.value[2]-r2.value[2])<10.0^(-15) && abs(r1.value[3]-r2.value[3])<10.0^(-15) )
  # i.e. they are moved together
  @test distance(r1,r2) < 10.0^(-16)
  (r3,r4) = proxTV(pi/8,(p,q));
  # they cross correlate
  @test (abs(r3.value[1]-r4.value[2])<10.0^(-15) && abs(r3.value[2]-r4.value[1])<10.0^(-15) && abs(r3.value[3]-r4.value[3])<10.0^(-15) )
  @test distance(r3,r4) == pi/4 # and have moved half their distance
end
