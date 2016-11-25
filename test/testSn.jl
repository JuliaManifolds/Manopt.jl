using Sn
@testset "Sphere Sn functions" begin
  p = Sn.SnPoint([1,0,0])
  q = Sn.SnPoint([0,1,0])
  r = Sn.SnPoint([0,0,1])
  xi = Sn.log(p,q)
  q2 = Sn.exp(p,xi)
  nu = Sn.log(p,r)
  @test_approx_eq_eps(norm(q.value-q2.value),0,10.0^(-16))
  @test Sn.distance(p,q) ≈ norm(xi.value)
  @test Sn.dot(xi,nu) ≈ 0
  # different base points throws an error
  @test_throws ErrorException Sn.dot(Sn.log(q,r,true),Sn.log(p,q,true))
end
