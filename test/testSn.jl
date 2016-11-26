@testset "Sphere Sn" begin
  p = SnPoint([1,0,0])
  q = SnPoint([0,1,0])
  r = SnPoint([0,0,1])
  xi = log(p,q)
  q2 = exp(p,xi)
  nu = log(p,r)
  @test_approx_eq_eps(norm(q.value-q2.value),0,10.0^(-16))
  @test distance(p,q) ≈ norm(xi.value)
  @test dot(xi,nu) ≈ 0
  # different base points throws an error
  @test_throws ErrorException dot(log(q,r,true),log(p,q,true))
end
