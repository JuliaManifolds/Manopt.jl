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
  # mean
  @test_approx_eq_eps(norm(mean([p,q,r]).value-1/sqrt(3)*ones(3)),0,10.0^(-7))
  @test_approx_eq_eps(norm(mean([p,q]).value-[1/sqrt(2),1/sqrt(2),0]),0,10.0^(-15))
end
