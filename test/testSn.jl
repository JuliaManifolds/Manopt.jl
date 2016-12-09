@testset "Sphere Sn" begin
  p = SnPoint([1,0,0])
  q = SnPoint([0,1,0])
  r = SnPoint([0,0,1])
  ξ = log(p,q)
  q2 = exp(p,ξ)
  ν = log(p,r)
  @test_approx_eq_eps(norm(q.value-q2.value),0,10.0^(-16))
  @test distance(p,q) ≈ norm(ξ.value)
  @test dot(ξ,ν) ≈ 0
  @test_approx_eq_eps(norm(mean([p,q,r]).value-1/sqrt(3)*ones(3)),0,10.0^(-7))
  @test_approx_eq_eps(norm(mean([p,q]).value-[1/sqrt(2),1/sqrt(2),0]),0,10.0^(-15))
  @test_throws ErrorException dot(log(q,r,true),log(p,q,true))
end
