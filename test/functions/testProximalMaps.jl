@testset "proximal maps" begin
  #
  # proxTV
  p = [1.,0.,0.]
  q = [0.,1.,0.]
  M = Sphere(2)
  N = PowerManifold(M, NestedPowerRepresentation(), 2)
  @test_throws ErrorException proxDistance(M, 1., p, q, 3)
  @test distance(M, proxDistance(M, distance(M,p,q)/2, p,q,1), shortest_geodesic(M, p, q, 0.5)) ≈ 0
  (r,s) = proxTV(M,π/4,(p,q))
  @test norm( r - s ) < eps(Float64)
  # i.e. they are moved together
  @test distance(M,r,s) < eps(Float64)
  (t,u) = proxTV(M,π/8,(p,q));
  @test_throws ErrorException proxTV(M, π, (p,q), 3)
  # they cross correlate
  @test ( abs(t[1]-u[2])< eps(Float64) && abs(t[2]-u[1]) < eps(Float64) && abs(t[3]-u[3])< eps(Float64) )
  @test distance(M,t,u) == π/4 # and have moved half their distance
  #
  (v,w) = proxTV(M,1.,(p,q),2)
  vC, wC = shortest_geodesic(M, p, q, [1/3, 2/3])
  @test distance(M, v, vC) ≈ 0
  @test distance(M, w, wC) ≈ 0
  # proxTV on Power
  T = proxTV(N, π/8, [p,q])
  @test distance(N, T, [t, u]) ≈ 0
  # parallelproxTV
  N2 = PowerManifold(M, NestedPowerRepresentation(), 3)
  r = geodesic(M, p, q, 0.5)
  s, t = proxTV(M, π/16, (r, q) )
  u, v = proxTV(M, π/16, (p, r) )
  y = proxParallelTV(N2, π/16, [[p,r,q], [p,r,q]])
  @test distance(N2, y[1], [p,s,t]) ≈ 0 # even indices in first comp
  @test distance(N2, y[2], [u,v,q]) ≈ 0 # odd in second
  # dimensions of x have to fit, here they don't
  @test_throws ErrorException proxParallelTV(N2, π/16, [[p,r,q]])
  # proxTV2
  p2, r2, q2 = proxTV2(M,1., (p,r,q) )
  sum(distance.(Ref(M), [p,r,q], [p2, r2, q2] )) ≈ 0
  @test_throws ErrorException proxTV2(M, 1., (p,r,q), 2) # since proxTV is only defined for p=1
  distance(PowerManifold(M,NestedPowerRepresentation(), 3), [p2,r2,q2], proxTV2(PowerManifold(M, NestedPowerRepresentation(), 3),1., [p,r,q])) ≈ 0
  # Circle
  M2 = Circle()
  N2 = PowerManifold(M2,3)
  pS, rS, qS = [-0.5, 0.1, 0.5]
  d = dot([pS, rS, qS], [1., -2., 1.])
  m = min(0.3, abs( sym_rem(d)/6) )
  s = sign(sym_rem(d))
  pSc, rSc, qSc = sym_rem.( [pS, rS, qS] .- m .* s .* [1., -2., 1.] )
  pSr, rSr, qSr = proxTV2(M2, 0.3, (pS, rS, qS) )
  @test sum( distance.(Ref(M2), [pSc, rSc, qSc], [pSr, rSr, qSr]) ) ≈ 0
  # p=2
  t = 0.3*sym_rem(d)/(1+0.3*6.)
  @test sum(
    distance.( Ref(M2),
      [proxTV2(M2,0.3,(pS, rS, qS),2)...],
      [pS, rS, qS] .- t.*[1., -2., 1.]
  )) ≈ 0
  # others fail
  @test_throws ErrorException proxTV2(M2, 0.3, (pS, rS, qS),3)
  # Rn
  M3 = Euclidean(1)
  pR, rR, qR = [pS, rS, qS]
  m = min.( Ref(0.3), abs.( [pR, rR, qR].*[1., -2., 1.] )/6 )
  s = sign(d)  # we can reuse d
  pRc, rRc, qRc = [pR, rR, qR]  .- m .* s .* [1., -2., 1.]
  pRr, rRr, qRr = proxTV2(M3, 0.3, (pR, rR, qR) )
  @test sum( distance.(Ref(M3), [pRc, rRc, qRc], [pRr, rRr, qRr]) ) ≈ 0
  # p=2
  t = 0.3*d/(1+0.3*6.)
  @test sum(
    distance.( Ref(M3),
      [proxTV2(M3, 0.3, (pR, rR, qR), 2)...],
      [pR, rR, qR] .- t.*[1., -2., 1.]
  )) ≈ 0
  # others fail
  @test_throws ErrorException proxTV2(M3, 0.3, (pR, rR, qR),3)
  #
  # collaborative integer tests
  #
  @test_throws ErrorException proxTV2(M3, 0.3, (pS, rS, qS),3)
  ξR, ηR, νR = [pS, rS, qS]
  N3 = PowerManifold(M3,3)
  P = [pR rR qR]
  Ξ = [ξR ηR νR]
  @test proxCollaborativeTV(N3,0.,P, Ξ,1, 1) == Ξ
  @test proxCollaborativeTV(N3,0.,P, Ξ,1., 1) == Ξ
  @test proxCollaborativeTV(N3,0.,P, Ξ,1, 1.) == Ξ
  @test proxCollaborativeTV(N3,0.,P, Ξ,1., 1.) == Ξ

  @test proxCollaborativeTV(N3,0.,P, Ξ, 2, 1) == Ξ
  @test norm(N3,P, proxCollaborativeTV(N3,0.,P, Ξ, 2, Inf)) ≈ 0
  @test norm(N3,P, proxCollaborativeTV(N3,0.,P, Ξ, 1, Inf)) ≈ 0
  @test norm(N3,P, proxCollaborativeTV(N3,0.,P, Ξ, Inf, Inf)) ≈ 0
  @test_throws ErrorException proxCollaborativeTV(N3,0.,P, Ξ, 3,3)
  @test_throws ErrorException proxCollaborativeTV(N3,0.,P, Ξ, 3,1)
  @test_throws ErrorException proxCollaborativeTV(N3,0.,P, Ξ, 3,Inf)
end
