@testset "Differentials on Sn(2) and SPD(2)" begin
p = [1.,0.,0.]
q = [0.,1.,0.]
M = Sphere(2)
X = log(M,p,q)
# Text differentials (1) Dx of Log_xy
@test DqLog(M,p,p,X) == -X
@test DqLog(M,p,q,X) == -X
@test DyLog(M,p,p,X) == X
@test DyLog(M,p,q,X) == zero_tangent_vector(M,q)
@test DpExp(M,p,zero_tangent_vector(M,p),X) == X
@test norm(M,q, DpExp(M,p,X,X) - [-π/2, 0., 0.]) ≈ 0 atol=6*10^(-16)
@test DξExp(M,p,zero_tangent_vector(M,p),X) == X
@test norm(M,q,DξExp(M,p,X,zero_tangent_vector(M,p))) ≈ 0
for t in [0,0.15,0.33,0.66,0.9]
    @test DpGeo(M,p,p,t,X) == (1-t)*X
    @test norm(M,p,DqGeo(M,p,p,t,X) - t*X)  ≈ 0 atol=10.0^(-16)
end
N = PowerManifold(M, NestedPowerRepresentation(), 3)
x = [p,q,p]
y = [p,p,q]
V = [X, zero_tangent_vector(M,p), -X]
@test norm(
        N,
        x,
        DforwardLogs(N,x,V)
     - [-X, [π/2, 0., 0.],zero_tangent_vector(M,p)] )  ≈ 0 atol=8*10.0^(-16)
#
# Single differentials on Hn
M2 = SymmetricPositiveDefinite(2)
p2 = [1. 0.; 0. 1.]
X2 = [0.5 1.;1. 0.5]
q2 = exp(M2,p2,X2)
# Text differentials (1) Dx of Log_xy
@test norm(M2, p2, DqLog(M2, p2, p2, X2) + X2) ≈ 0 atol=4*10^(-16)
@test norm(M2, q2, DqLog(M2, p2, q2, X2) + X2) ≈ 0 atol=4*10^(-16)
@test norm(M2, p2, DyLog(M2, p2, p2, X2) - X2) ≈ 0 atol=4*10^(-16)
@test norm(M2, q2, DyLog(M2, p2, q2, zero_tangent_vector(M2,p2))) ≈ 0 atol=4*10^(-16)
@test norm(M2, p2, DpExp(M2, p2, zero_tangent_vector(M2,p2), X2) - X2) ≈ 0 atol=4*10^(-16)
@test norm(M2, p2, DξExp(M2,p2,zero_tangent_vector(M2,p2),X2) - X2) ≈ 0 atol=4*10^(-16)
for t in [0,0.15,0.33,0.66,0.9]
    @test norm(M2, p2, DpGeo(M2, p2, p2, t, X2) - (1-t)*X2 ) ≈ 0 atol=4*10^(-16)
    @test norm(M2, p2, DqGeo(M2, p2, p2, t, X2) - t*X2) ≈ 0 atol=4*10.0^(-16)
end
@test norm(M2, q2, DpGeo(M2, p2, q2, 1., X2)) ≈ 0 atol=4*10.0^(-16)
@test norm(M2, q2, DpExp(M2, p2, X2, zero_tangent_vector(M2,p2) )) ≈ 0 atol=4*10.0^(-16)
@test norm(M2,q2,DξExp(M2,p2,X2,zero_tangent_vector(M2,p2))) ≈ 0 atol=4*10.0^(-16)
end
#
# And a final Rn
#
M3 = Euclidean(2)
x3 = [1., 2.]
ξ3 = [1.,0.]
@test norm(M3, x3, DpExp(M3, x3, ξ3, ξ3) - ξ3) ≈ 0 atol=4*10.0^(-16)