@testset "Differentials on Sn(2) and SPD(2)" begin
x = [1.,0.,0.]
y = [0.,1.,0.]
M = Sphere(2)
ξ = log(M,x,y)
# Text differentials (1) Dx of Log_xy
@test DqLog(M,x,x,ξ) == -ξ
@test DqLog(M,x,y,ξ) == -ξ
@test DyLog(M,x,x,ξ) == ξ
@test DyLog(M,x,y,ξ) == zero_tangent_vector(M,y)
@test DpExp(M,x,zero_tangent_vector(M,x),ξ) == ξ
@test norm(M,y, DpExp(M,x,ξ,ξ) - [-π/2, 0., 0.]) ≈ 0 atol=2*10^(-16)
@test DξExp(M,x,zero_tangent_vector(M,x),ξ) == ξ
@test norm(M,y,DξExp(M,x,ξ,zero_tangent_vector(M,x))) ≈ 0
for t in [0,0.15,0.33,0.66,0.9]
    @test DpGeo(M,x,x,t,ξ) == (1-t)*ξ
    @test norm(M,x,DqGeo(M,x,x,t,ξ) - t*ξ)  ≈ 0 atol=10.0^(-16)
end
Mp = PowerManifold(M,2)
xP = [x,y,x]
yP = [x,x,y]
ξP = [ξ, zero_tangent_vector(M,x), -ξ]
@test norm(
        Mp,
        xP,
        DforwardLogs(Mp,xP,ξP)
     - [-ξ, [π/2, 0., 0.],zero_tangent_vector(M,x)] )  ≈ 0 atol=3*10.0^(-16)
#
# Single differentials on Hn
M2 = SymmetricPositiveDefinite(2)
x2 = SPDPoint([1. 0.; 0. 1.])
ξ2 = SPDTVector([0.5 1.;1. 0.5])
y2 = exp(M2,x2,ξ2)
# Text differentials (1) Dx of Log_xy
@test norm(M2, x2, DqLog(M2, x2, x2, ξ2) + ξ2) ≈ 0 atol=4*10^(-16)
@test norm(M2, y2, DqLog(M2, x2, y2, ξ2) + ξ2) ≈ 0 atol=4*10^(-16)
@test norm(M2, x2, DyLog(M2, x2, x2, ξ2) - ξ2) ≈ 0 atol=4*10^(-16)
@test norm(M2, y2, DyLog(M2, x2, y2, zero_tangent_vector(M2,x2))) ≈ 0 atol=4*10^(-16)
@test norm(M2, x2, DpExp(M2, x2, zero_tangent_vector(M2,x2), ξ2) - ξ2) ≈ 0 atol=4*10^(-16)
@test norm(M2, x2, DξExp(M2,x2,zero_tangent_vector(M2,x2),ξ2) - ξ2) ≈ 0 atol=4*10^(-16)
for t in [0,0.15,0.33,0.66,0.9]
    @test norm(M2, x2, DpGeo(M2, x2, x2, t, ξ2) - (1-t)*ξ2 ) ≈ 0 atol=4*10^(-16)
    @test norm(M2, x2, DqGeo(M2, x2, x2, t, ξ2) - t*ξ2) ≈ 0 atol=4*10.0^(-16)
end
@test norm(M2, y2, DpGeo(M2, x2, y2, 1., ξ2)) ≈ 0 atol=4*10.0^(-16)
@test norm(M2, y2, DpExp(M2, x2, ξ2, zero_tangent_vector(M2,x2) )) ≈ 0 atol=4*10.0^(-16)
@test norm(M2,y2,DξExp(M2,x2,ξ2,zero_tangent_vector(M2,x2))) ≈ 0 atol=4*10.0^(-16)
end
#
# And a final Rn
#
M3 = Euclidean(2)
x3 = [1., 2.]
ξ3 = [1.,0.]
@test norm(M3, x3, DpExp(M3, x3, ξ3, ξ3) - ξ3) ≈ 0 atol=4*10.0^(-16)