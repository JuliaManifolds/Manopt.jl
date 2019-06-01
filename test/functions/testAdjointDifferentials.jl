@testset "Differentials (on Sphere(2))" begin
# The Adjoint Differentials test using the same variables as the differentials
# test
x = SnPoint([1.,0.,0.])
y = SnPoint([0.,1.,0.])
M = Sphere(2)
ξ = log(M,x,y)
# Text differentials (1) Dx of Log_xy
@test AdjDxLog(M,x,x,ξ) == -ξ
@test AdjDyLog(M,x,x,ξ) == ξ
@test AdjDxExp(M,x,zeroTVector(M,x),ξ) == ξ
@test AdjDξExp(M,x,zeroTVector(M,x),ξ) == ξ
for t in [0,0.15,0.33,0.66,0.9]
    @test AdjDxGeo(M,x,x,t,ξ) == (1-t)*ξ
    @test norm(M,x,AdjDyGeo(M,x,x,t,ξ) - t*ξ)  ≈ 0 atol=10.0^(-16)
end
Mp = Power(M,2)
xP = PowPoint([x,y,x])
yP = PowPoint([x,x,y])
ξP = PowTVector([ξ, zeroTVector(M,x), -ξ])
@test norm(Mp,xP, AdjDforwardLogs(Mp,xP,ξP)
     - PowTVector([-ξ, zeroTVector(M,x),zeroTVector(M,x)])  )  ≈ 0 atol=3*10.0^(-16)
end