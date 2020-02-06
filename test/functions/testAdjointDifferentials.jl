@testset "Differentials (on Sphere(2))" begin
# The Adjoint Differentials test using the same variables as the differentials
# test
x = [1.,0.,0.]
y = [0.,1.,0.]
M = Sphere(2)
ξ = log(M,x,y)
# Text differentials (1) Dx of Log_xy
@test AdjDpLog(M,x,x,ξ) == -ξ
@test AdjDqLog(M,x,x,ξ) == ξ
@test AdjDpExp(M,x,zero_tangent_vector(M,x),ξ) == ξ
@test AdjDpExp(M,x,zero_tangent_vector(M,x),ξ) == ξ
for t in [0,0.15,0.33,0.66,0.9]
    @test AdjDpGeo(M,x,x,t,ξ) == (1-t)*ξ
    @test norm(M,x,AdjDqGeo(M,x,x,t,ξ) - t*ξ)  ≈ 0 atol=10.0^(-16)
end
Mp = PowerManifold(M,2)
xP = [x,y,x]
yP = [x,x,y]
ξP = [ξ, zero_tangent_vector(M,x), -ξ]
@test norm(Mp,xP, AdjDforwardLogs(Mp,xP,ξP)
     - [-ξ, zero_tangent_vector(M,x),zero_tangent_vector(M,x)])  )  ≈ 0 atol=3*10.0^(-16
end