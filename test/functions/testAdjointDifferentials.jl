@testset "Differentials (on Sphere(2))" begin
# The Adjoint Differentials test using the same variables as the differentials
# test
x = [1.,0.,0.]
y = [0.,1.,0.]
M = Sphere(2)
ξ = log(M,x,y)
# Text differentials (1) Dx of Log_xy
@test adjoint_differential_log_basepoint(M,x,x,ξ) == -ξ
@test adjoint_differential_log_argument(M,x,x,ξ) == ξ
@test adjoint_differential_exp_basepoint(M,x,zero_tangent_vector(M,x),ξ) == ξ
@test adjoint_differential_exp_argument(M,x,zero_tangent_vector(M,x),ξ) == ξ
for t in [0,0.15,0.33,0.66,0.9]
    @test adjoint_differential_geodesic_startpoint(M,x,x,t,ξ) == (1-t)*ξ
    @test norm(M,x,adjoint_differential_geodesic_endpoint(M,x,x,t,ξ) - t*ξ)  ≈ 0 atol=10.0^(-16)
end
Mp = PowerManifold(M, NestedPowerRepresentation(), 3)
xP = [x, y, x]
yP = [x, x, y]
ξP = [ξ, zero_tangent_vector(M,x), -ξ]
@test norm(
        Mp,
        xP,
        adjoint_differential_forward_logs(Mp,xP,ξP) - [-ξ, zero_tangent_vector(M,x), zero_tangent_vector(M,x)]
    ) ≈ 0 atol=4*10.0^(-16)
end
