using Manopt, Manifolds, Test

@testset "Riemannian quasi-Newton with corners" begin
    M = Hyperrectangle([-1.0, 2.0, -3.0], [1.0, 4.0, 9.0])
    f1(::Hyperrectangle, p) = p[1]^2 + 2 * p[2]^2 + p[1] * p[3]
end
