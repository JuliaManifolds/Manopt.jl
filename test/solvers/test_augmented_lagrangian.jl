using Manopt, ManifoldsBase, Manifolds, Test
using LinearAlgebra: I, tr

@testset "Test RALM with a nonneg. PCA" begin
    d = 20
    M = Sphere(d - 1)
    S = [ones(4)..., zeros(d - 4)...]
    v0 = project(M, S)
    Z = v0 * v0'
    F(M, p) = -tr(transpose(p) * Z * p) / 2
    gradF(M, p) = project(M, p, -transpose.(Z) * p / 2 - Z * p / 2)
    G(M, p) = -p # i.e. p ≥ 0
    mI = -Matrix{Float64}(I, d, d)
    gradG(M, p) = [project(M, p, mI[:, i]) for i in 1:d]
    x0 = project(M, ones(d))
    sol = augmented_Lagrangian_method(M, F, gradF; G=G, gradG=gradG, x=x0)
    @test distance(M, sol, v0) < 8 * 1e-4
end
