using Manopt, ManifoldsBase, Manifolds, Test
using LinearAlgebra: I, tr

@testset "Test REPM with a nonneg. PCA" begin
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
    sol_lse = exact_penalty_method(M, F, gradF, x0; G=G, gradG=gradG)
    sol_lqh = exact_penalty_method(
        M, F, gradF, x0; G=G, gradG=gradG, smoothing=LinearQuadraticHuber()
    )
    @test distance(M, v0, sol_lse) < 8 * 1e-4
    @test distance(M, v0, sol_lqh) < 8 * 1e-4
end
