using Manopt, ManifoldsBase, Manifolds, Test
using LinearAlgebra: I, tr

@testset "Test REPM with a nonneg. PCA" begin
    d = 20
    M = Sphere(d - 1)
    S = [ones(4)..., zeros(d - 4)...]
    v0 = project(M, S)
    Z = v0 * v0'
    f(M, p) = -tr(transpose(p) * Z * p) / 2
    gradF(M, p) = project(M, p, -transpose.(Z) * p / 2 - Z * p / 2)
    g(M, p) = -p # i.e. p ≥ 0
    mI = -Matrix{Float64}(I, d, d)
    grad_g(M, p) = [project(M, p, mI[:, i]) for i in 1:d]
    x0 = project(M, ones(d))
    sol_lse = exact_penalty_method(M, f, grad_f, x0; G=g, gradG=grad_g)
    sol_lqh = exact_penalty_method(
        M, f, grad_f, x0; G=g, gradG=grad_g, smoothing=LinearQuadraticHuber()
    )
    @test distance(M, v0, sol_lse) < 1e-3
    @test distance(M, v0, sol_lqh) < 1e-3
    # Dummy options
    O = ExactPenaltyMethodState(M, x0, CostProblem(M, f), NelderMeadState(M))
    set_iterate!(O, 2 .* x0)
    @test get_iterate(O) == 2 .* x0
end
