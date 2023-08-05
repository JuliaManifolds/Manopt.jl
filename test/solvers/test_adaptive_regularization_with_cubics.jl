using Manopt, Manifolds, Test, Random
using LinearAlgebra: I, tr

@testset "Adaptive Reguilarization with Cubics" begin
    @testset "A solver run" begin
        Random.seed!(42)
        n = 8
        k = 3
        A_init = randn(n, n)
        A = (A_init + A_init') / 2

        M = Grassmann(n, k)

        f(M, p) = -0.5 * tr(p' * A * p)
        grad_f(M, p) = -A * p + p * (p' * A * p)
        Hess_f(M, p, X) = -A * X + p * p' * A * X + X * p' * A * p

        p0 = Matrix{Float64}(I, n, n)[:, 1:k]
        p1 = adaptive_regularization_with_cubics(
            M, f, grad_f, Hess_f, p0; θ=0.5, σ=100.0, retraction_method=PolarRetraction()
        )
        p2 = copy(M, p0)
        adaptive_regularization_with_cubics!(
            M, f, grad_f, Hess_f, p2; θ=0.5, σ=100.0, retraction_method=PolarRetraction()
        )

        @test isapprox(M, p1, p2)
    end
end
