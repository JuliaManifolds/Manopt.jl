using LinearAlgebra, Manifolds, Manopt, Test

@testset "Difference of Convex" begin
    g(M, p) = log(det(p))^4 + 1 / 4
    grad_g(M, p) = 4 * (log(det(p)))^3 * p
    function grad_g!(M, X, p)
        copyto!(M, X, p)
        X .*= 4 * (log(det(p)))^3
        return X
    end
    h(M, p) = log(det(p))^2
    grad_h(M, p) = 2 * log(det(p)) * p
    function grad_h!(M, X, p)
        copyto!(M, X, p)
        X .*= 2 * (log(det(p)))
        return X
    end
    f(M, p) = g(M, p) - h(M, p)

    n = 2
    M = SymmetricPositiveDefinite(n)
    p0 = log(2) * Matrix{Float64}(I, n, n)
    p1 = difference_of_convex_algorithm(
        M, f, g, grad_h!, p0; grad_g=grad_g!, evaluation=InplaceEvaluation()
    )
    p2 = difference_of_convex_algorithm(
        M,
        f,
        g,
        grad_h!,
        p0;
        grad_g=grad_g!,
        sub_hess=nothing,
        evaluation=InplaceEvaluation(),
    )
    p3 = difference_of_convex_algorithm(M, f, g, grad_h, p0; grad_g=grad_g)
    p4 = difference_of_convex_algorithm(
        M, f, g, grad_h, p0; grad_g=grad_g, sub_hess=nothing
    )
    @test isapprox(M, p1, p2)
    @test isapprox(M, p2, p3)
    @test isapprox(M, p3, p4)
    @test f(M, p4) ≈ 0.0
    # not provided grad_g or problem nothing
    @test_throws ErrorException difference_of_convex_algorithm(
        M, f, g, grad_h, p0; sub_problem=nothing
    )
    @test_throws ErrorException difference_of_convex_algorithm(M, f, g, grad_h, p0)

    p5 = difference_of_convex_proximal_point(
        M, grad_h!, p0; g=g, grad_g=grad_g!, evaluation=InplaceEvaluation()
    )
    p6 = difference_of_convex_proximal_point(
        M,
        grad_h!,
        p0;
        g=g,
        grad_g=grad_g!,
        evaluation=InplaceEvaluation(),
        sub_hess=nothing,
    )
    @test isapprox(M, p4, p5)
    @test isapprox(M, p5, p6)
    @test f(M, p4) ≈ 0.0

    @test_throws ErrorException difference_of_convex_proximal_point(
        M, grad_h, p0; sub_problem=nothing
    )
    @test_throws ErrorException difference_of_convex_proximal_point(M, grad_h, p0)
    # we need both g and grad g here
    @test_throws ErrorException difference_of_convex_proximal_point(M, grad_h, p0; g=g)
    @test_throws ErrorException difference_of_convex_proximal_point(
        M, grad_h, p0, grad_g=grad_g
    )
end
