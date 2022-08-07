using Manopt, Manifolds, ManifoldsBase, LinearAlgebra
import Random: seed!

@testset "Riemannian Trust-Region with Hessian update" begin
    n = 100

    A = randn(n, n)
    A = (A + A') / 2
    cost(::Sphere, p::Array{Float64,1}) = p' * A * p
    grad(::Sphere, p::Array{Float64,1}) = 2 * (A * p - p * p' * A * p)
    function hess(::Sphere, p::Array{Float64,1}, X::Array{Float64,1})
        return 2 * (A * X - p * p' * A * X - X * p' * A * p - p * p' * X * p' * A * p)
    end
    M = Sphere(n - 1)
    x = random_point(M)

    ev = eigvecs(A)[:, 1]

    @testset "Allocating Variant" begin
        X = trust_regions(M, cost, grad, hess, x)
        @test norm(abs.(X) - abs.(ev)) ≈ 0 atol = 1e-14
        X2 = deepcopy(x)
        trust_regions!(M, cost, grad, hess, X2)
        @test norm(abs.(X2) - abs.(ev)) ≈ 0 atol = 1e-14
        @test isapprox(M, X, X2)

        XaHSR1 = trust_regions(
            M,
            cost,
            grad,
            ApproxHessianSymmetricRankOne(M, x, grad; nu=eps(Float64)^2),
            x;
            stopping_criterion=StopWhenAny(
                StopAfterIteration(10000), StopWhenGradientNormLess(10^(-6))
            ),
            trust_region_radius=1.0,
            θ=0.1,
            κ=0.9,
            retraction_method=ProjectionRetraction(),
        )

        @test norm(abs.(XaHSR1) - abs.(ev)) ≈ 0 atol = 1e-7
        @test cost(M, XaHSR1) ≈ cost(M, X)

        XaHSR1_2 = deepcopy(x)

        trust_regions!(
            M,
            cost,
            grad,
            ApproxHessianSymmetricRankOne(M, XaHSR1_2, grad; nu=eps(Float64)^2),
            XaHSR1_2;
            stopping_criterion=StopWhenAny(
                StopAfterIteration(10000), StopWhenGradientNormLess(10^(-6))
            ),
            trust_region_radius=1.0,
            θ=0.1,
            κ=0.9,
            retraction_method=ProjectionRetraction(),
        )

        @test norm(abs.(XaHSR1_2) - abs.(ev)) ≈ 0 atol = 1e-7
        @test cost(M, XaHSR1_2) ≈ cost(M, X)
        @test isapprox(M, XaHSR1, XaHSR1_2; atol=1e-6)

        XaHBFGS = trust_regions(
            M,
            cost,
            grad,
            ApproxHessianBFGS(M, x, grad),
            x;
            stopping_criterion=StopWhenAny(
                StopAfterIteration(10000), StopWhenGradientNormLess(10^(-6))
            ),
            trust_region_radius=1.0,
            θ=0.1,
            κ=0.9,
            retraction_method=ProjectionRetraction(),
        )

        @test norm(abs.(XaHBFGS) - abs.(ev)) ≈ 0 atol = 1e-7
        @test cost(M, XaHBFGS) ≈ cost(M, X)

        XaHBFGS_2 = deepcopy(x)

        trust_regions!(
            M,
            cost,
            grad,
            ApproxHessianBFGS(M, XaHBFGS_2, grad),
            XaHBFGS_2;
            stopping_criterion=StopWhenAny(
                StopAfterIteration(10000), StopWhenGradientNormLess(10^(-6))
            ),
            trust_region_radius=1.0,
            θ=0.1,
            κ=0.9,
            retraction_method=ProjectionRetraction(),
        )

        @test norm(abs.(XaHBFGS_2) - abs.(ev)) ≈ 0 atol = 1e-7
        @test cost(M, XaHBFGS_2) ≈ cost(M, X)
        @test isapprox(M, XaHBFGS, XaHBFGS_2; atol=1e-6)

        
    end
end
