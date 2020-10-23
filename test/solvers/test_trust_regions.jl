A = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]

function cost(X::ProductRepr)
    return cost([submanifold_components(X)...])
end
function cost(X::Array{Matrix{Float64},1})
    return -0.5 * norm(transpose(X[1]) * A * X[2])^2
end

function egrad(X::Array{Matrix{Float64},1})
    U = X[1]
    V = X[2]
    AV = A * V
    AtU = transpose(A) * U
    return [-AV * (transpose(AV) * U), -AtU * (transpose(AtU) * V)]
end

function rgrad(M::ProductManifold, X::ProductRepr)
    eG = egrad([submanifold_components(M, X)...])
    x = [submanifold_components(M, X)...]
    return Manifolds.ProductRepr(project.(M.manifolds, x, eG)...)
end

function e2rHess(
    M::Grassmann, x, ξ, eGrad::Matrix{T}, Hess::Matrix{T}
) where {T<:Union{U,Complex{U}}} where {U<:AbstractFloat}
    pxHess = project(M, x, Hess)
    xtGrad = x' * eGrad
    ξxtGrad = ξ * xtGrad
    return pxHess - ξxtGrad
end

function eHess(X::Array{Matrix{Float64},1}, H::Array{Matrix{Float64},1})
    U = X[1]
    V = X[2]
    Udot = H[1]
    Vdot = H[2]
    AV = A * V
    AtU = transpose(A) * U
    AVdot = A * Vdot
    AtUdot = transpose(A) * Udot
    return [
        -(
            AVdot * transpose(AV) * U +
            AV * transpose(AVdot) * U +
            AV * transpose(AV) * Udot
        ),
        -(
            AtUdot * transpose(AtU) * V +
            AtU * transpose(AtUdot) * V +
            AtU * transpose(AtU) * Vdot
        ),
    ]
end

function rhess(M::ProductManifold, X::ProductRepr, H::ProductRepr)
    x = [submanifold_components(M, X)...]
    h = [submanifold_components(M, H)...]
    eG = egrad(x)
    eH = eHess(x, h)
    return Manifolds.ProductRepr(e2rHess.(M.manifolds, x, h, eG, eH)...)
end

@testset "Manopt Trust-Region" begin
    import Random: seed!
    seed!(42)

    N = Grassmann(3, 2)
    M = N × N

    x = random_point(M)

    @test_throws ErrorException trust_regions(M, cost, rgrad, x, rhess; ρ_prime=0.3)
    @test_throws ErrorException trust_regions(M, cost, rgrad, x, rhess; Δ_bar=-0.1)
    @test_throws ErrorException trust_regions(M, cost, rgrad, x, rhess; Δ=-0.1)
    @test_throws ErrorException trust_regions(M, cost, rgrad, x, rhess; Δ_bar=0.1, Δ=0.11)

    X = trust_regions(M, cost, rgrad, x, rhess; Δ_bar=4 * sqrt(2 * 2))
    opt = trust_regions(
        M, cost, rgrad, x, rhess; Δ_bar=4 * sqrt(2 * 2), return_options=true
    )
    @test isapprox(M, X, get_solver_result(opt))

    @test cost(X) + 142.5 ≈ 0 atol = 10.0^(-13)

    XuR = trust_regions(M, cost, rgrad, x, rhess; Δ_bar=4 * sqrt(2 * 2), useRandom=true)

    @test cost(XuR) + 142.5 ≈ 0 atol = 10.0^(-12)

    XaH = trust_regions(
        M,
        cost,
        rgrad,
        x,
        (p, x, ξ) -> approxHessianFD(
            p,
            x,
            x -> rgrad(p, x),
            ξ;
            stepsize=2^(-9),
            transport=ProductVectorTransport(ProjectionTransport(), ProjectionTransport()),
        );
        stopping_criterion=StopWhenAny(
            StopAfterIteration(2000), StopWhenGradientNormLess(10^(-6))
        ),
        Δ_bar=4 * sqrt(2 * 2),
    )
    @test cost(XaH) + 142.5 ≈ 0 atol = 10.0^(-10)

    ξ = random_tangent(M, x)
    @test_throws ErrorException getHessian(SubGradientProblem(M, cost, rgrad), x, ξ)

    # Test the random step trust region
    p = HessianProblem(M, cost, rgrad, rhess, (M, x, ξ) -> ξ)
    o = TrustRegionsOptions(
        x,
        StopAfterIteration(2000),
        10.0^(-8),
        sqrt(manifold_dimension(M)),
        retract,
        true,
        0.1,
        1000.0,
    )
    @test step_solver!(p, o, 0) == nothing

    η = truncated_conjugate_gradient_descent(M, cost, rgrad, x, ξ, rhess, 0.5)
    ηOpt = truncated_conjugate_gradient_descent(
        M, cost, rgrad, x, ξ, rhess, 0.5; return_options=true
    )
    @test submanifold_components(get_solver_result(ηOpt)) == submanifold_components(η)
end
