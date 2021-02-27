using Manopt, Manifolds, ManifoldsBase, Test

@testset "Chambolle-Pock" begin
    #
    # Perform an really easy test, just compute a mid point
    #
    pixelM = Sphere(2)
    M = PowerManifold(pixelM, NestedPowerRepresentation(), 2)
    data = [[1.0, 0.0, 0.0], 1 / sqrt(2) .* [1.0, 1.0, 0.0]]
    α = 1
    # known minimizer
    δ = min(α / distance(pixelM, data[1], data[2]), 0.5)
    x_hat = shortest_geodesic(M, data, reverse(data), δ)
    N = TangentBundle(M)
    fidelity(M, x) = 1 / 2 * distance(M, x, f)^2
    Λ(M, x) = ProductRepr(x, forward_logs(M, x))
    prior(M, x) = norm(norm.(Ref(M.manifold), x, submanifold_component(N, Λ(x), 2)), 1)
    cost(M, x) = (1 / α) * fidelity(M, x) + prior(M, x)
    prox_F(M, m, λ, x) = prox_distance(M, λ / α, data, x, 2)
    function prox_G_dual(N, n, λ, ξ)
        return ProductRepr(
            submanifold_component(N, ξ, 1),
            project_collaborative_TV(
                base_manifold(N),
                λ,
                submanifold_component(N, n, 1),
                submanifold_component(N, ξ, 2),
                Inf,
                Inf,
                1.0,
            ),
        )
    end
    DΛ(M, m, X) = ProductRepr(zero_tangent_vector(M, m), differential_forward_logs(M, m, X))
    function adjoint_DΛ(N, n, ξ)
        return adjoint_differential_forward_logs(
            N.manifold, m, submanifold_component(N, ξ, 2)
        )
    end

    m = fill(mid_point(pixelM, data[1], data[2]), 2)
    n = Λ(M, m)
    x0 = deepcopy(data)
    ξ0 = ProductRepr(zero_tangent_vector(M, m), zero_tangent_vector(M, m))
    @testset "Test Variants" begin
        callargs_linearized = [M, N, cost, x0, ξ0, m, n, prox_F, prox_G_dual, adjoint_DΛ]
        o1 = ChambollePock(
            callargs_linearized...;
            linearized_forward_operator=DΛ,
            relax=:dual,
            variant=:linearized,
        )
        o2 = ChambollePock(
            callargs_linearized...;
            linearized_forward_operator=DΛ,
            relax=:primal,
            variant=:linearized,
        )
        @test o1 ≈ o2 atol = 2 * 1e-7
        callargs_exact = [M, N, cost, x0, ξ0, m, n, prox_F, prox_G_dual, adjoint_DΛ]
        o3 = ChambollePock(callargs_exact...; Λ=Λ, relax=:dual, variant=:exact)
        o4 = ChambollePock(callargs_exact...; Λ=Λ, relax=:primal, variant=:exact)
        @test o3 ≈ o4 atol = 2 * 1e-7
        @test o1 ≈ o3
        o1a = ChambollePock(
            callargs_linearized...;
            linearized_forward_operator=DΛ,
            relax=:dual,
            variant=:linearized,
            return_options=true,
        )
        @test get_solver_result(o1a) == o1
        o2a = ChambollePock(
            callargs_linearized...;
            linearized_forward_operator=DΛ,
            relax=:dual,
            variant=:linearized,
            update_dual_base=(p, o, i) -> o.n,
        )
        @test o2a ≈ o3
    end
end
