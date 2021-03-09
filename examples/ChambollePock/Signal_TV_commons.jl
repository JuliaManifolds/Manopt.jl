#
# Prepare cost, proximal maps and differentials
M = PowerManifold(pixelM, NestedPowerRepresentation(), size(f)...)
N = TangentBundle(M)
fidelity(M, x) = 1 / 2 * distance(M, x, f)^2
Λ(M, x) = ProductRepr(x, forward_logs(M, x))
prior(M, x) = norm(norm.(Ref(pixelM), x, submanifold_component(N, Λ(M, x), 2)), 1)
cost(M, x) = (1 / α) * fidelity(M, x) + prior(M, x)

prox_F(M, λ, x) = prox_distance(M, λ / α, f, x, 2)
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
adjoint_DΛ(N, m, n, ξ) = adjoint_differential_forward_logs(N.manifold, m, ξ[N, :vector])
