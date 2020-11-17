#
# Prepare cost, proximal maps and differentials
M = PowerManifold(pixelM, NestedPowerRepresentation(), size(f)...)
N = TangentBundle(M)
fidelity(x) = 1/2*distance(M,x,f)^2
Λ(x) = ProductRepr(x,forward_logs(M,x))
prior(x) = norm(norm.(Ref(pixelM), x, submanifold_component(N, Λ(x), 2)), 1)
cost(x) = (1/α)*fidelity(x) + prior(x)

prox_F(M,m,λ,x) = prox_distance(M,λ/α,f,x,2)
prox_G_dual(N,n,λ,ξ) = ProductRepr(
    submanifold_component(N,ξ,1),
    project_collaborative_TV(
        base_manifold(N),
        λ,
        submanifold_component(N,n,1),
        submanifold_component(N,ξ,2),
        Inf,
        Inf,
        1.),
    )

DΛ(m,X) = ProductRepr( zero_tangent_vector(M,m), differential_forward_logs(M,m,X))
adjoint_DΛ(m,ξ) = adjoint_differential_forward_logs(M, m, submanifold_component(N, ξ, 2))
