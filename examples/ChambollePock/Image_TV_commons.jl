#
# Model TV for a given pixelManifold if
# * pixelM
# * and given data f
# * weight α
#
rep(d) = (d > 1) ? [ones(Int, d)..., d] : d
res(s::Int) = res([s])
res(s) = (length(s) > 1) ? [s..., length(s)] : s
d = length(size(f))
s = size(f)
M = PowerManifold(pixelM, NestedPowerRepresentation(), s...)
M2 = PowerManifold(pixelM, NestedPowerRepresentation(), res(s)...)
N = TangentBundle(M2)
m2(m) = repeat(m; inner=rep(length(size(m))))
#
# Build TV functionals
#
fidelity(M, x) = 1 / 2 * distance(M, x, f)^2
function Λ(M, x)
    return ProductRepr(m2(x), forward_logs(M, x)) # on N=TM, namely in T_xM
end
function prior(M, x)
    # inner 2-norm over logs, 1-norm over the pixel
    return norm(norm.(Ref(pixelM), x, submanifold_component(N, Λ(x), 2)), 1)
end
cost(M, x) = (1 / α) * fidelity(M, x) + prior(M, x)

proxFidelity(M, λ, x) = prox_distance(M, λ / α, f, x, 2)
function proxPriorDual(N, n, λ, ξ)
    return ProductRepr(
        submanifold_component(N, ξ, 1),
        project_collaborative_TV(
            base_manifold(N),
            λ,
            submanifold_component(N, n, 1),
            submanifold_component(N, ξ, 2),
            Inf,
            Inf,
        ),
    )
end
function DΛ(m, ξm)
    return ProductRepr(
        repeat(ξm; inner=rep(length(size(m)))), differential_forward_logs(M, m, ξm)
    )
end
AdjDΛ(m, ξn) = adjoint_differential_forward_logs(M, m, submanifold_component(N, ξn, 2));
