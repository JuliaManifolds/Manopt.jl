#
# Model TV for a given pixelManifold if
# * pixelM
# * and given data f
# * weight α
#
rep(d) = (d>1) ? [ones(Int,d)...,d] : d
res(s::Int) = res([s])
res(s) = (length(s) > 1) ? [s...,length(s)] : s
d = length(size(f))
s = size(f)
M = PowerManifold(pixelM, NestedPowerRepresentation(), s...)
M2 = PowerManifold(pixelM, NestedPowerRepresentation(), res(s)... )
N = TangentBundle(M2)
m2(m) = repeat(m,inner=rep(length(size(m))))
#
# Build TV functionals
#
fidelity(x) = 1/2*distance(M,x,f)^2
function Λ(x)
    return ProductRepr(m2(x),forward_logs(M,x)) # on N=TM, namely in T_xM
end
function prior(x)
    # inner 2-norm over logs, 1-norm over the pixel
    return norm(norm.(Ref(pixelM), x, submanifold_component(N, Λ(x), 2)), 1)
end
cost(x) = (1/α)*fidelity(x) + prior(x)

proxFidelity(M,m,λ,x) = prox_distance(M,λ/α,f,x,2)
proxPriorDual(N,n,λ,ξ) = ProductRepr(
    submanifold_component(N,ξ,1),
    prox_collaborative_TV(
        base_manifold(N),
        λ,
        submanifold_component(N,n,1),
        submanifold_component(N,ξ,2),
        Inf,
        Inf,
    ),
)
DΛ(m,ξm) = ProductRepr(
    repeat(ξm, inner=rep(length(size(m))) ),
    differential_forward_logs(M, m, ξm),
)
AdjDΛ(m,ξn) = adjoint_differential_forward_logs(M, m, submanifold_component(N, ξn, 2))