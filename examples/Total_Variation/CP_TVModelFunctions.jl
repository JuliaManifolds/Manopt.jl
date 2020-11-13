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
M = PowerManifold(pixelM, s...)
M2 = PowerManifold(pixelM, res(s)... )
N = TangentBundle(M2)
m2(m) = repeat(m,inner=rep(length(size(m))))
#
# Build TV functionals
#
fidelity(x) = 1/2*distance(M,x,f)^2
function Λ(x)
    return ProductRepr(m2(x),forwardLogs(M,x)) # on N=TM, namely in T_xM
end
function prior(x)
  # inner 2-norm over logs, 1-norm over the pixel
  return norm(norm.(Ref(pixelM),getValue(x), getValue(getTangent(Λ(x)))), 1)
end
cost(x) = (1/α)*fidelity(x) + prior(x)

proxFidelity(M,m,λ,x) = proxDistance(M,λ/α,f,x,2)
proxPriorDual(N,n,λ,ξ) = ProductRepr(
    getBase(ξ),
    projCollaborativeTV(getBase(N),λ,getBase(n),getTangent(ξ), Inf, Inf)
)
DΛ(m,ξm) = TProductRepr(
    repeat( ξm, inner=rep(length(size(m))) ),
    DforwardLogs(M,m,ξm)
)
AdjDΛ(m,ξn) = AdjDforwardLogs(M,m,getTangent(ξn))