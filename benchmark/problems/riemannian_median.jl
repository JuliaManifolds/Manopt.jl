"""
    RiemannianMedian

The Riemannian median of `n` points on the `Hyperbolic` space, providing the
non-smooth cost `f`, the proximal maps `proxes` and `proxes!` of its summands as
well as a starting point `p0` and a stopping criterion `sc`.
"""
module RiemannianMedian

using Manopt, Manifolds, ManifoldDiff, Random
using ManifoldDiff: prox_distance, prox_distance!

const d = 2
const n = 100
const σ = 1.0
const M = Hyperbolic(d)
const data = let rng = MersenneTwister(42), p = [i == d + 1 ? 1.0 : 0.0 for i in 1:(d + 1)]
    [exp(M, p, σ * rand(rng, M; vector_at = p)) for _ in 1:n]
end
const p0 = data[1]
# CPPA cycles through all `n` proximal maps per iteration, so a few hundred
# iterations already are a few 10 000 proximal maps.
const sc = StopAfterIteration(100) | StopWhenChangeLess(M, 1.0e-9)

f(M, p) = sum(distance(M, p, q) for q in data) / n
# the proximal maps of the summands, `r = 1` being the one of the distance
# itself, as opposed to the squared distance of the mean
const proxes = Function[(M, λ, p) -> prox_distance(M, λ / n, q, p, 1) for q in data]
const proxes! = Function[
    (M, r, λ, p) -> prox_distance!(M, r, λ / n, q, p, 1) for q in data
]

end
