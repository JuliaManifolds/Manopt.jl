#
# The Riemannian mean of `n` points on the `Sphere`
#
module RiemannianMean

using Manopt, Manifolds, ManifoldDiff, Random
using ManifoldDiff: grad_distance, grad_distance!

const d = 30
const n = 800
const σ = π / 8
const M = Sphere(d)
const data = let rng = MersenneTwister(42), p = [i == 2 ? 1.0 : 0.0 for i in 1:(d + 1)]
    [exp(M, p, σ * rand(rng, M; vector_at = p)) for _ in 1:n]
end
const p0 = [i ≤ 2 ? 1 / sqrt(2) : 0.0 for i in 1:(d + 1)]
const sc = StopWhenGradientNormLess(1.0e-9) | StopAfterIteration(200)

f(M, p) = sum(distance(M, p, q)^2 for q in data) / (2 * n)
grad_f(M, p) = sum(grad_distance(M, q, p) for q in data) / n

struct GradF!{TD, TTMP}
    data::TD
    tmp::TTMP
end
function (grad_f!::GradF!)(M, X, p)
    zero_vector!(M, X, p)
    for q in grad_f!.data
        grad_distance!(M, grad_f!.tmp, q, p)
        X .+= grad_f!.tmp
    end
    X ./= length(grad_f!.data)
    return X
end

const grad_f! = GradF!(data, similar(data[1]))

end
