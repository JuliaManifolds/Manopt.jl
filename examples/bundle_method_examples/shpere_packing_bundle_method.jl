using FiniteDifferences, ManifoldDiff, Manopt, Manifolds, Random, QuadraticModels, RipQP

l = 100
Random.seed!(42)
r_backend = ManifoldDiff.TangentDiffBackend(ManifoldDiff.FiniteDifferencesBackend())

N = SymmetricPositiveDefinite(2)
M = ProductManifold(N, N, N, N)
data = [rand(M) for i in 1:l]

function F(M, y)
    return [
        [
            (i == j) ? Inf : distance(N, get_component(M, y, i), get_component(M, y, j)) for
            i in 1:Manifolds.number_of_components(M)
        ] for j in 1:Manifolds.number_of_components(M)
    ]
end
G(M, y) = -minimum(minimum.(F(M, y)))
gradG(M, y) = ManifoldDiff.gradient(M, G, y, r_backend)

b = bundle_method(M, G, gradG, rand(M); debug = [:Iteration, :Cost, "\n"], stopping_criterion = StopAfterIteration(30))
s = subgradient_method(M, G, gradG, rand(M); debug = [:Iteration, :Cost, "\n"], stopping_criterion = StopAfterIteration(30))
distance(M, b, s)