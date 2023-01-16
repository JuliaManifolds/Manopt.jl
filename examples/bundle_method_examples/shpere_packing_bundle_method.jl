using FiniteDifferences, ManifoldDiff, Manopt, Manifolds, Random, QuadraticModels, RipQP 

l = 100
N = SymmetricPositiveDefinite(3)
M = ProductManifold(N,N,N,N)
Random.seed!(42)
data = [rand(M) for i in 1:l]

F(M, y) = [[(i == j) ? Inf : distance(N, get_component(M, y, i), get_component(M, y, j)) for i in 1:Manifolds.number_of_components(M)] for j in 1:Manifolds.number_of_components(M)]
G(M, y) = - minimum(minimum.(F(M,y)))

r_backend = ManifoldDiff.TangentDiffBackend(ManifoldDiff.FiniteDifferencesBackend())
gradG(M, y) = ManifoldDiff.gradient(M, G, y, r_backend)

b = bundle_method(M, G, gradG, rand(M))#; stopping_criterion = StopAfterIteration(35))#, debug = [:Iteration, :Cost, "\n"])
s = subgradient_method(M, G, gradG, rand(M))#; stopping_criterion = StopAfterIteration(35))#, debug = [:Iteration, :Cost, "\n"])
