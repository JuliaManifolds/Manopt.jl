using Manopt, Manifolds, Random

M = SymmetricPositiveDefinite(33)
Random.seed!(42)
data = [rand(M) for i in 1:100]
F(M, y) = sum(1 / (2 * length(data)) * distance.(Ref(M), data, Ref(y)) .^ 2)
gradF(M, y) = sum(1 / length(data) * grad_distance.(Ref(M), data, Ref(y)))
b_mean = bundle_method(M, F, gradF, data[1]; stopping_criterion=StopAfterIteration(100)) #debug = [:Iteration, :Cost, "\n"])
m_mean = mean(M, data)
dist = distance(M, b_mean, m_mean)
