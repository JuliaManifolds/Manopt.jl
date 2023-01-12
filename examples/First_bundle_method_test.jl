using Manopt, Manifolds, Random, QuadraticModels, RipQP

M = SymmetricPositiveDefinite(3)
Random.seed!(42)
data = [rand(M; σ=0.4) for i in 1:100]

F(M, y) = sum(1 / (2 * length(data)) * distance.(Ref(M), data, Ref(y)) .^ 2)
gradF(M, y) = sum(1 / length(data) * grad_distance.(Ref(M), data, Ref(y)))
@time b_mean = bundle_method(
    M, F, gradF, data[1]; stopping_criterion=StopAfterIteration(100)
)#, debug = [:Iteration, :Cost, "\n"])
@time m_mean = mean(M, data)
mean_dist = distance(M, b_mean, m_mean)
println("Distance between means: $mean_dist")
println("F(b_mean) = $(F(M, b_mean))")

F2(M, y) = sum(1 / (2 * length(data)) * distance.(Ref(M), Ref(y), data))
gradF2(M, y) = sum(1 / (2 * length(data)) * grad_distance.(Ref(M), data, Ref(y), 1))
@time b_median = bundle_method(
    M, F2, gradF2, data[1]; stopping_criterion=StopAfterIteration(100)
)#, debug = [:Iteration, :Cost, "\n"])
@time m_median = median(M, data)
median_dist = distance(M, b_median, m_median)
println("Distance between medians: $median_dist")
println("F2($b_median) = $(F2(M, b_median))")
