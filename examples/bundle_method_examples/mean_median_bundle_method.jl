using Manopt, Manifolds, Random, QuadraticModels, RipQP

M = SymmetricPositiveDefinite(4)
Random.seed!(55)
data = [rand(M; σ=0.4) for i in 1:100]

F(M, y) = sum(1 / (2 * length(data)) * distance.(Ref(M), data, Ref(y)) .^ 2)
gradF(M, y) = sum(1 / length(data) * grad_distance.(Ref(M), data, Ref(y)))
@time b_mean = bundle_method(
    M,
    F,
    gradF,
    data[1];
    stopping_criterion=StopAfterIteration(100)#StopWhenAny(StopWhenChangeLess(1e-12), StopAfterIteration(5000)),
)
@time m_mean = mean(M, data)
mean_dist = distance(M, b_mean, m_mean)
println("Distance between means: $mean_dist")
println(
    "$(F(M, b_mean) <= F(M,m_mean) ? "F(bundle_mean) ≤ F(manifolds_mean)" : "F(bundle_mean) > F(manifolds_mean)")",
)

F2(M, y) = sum(1 / (2 * length(data)) * distance.(Ref(M), Ref(y), data))
gradF2(M, y) = sum(1 / (2 * length(data)) * grad_distance.(Ref(M), data, Ref(y), 1))
@time b_median = bundle_method(
    M,
    F2,
    gradF2,
    data[1];
    stopping_criterion=StopAfterIteration(100) #StopWhenAny(StopWhenChangeLess(1e-12), StopAfterIteration(5000)),
)
# @time m_median = median(M, data)
# median_dist = distance(M, b_median, m_median)
# println("Distance between medians: $median_dist")
# println(
#     "$(F2(M, b_median) <= F2(M,m_median) ? "F2(bundle_median) ≤ F2(manifolds_median)" : "F2(bundle_median) > F2(manifolds_median)")",
# )
