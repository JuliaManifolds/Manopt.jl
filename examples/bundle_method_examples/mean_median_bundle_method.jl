using Manopt, Manifolds, ManifoldDiff, Random, QuadraticModels, RipQP

N = 10000
# M = Hyperbolic(9)
M = SymmetricPositiveDefinite(4)
Random.seed!(42)
p = rand(M)
σ = 0.5
data = [exp(M, p, σ * rand(M; vector_at=p)) for i in 1:N]
tol = 1e-8

# F(M, y) = sum(1 / (2 * length(data)) * distance.(Ref(M), data, Ref(y)) .^ 2)
# gradF(M, y) = sum(1 / length(data) * grad_distance.(Ref(M), data, Ref(y)))
# @time b_mean = bundle_method(
#     M,
#     F,
#     gradF,
#     data[1];
#     #StopWhenAny(StopWhenChangeLess(1e-12), StopAfterIteration(5000)),
#     debug=["    ", :Iteration, :Cost, :diam, "\n"],
# )
# @time m_mean = mean(M, data; stop_iter=100)
# mean_dist = distance(M, b_mean, m_mean)
# println("Distance between means: $mean_dist")
# println(
#     "$(F(M, b_mean) < F(M,m_mean) ? "F(bundle_mean) < F(manifolds_mean)" : "F(bundle_mean) ≥ F(manifolds_mean)")",
# )
# println("    |F(bundle_mean) - F(manifolds_mean)| = $(abs(F(M, b_mean) - F(M, m_mean)))")

F2(M, y) = sum(1 / length(data) * distance.(Ref(M), Ref(y), data))
function gradF2(M, y)
    return sum(
        1 / length(data) *
        ManifoldDiff.subgrad_distance.(Ref(M), data, Ref(y), 1; atol=√eps()),
    )
end
@time b_median = bundle_method(
    M,
    F2,
    gradF2,
    data[1];
    diam=2.5,
    stopping_criterion=StopWhenBundleLess(tol),#StopWhenAny(StopWhenChangeLess(1e-12), StopAfterIteration(5000)),
    debug=[
        :Iteration,
        (:Cost, "F(p): %1.16f "),
        (:ξ, "ξ: %1.16f "),
        (:ε, "ε: %1.16f "),
        (:diam, "diam: %1.16f "),
        "\n",
        1,
        :Stop,
    ],
)
# @time m_median = median(M, data)
# median_dist = distance(M, b_median, m_median)
# println("Distance between medians: $median_dist")
# println(
#     "$(F2(M, b_median) < F2(M,m_median) ? "F2(bundle_median) < F2(manifolds_median)" : "F2(bundle_median) ≥ F2(manifolds_median)")",
# )
# println(
#     "    |F2(bundle_median) - F2(manifolds_median)| = $(abs(F2(M, b_median) - F2(M, m_median)))",
# )
