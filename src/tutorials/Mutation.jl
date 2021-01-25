# # [Illustration how to use mutating gradient functions](@id Mutations)
#
#
#
using Manopt, Manifolds, Random, BenchmarkTools, Test
Random.seed!(42)
m = 30
M = Sphere(m)
n = 800
σ = π / 8
x = zeros(Float64, m + 1)
x[end] = 1.0
data = [exp(M, x, random_tangent(M, x, Val(:Gaussian), σ)) for i in 1:n]
F(y) = sum(1 / (2 * n) * distance.(Ref(M), Ref(y), data) .^ 2)
∇F(y) = sum(1 / n * ∇distance.(Ref(M), data, Ref(y)))
function ∇F!(X, y)
    return X .= sum(1 / n * ∇distance.(Ref(M), data, Ref(y)))
end

sc = StopWhenGradientNormLess(10.0^-10)
x0 = random_point(M)
m1 = gradient_descent(M, F, ∇F, x0; stopping_criterion=sc)
@btime m1 = gradient_descent(M, F, ∇F, x0; stopping_criterion=sc)

m2 = deepcopy(x0)
@btime gradient_descent!(M, F, ∇F, m2; stopping_criterion=sc)

m3 = deepcopy(x0)
@btime gradient_descent!(
    M, F, ∇F!, m3; evaluation=MutatingEvaluation(), stopping_criterion=sc
)

@test distance(M, m1, m2) ≈ 0
@test distance(M, m1, m3) ≈ 0

# This results in
# include("src/tutorials/Mutation.jl")
#   14.832 ms (170084 allocations: 55.01 MiB)
#   781.083 μs (9633 allocations: 3.11 MiB)
#   777.958 μs (9645 allocations: 3.12 MiB)
