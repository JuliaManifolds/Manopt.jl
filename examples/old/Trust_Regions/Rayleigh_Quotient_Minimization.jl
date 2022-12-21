using Manopt, Manifolds, ManifoldsBase, Random, LinearAlgebra, BenchmarkTools
Random.seed!(1)

# function run_rayleigh_minimization(n::Int)
#     A = randn(n, n)
#     A = (A + A') / 2
#     F(::Sphere, p::Array{Float64,1}) = p' * A * p
#     gradF(::Sphere, p::Array{Float64,1}) = 2 * (A * p - p * p' * A * p)
#     function HessF(::Sphere, p::Array{Float64,1}, X::Array{Float64,1})
#         return 2 * (A * X - p * p' * A * X - X * p' * A * p - p * p' * X * p' * A * p)
#     end
#     M = Sphere(n - 1)
#     x = random_point(M)
# return trust_regions!(
#     M,
#     F,
#     gradF,
#     ApproxHessianSymmetricRankOne(M, x, gradF; nu=eps(Float64)^2),
#     x;
#     stopping_criterion=StopWhenAny(
#         StopAfterIteration(1000), StopWhenGradientNormLess(10^(-6))
#     ),
#     θ=0.1,
#     κ=0.9,
#     trust_region_radius=1.0,
#     retraction_method=ProjectionRetraction(),
# )
# end
# io = IOBuffer()

# for n in [50, 100, 200]
#     b = @benchmark run_rayleigh_minimization($n) samples = 30 evals = 10 seconds = 600
#     show(io, "text/plain", b)
#     s = String(take!(io))
#     println("Benchmarking $(n):\n", s, "\n\n")
# end

n = 100

A = randn(n, n)
A = (A + A') / 2
F(::Sphere, p::Array{Float64,1}) = p' * A * p
gradF(::Sphere, p::Array{Float64,1}) = 2 * (A * p - p * p' * A * p)
function HessF(::Sphere, p::Array{Float64,1}, X::Array{Float64,1})
    return 2 * (A * X - p * p' * A * X - X * p' * A * p - p * p' * X * p' * A * p)
end
M = Sphere(n - 1)
x = random_point(M)
x = trust_regions!(
    M,
    F,
    gradF,
    ApproxHessianBFGS(M, x, gradF),
    x;
    stopping_criterion=StopWhenAny(
        StopAfterIteration(10000), StopWhenGradientNormLess(10^(-6))
    ),
    trust_region_radius=1.0,
    θ=0.1,
    κ=0.9,
    retraction_method=ProjectionRetraction(),
)

ev = eigvecs(A)[:, 1]

return norm(abs.(x) - abs.(ev))
