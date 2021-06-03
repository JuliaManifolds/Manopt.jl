using Manopt, Manifolds, Random, LinearAlgebra

Random.seed!(1)
n = 64

#=
Lambda1 = 0.01*ones(Int(floor((n-1)/2)))
Lambda2 = 2.0*ones(n - 1 - Int(floor((n-1)/2)))
Lambda = append!([0.], Lambda1)
Lambda = append!(Lambda, Lambda2)
Lambda = diagm(Lambda)
=#

Lambda = Float64.(diagm(1:n))

Q = qr(randn(n, n)).Q
A = Q * Lambda * Q'

F(::Sphere, p::Array{Float64,1}) = p' * A * p
gradF(::Sphere, p::Array{Float64,1}) = 2 * (A * p - p * p' * A * p)
function HessF(::Sphere, p::Array{Float64,1}, X::Array{Float64,1})
    return 2 * (A * X - p * p' * A * X - X * p' * A * p - p * p' * X * p' * A * p)
end
Id(::Sphere, p::Array{Float64,1}, X::Array{Float64,1}) = X

M = Sphere(n - 1)
x = random_point(M)

trust_regions!(
    M,
    F,
    gradF,
    ApproxHessianSymmetricRankOne(M, x, gradF; nu=sqrt(eps(Float64))),
    x;
    stopping_criterion=StopWhenAny(
        StopAfterIteration(500), StopWhenGradientNormLess(norm(M, x, gradF(M, x)) * 10^(-6))
    ),
    retraction_method=ProjectionRetraction(),
    θ=0.1,
    κ=0.9,
    trust_region_radius=1.0,
    debug=[:Iteration, " ", :Cost, " | ", DebugEntry(:trust_region_radius), "\n", 1, :Stop],
)

#=
trust_regions!(
    M,
    F,
    gradF,
    HessF,
    x;
    stopping_criterion=StopWhenAny(
        StopAfterIteration(500),
        StopWhenGradientNormLess(norm(M, x, gradF(M, x)) * 10^(-6)),
    ),
    retraction_method=ProjectionRetraction(),
    trust_region_radius=1.0,
    debug=[
        :Iteration, " ", :Cost, " | ", DebugEntry(:trust_region_radius), "\n", 1, :Stop
    ],
)
=#

#=
trust_regions!(
    M,
    F,
    gradF,
    Id,
    x;
    stopping_criterion=StopWhenAny(
        StopAfterIteration(500),
        StopWhenGradientNormLess(norm(M, x, gradF(M, x)) * 10^(-6)),
    ),
    retraction_method=ProjectionRetraction(),
    θ=0.1,
    κ=0.9,
    trust_region_radius=1.0,
    debug=[
        :Iteration, " ", :Cost, " | ", DebugEntry(:trust_region_radius), "\n", 1, :Stop
    ],
)
=#
