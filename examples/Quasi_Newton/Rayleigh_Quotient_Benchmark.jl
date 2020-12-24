using Manopt, Manifolds, ManifoldsBase, Random, LinearAlgebra, BenchmarkTools

function run_rayleigh_experiment(n::Int; seed=42)
    Random.seed!(seed)
    A = randn(n,n)
    A = (A + A') / 2
    F(X::Array{Float64,1}) = X' * A * X
    ∇F(X::Array{Float64,1}) = 2 * (A * X - X * X' * A * X)
    M = Sphere(n-1)
    x = random_point(M)
    return quasi_Newton(
        M,
        F,
        ∇F,
        x;
        memory_size=-1,
        stopping_criterion=StopWhenAny(
        StopAfterIteration(max(10000)), StopWhenGradientNormLess(
            10^(-6)
        )),
        debug=[:Iteration, " ", :Cost, "\n", 1, :Stop]
    )
end
io = IOBuffer()

for n ∈ [100,300]
    b =  @benchmark run_rayleigh_experiment($n) samples = 30
    show(io, "text/plain", b)
    s = String(take!(io))
    println("Benchmarking $(n):\n", s, "\n\n")
end
