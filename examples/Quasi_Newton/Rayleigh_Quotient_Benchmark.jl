using Manopt, Manifolds, ManifoldsBase, Random, LinearAlgebra, BenchmarkTools
Random.seed!(42)

function run_rayleigh_experiment(n::Int)
    A = randn(n, n)
    A = (A + A') / 2
    F(::Sphere, X::Array{Float64,1}) = X' * A * X
    gradF(::Sphere, X::Array{Float64,1}) = 2 * (A * X - X * X' * A * X)
    M = Sphere(n - 1)
    x = random_point(M)
    return quasi_Newton(
        M,
        F,
        gradF,
        x;
        #memory_size=-1,
        stopping_criterion=StopWhenAny(
            StopAfterIteration(max(1000)), StopWhenGradientNormLess(10^(-6))
        ),
        debug=[:Iteration, " ", :Cost, "\n", 1, :Stop],
    )
end
io = IOBuffer()

for n in [100]
    b = @benchmark run_rayleigh_experiment($n) samples = 30
    show(io, "text/plain", b)
    s = String(take!(io))
    println("Benchmarking $(n):\n", s, "\n\n")
end
