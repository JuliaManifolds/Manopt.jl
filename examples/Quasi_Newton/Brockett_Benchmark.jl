using Manifolds, Manopt, Random, LinearAlgebra, BenchmarkTools, Profile
import Manifolds: vector_transport_to!
vector_transport_to!(M::Stiefel, Y, p, X, q, ::ProjectionTransport) = project!(M, Y, q, X)

function run_brocket_experiment(n::Int, k::Int, m::Int; seed=42)
    Random.seed!(42)
    M = Stiefel(n, k)
    A = randn(n, n)
    A = (A + A')/2
    N = diagm(k:-1:1)
    F(X::Array{Float64,2}) = tr(X' * A * X * N)
    ∇F(X::Array{Float64,2}) = 2 * A * X * N - X * X' * A * X * N - X * N * X' * A * X
    x = random_point(M)
    return quasi_Newton(
        M,
        F,
        ∇F,
        x;
        memory_size=m,
        vector_transport_method=ProjectionTransport(),
        retraction_method=QRRetraction(),
#        stopping_criterion=StopWhenGradientNormLess(norm(M, x, ∇F(x)) * 10^(-6)),
        stopping_criterion=StopAfterIteration(230),
        cautious_update=true,
#        debug = [:Iteration," ", :Cost, " ", DebugGradientNorm(), "\n", 1],
    )
end

io = IOBuffer()

for e ∈ [
        (32, 32, 1), (32, 32, 2), (32, 32, 4),(32, 32, 8), (32, 32, 16), (32, 32, 32),
        (1000, 2, 4),
        (1000, 3, 4),
        (1000, 4, 4),
        (1000, 5, 4),
        ]
    b =  @benchmark run_brocket_experiment($(e[1]),$(e[2]), $(e[3])) samples = 5
    show(io, "text/plain", b)
    s = String(take!(io))
    println("Benchmarking $(e):\n", s, "\n\n")
end
