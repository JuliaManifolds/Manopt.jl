using Manifolds, Manopt, Random, LinearAlgebra, BenchmarkTools, Profile
import Manifolds: vector_transport_to!
vector_transport_to!(M::Stiefel, Y, p, X, q, ::ProjectionTransport) = project!(M, Y, q, X)

struct GradF
    A::Matrix{Float64}
    N::Diagonal{Float64,Vector{Float64}}
end
function (gradF::GradF)(X::Array{Float64,2})
    AX = gradF.A * X
    XpAX = X' * AX
    return 2 .* AX * gradF.N .- X * XpAX * gradF.N .- X * gradF.N * XpAX
end

function run_brocket_experiment(n::Int, k::Int, m::Int; seed=42)
    Random.seed!(42)
    M = Stiefel(n, k)
    A = randn(n, n)
    A = (A + A') / 2
    F(X::Array{Float64,2}) = tr((X' * A * X) * Diagonal(k:-1:1))
    gradF = GradF(A, Diagonal(Float64.(collect(k:-1:1))))
    x = random_point(M)
    return quasi_Newton(
        M,
        F,
        gradF,
        x;
        memory_size=m,
        vector_transport_method=ProjectionTransport(),
        retraction_method=QRRetraction(),
        stopping_criterion=StopWhenGradientNormLess(norm(M, x, gradF(x)) * 10^(-6)),
        cautious_update=true,
        #        debug = [:Iteration," ", :Cost, " ", DebugGradientNorm(), "\n", 10],
    )
end

io = IOBuffer()

for e in [
    (32, 32, 1),
    (32, 32, 2),
    (32, 32, 4),
    (32, 32, 8),
    (32, 32, 16),
    (32, 32, 32),
    (1000, 2, 4),
    (1000, 3, 4),
    (1000, 4, 4),
    (1000, 5, 4),
]
    println("Benchmarking $(e):")
    b = @benchmark run_brocket_experiment($(e[1]), $(e[2]), $(e[3])) samples = 50
    #run_brocket_experiment(e[1], e[2], e[3])
    show(io, "text/plain", b)
    s = String(take!(io))
    println(s, "\n\n")
end
