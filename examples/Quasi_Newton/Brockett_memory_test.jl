using Manifolds, Manopt, Random, LinearAlgebra, BenchmarkTools
import Manifolds: vector_transport_to!
vector_transport_to!(M::Stiefel,Y,p,X,q,::ProjectionTransport) = (Y .= project(M, q, X))
M = Stiefel(1000,3)
A = randn(1000,1000)
A = (A + A')
N = diagm(3:-1:1)
F(X::Array{Float64,2}) = tr(X' * A * X * N)
∇F(X::Array{Float64,2}) = 2 * A * X * N - X * X' * A * X * N - X * N * X' * A * X

Random.seed!(42)
x = random_point(M)
bench = @benchmark quasi_Newton(M, F, ∇F, $x;
    memory_size = 4,
    vector_transport_method = ProjectionTransport(),
    retraction_method = QRRetraction(),
    cautious_update = true,
    stopping_criterion = StopAfterIteration(50),
    debug = [:Iteration, " ", :Cost, "\n", 10],
#    stopping_criterion = StopWhenGradientNormLess(norm(M,$x,∇F($x))*10^(-6))
) seconds = 600 samples = 1 evals = 1
