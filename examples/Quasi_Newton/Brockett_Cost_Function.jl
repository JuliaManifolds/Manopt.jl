#
#   Brockett Cost Function on Stiefel(n,k)
#
using Manopt, Manifolds, ManifoldsBase, LinearAlgebra, Random
import Manifolds: vector_transport_to!

vector_transport_to!(::Stiefel,Y,p,X,q,::ProjectionTransport) = project!(M, Y, q, X)

Random.seed!(42)
n = 32
k = 32
M = Stiefel(n,k)
A = randn(n,n)
A = (A + A')
N = diagm(k:-1:1)
F(X::Array{Float64,2}) = tr(X' * A * X * N)
∇F(X::Array{Float64,2}) = 2 * A * X * N - X * X' * A * X * N - X * N * X' * A * X
x = random_point(M)
@time quasi_Newton(M,F,∇F,x; memory_size = 4, vector_transport_method = ProjectionTransport(), retraction_method = QRRetraction(), cautious_update = true, stopping_criterion = StopWhenGradientNormLess(norm(M,x,∇F(x))*10^(-6)), debug = [:Iteration, " ", :Cost, "\n", 1, :Stop]) 