#
#   Brockett Cost Function on Stiefel(n,k)
#
using Manopt, Manifolds, ManifoldsBase, LinearAlgebra, Random, BenchmarkTools
import Manifolds: vector_transport_to!
vector_transport_to!(::Stiefel,Y,p,X,q,::ProjectionTransport) = (Y .= project(M, q, X))
# vector_transport_to!(::Rotations,Y,p,X,q,::ProjectionTransport) = (Y .= project(M, q, X))
# see Huang:2013, 10.3.2 Vector Transport
Random.seed!(42)
n = 1000
k = 5
 M = Stiefel(n,k)
# M = Rotations(n)
A = randn(n,n)
A = (A + A')
N = diagm(k:-1:1)
F(X::Array{Float64,2}) = tr(X' * A * X * N)
∇F(X::Array{Float64,2}) = 2 * A * X * N - X * X' * A * X * N - X * N * X' * A * X
x = random_point(M)
@benchmark quasi_Newton(M,F,∇F,x; memory_size = 32, vector_transport_method = ProjectionTransport(), stopping_criterion = StopWhenGradientNormLess(norm(M,x,∇F(x))*10^(-6))) seconds = 600 samples = 100