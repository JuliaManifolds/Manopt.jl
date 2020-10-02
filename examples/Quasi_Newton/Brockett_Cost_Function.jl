#
#   Brockett Cost Function on Stiefel(n,k)
#
using Manopt, Manifolds, ManifoldsBase, LinearAlgebra, Random
import Manifolds: vector_transport_to!
vector_transport_to!(::Stiefel,Y,p,X,q,::ProjectionTransport) = (Y .= project(M, q, X))
# see Huang:2013, 10.3.2 Vector Transport
Random.seed!(42)
n = 32
k = 32
M = Stiefel(n,k)
A = randn(n,n)
A = (A + A')
N = diagm(k:-1:1)
F(X::Array{Float64,2}) = tr(X' * A * X * N)
∇F(X::Array{Float64,2}) = project(M, X, 2 * A * X * N)
x = random_point(M)
B1 = quasi_Newton(M,F,∇F,x; memory_size = 1000, vector_transport_method = ProjectionTransport(), debug = [:Iteration, " ", :Cost, "\n", 1, :Stop])
