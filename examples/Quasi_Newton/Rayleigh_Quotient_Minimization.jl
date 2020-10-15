#
#   Rayleigh quotient minimization on S^n
#
using Manopt, Manifolds, ManifoldsBase, Random, LinearAlgebra
Random.seed!(42)
"""
    rayleigh_quotient_minimization(A)

"""
n = 300
A = randn(n,n)
A = (A + A')/2
M = Sphere(n-1)
F(X::Array{Float64,1}) = transpose(X)*A*X
∇F(X::Array{Float64,1}) = 2*project(M,X,A*X)
∇2F(X::Array{Float64,1}, ξ::Array{Float64,1}) = 2*project(M,X,A*ξ - ξ*X'*A*X)
x = random_point(M)
op = operator_to_matrix(M,x,(ξ) -> ∇2F(x, ξ))
quasi_Newton(M,F,∇F,x; memory_size = -250, debug = [:Iteration, " ", :Cost, "\n", 1, :Stop])
