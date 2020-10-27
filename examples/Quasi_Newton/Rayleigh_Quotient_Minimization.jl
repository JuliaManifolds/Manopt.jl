#
#   Rayleigh quotient minimization on S^n
#
using Manopt, Manifolds, ManifoldsBase, Random, LinearAlgebra, BenchmarkTools
Random.seed!(42)
"""
    rayleigh_quotient_minimization(A)

"""
n = 300
A = randn(n,n)
A = (A + A')/2
M = Sphere(n-1)
F(X::Array{Float64,1}) = X' * A * X
∇F(X::Array{Float64,1}) = 2*(A*X - X * X' * A * X)
x = random_point(M)
@benchmark quasi_Newton(M,F,∇F,x; memory_size = -1, stopping_criterion = StopWhenGradientNormLess(norm(M,x,∇F(x))*10^(-6))) seconds = 600 samples = 100