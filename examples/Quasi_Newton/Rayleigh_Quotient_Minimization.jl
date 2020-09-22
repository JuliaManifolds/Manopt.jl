#
#   Rayleigh quotient minimization on S^n
#
using Manopt, Manifolds, ManifoldsBase
"""
    rayleigh_quotient_minimization(A)

"""
n = 5
A = rand(n,n)
A = (transpose(A) + A)/2
M = Sphere(n-1)
F(X::Array{Float64,1}) = transpose(X)*A*X
∇F(X::Array{Float64,1}) = 2*(A*X - X*transpose(X)*A*X)
x = random_point(M)
quasi_Newton(M,F,∇F,x; memory_size = -1)
