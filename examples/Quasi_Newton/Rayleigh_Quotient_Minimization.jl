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

function F(X::Array{Float64,1})
    return transpose(X)*A*X
end

function ∇F(X::Array{Float64,1})
    return 2*(A*X - X*transpose(X)*A*X)
end

x = random_point(M)

quasi_Newton(M,F,∇F,x)
