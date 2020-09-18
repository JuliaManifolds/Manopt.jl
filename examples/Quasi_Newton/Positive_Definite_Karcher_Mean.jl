#
#   Computes a Karcher mean of a collection of positive definite matrices
#
using Manopt, Manifolds, ManifoldsBase, LinearAlgebra

"""
    positive_definite_karcher_mean(A)

"""

n = 5
m = 50

A = zeros(n,n,m)

for i = 1:m
    noise = 0.01 * rand(n,n)
    noise = (noise + transpose(noise))/2
    V = eigvecs(noise)
    D = real(eigvals(noise))
    A[:, :, i] = V * diagm(max.(.01, D)) * transpose(V)
end

M = SymmetricPositiveDefinite(n)


function F(X::Array{Float64,2})
    f = 0
    for i = 1 : m
        f = f + distance(X, A[:, :, i])^2
    end
    return f/(2*m)
end

function ∇F(X::Array{Float64,2})
    g = zero_tangent_vector(M,X)
    for i = 1 : m
        g = g - 1/m * log(M, X, A[:, :, i])
    end
    return g
end

x = random_point(M)

quasi_Newton(M,F,∇F,x)
