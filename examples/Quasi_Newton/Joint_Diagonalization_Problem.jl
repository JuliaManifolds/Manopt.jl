#
#   The joint diagonalization problem on the Stiefel manifold St(n,k)
#
using Manopt, Manifolds, ManifoldsBase, LinearAlgebra

# Parameters
n = 5
k = 3
m = 50
A = randn(n,n,m)

for i = 1:m
    A[:,:,i] = 0.5*(transpose(A[:,:,i]) + A[:,:,i])
end

M = Stiefel(n,k)

function F(X::Array{Float64,2})
    f = 0
    for i = 1 : m
        f = f + norm(diag(transpose(X) * A[:, :, i] * X))^2
    end
    return -f
end

function ∇F(X::Array{Float64,2})
    g = zero_tangent_vector(M,X)
    for i = 1 : m
        g = g - 4 * A[:, :, i] * X * norm(diag(transpose(X) * A[:, :, i] * X))^2
    end
    X_g = transpose(X)*g
    return g - X * (X_g + transpose(X_g)) / 2
end

x = random_point(M)

quasi_Newton(M,F,∇F,x)
