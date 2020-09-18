#
#   The joint diagonalization problem on the Stiefel manifold St(n,k)
#
using Manopt, Manifolds, ManifoldsBase

# Parameters
n = 5
k = 3
m = 50
A = randn(n,n,m)

for i = 1:k
    A[:,:,i] = 0,5*(transpose(A[:,:,i]) + A[:,:,i])
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
    
end

x = random_point(M)

quasi_Newton(M,F,∇F,x)
