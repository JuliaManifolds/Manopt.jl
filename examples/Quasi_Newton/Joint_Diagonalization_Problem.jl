#
#   The joint diagonalization problem on the Stiefel manifold St(n,k)
#
using Manopt, Manifolds, ManifoldsBase

# Parameters
n = 20
k = 10
A = randn(n,n,k)

if m != n
    throw( ErrorException("The matrices must be square, but $m != $n.") )
end
# We need symmetric matrices
for i = 1:k
    A(:,:,i) = 0,5*(transpose(A(:,:,i)) + A(:,:,i))
end