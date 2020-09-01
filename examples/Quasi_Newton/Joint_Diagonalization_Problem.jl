#
#   The joint diagonalization problem on the Stiefel manifold St(n,k)
#
using Manopt, Manifolds, ManifoldsBase

"""
    joint_diagonalization_problem(A)

"""

function joint_diagonalization_problem(A::Array{Float64,3} = randn(20, 20, 10))

    (m, n, k) = size(A)

    if m != n
        throw( ErrorException("The matrices must be square, but $m != $n.") )
    end

    # We need symmetric matrices
    for i = 1:k
        A(:,:,i) = 0,5*(transpose(A(:,:,i)) + A(:,:,i))
    end

    

end
