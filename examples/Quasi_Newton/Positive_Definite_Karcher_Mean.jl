#
#   Computes a Karcher mean of a collection of positive definite matrices
#
using Manopt, Manifolds, ManifoldsBase

"""
    positive_definite_karcher_mean(A)

"""

function positive_definite_karcher_mean(A::Array{Float64,3} = randn(20, 20, 10))

    (m, n, k) = size(A)

    if m != n
        throw( ErrorException("The matrices must be square, but $m != $n.") )
    end


    # We need positive definite datrices
    for i = 1:k
        A(:,:,i) = transpose(A(:,:,i))*A(:,:,i)
    end

    M = SymmetricPositiveDefinite(m)

    function F(X::Array{Float64,2})
        f = 0
        for i = 1 : k
            f = f + distance(X, A(:, :, i))^2
        end
        f = f/(2*k)
    end

    function ∇F(X::Array{Float64,2})
        zero_tangent_vector(M,X)
        for i = 1 : k
            g = g - 1/k * log(M, X, A(:, :, i))
        end
    end

    x = random_point(M)

end
