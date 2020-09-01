#
#   Rayleigh quotient minimization on S^n
#
using Manopt, Manifolds, ManifoldsBase

"""
    rayleigh_quotient_minimization(A)

"""

function rayleigh_quotient_minimization(A::Array{Float64,2} = randn(20, 20))

    (m, n) = size(A)

    if m != n
        throw( ErrorException("The matrix must be square, but $m != $n.") )
    end

    A = transpose(A)*A

    M = Sphere(n-1)

    function F(X::Array{Float64,1})
        return transpose(x)*A*x
    end

    function ∇F(X::Array{Float64,1})
        return 2*(A*x - x*transpose(x)*A*x)
    end

    x = random_point(M)

end
