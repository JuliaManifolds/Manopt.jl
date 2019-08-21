#
#   SVD decomposition of a matrix truncated to a rank
#
using Manopt
import LinearAlgebra: norm, svd
export truncated_svd

@doc doc"""
    truncated_svd(A, p)

# Input
* `A` – a real-valued matrix A of size mxn
* `p` – an integer $p \leq min\{ m, n \}$

# Output
* `U` – an orthonormal matrix of size mxp
* `V` – an orthonormal matrix of size nxp
* `S` – a diagonal matrix of size pxp with nonnegative and decreasing diagonal
        entries
"""

function truncated_svd(A::Array{Float64,2} = randn(42, 60), p::Int64 = 5)
    (m, n) = size(A)

    if p > min(m,n)
        throw( ErrorException("The Rank p=$p must be smaller than the smallest dimension of A = $min(m, n).") )
    end

    prod = [Grassmannian(p, m), Grassmannian(p, n)]

    M = Product(prod)

    function cost(X::Array{AbstractArray{Float64,2},1})
        U = X[1]
        V = X[2]
        return -.5 * norm(transpose(U) * A * V)^2
    end

    function egrad(X::Array{AbstractArray{Float64,2},1})
        U = X[1]
        V = X[2]
        AV = A*V
        AtU = transpose(A)*U
        return [-AV*(transpose(AV)*U), -AtU*(transpose(AtU)*V)]
    end

    function ehess(X::Array{AbstractArray{Float64,2},1}, H::Array{AbstractArray{Float64,2},1})
        U = X[1]
        V = X[2]
        Udot = H[1]
        Vdot = H[2]
        AV = A*V
        AtU = transpose(A)*U
        AVdot = A*Vdot
        AtUdot = transpose(A)*Udot
        return [-(AVdot*transpose(AV)*U + AV*transpose(AVdot)*U + AV*transpose(AV)*Udot), -(AtUdot*transpose(AtU)*V + AtU*transpose(AtUdot)*V + AtU*transpose(AtU)*Vdot)]
    end

    X = trustRegionsSolver(M, cost, egrad, randomMPoint(M), ehess, x -> x,
    stopWhenAny(stopAfterIteration(5000), stopGradientTolerance(10^(-6))),
    4*sqrt(2*p))

    U = X[1]
    V = X[2]

    Spp = transpose(U)*A*V
    SVD = svd(Spp)
    U = U*SVD.U
    S = SVD.S
    V = V*SVD.V

    return [U, S, V]
end
