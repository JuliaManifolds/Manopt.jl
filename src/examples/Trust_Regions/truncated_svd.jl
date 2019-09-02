#
#   SVD decomposition of a matrix truncated to a rank
#
using Manopt
import LinearAlgebra: norm, svd, Diagonal
export truncated_svd

@doc doc"""
    truncated_svd(A, p)

return a singular value decomposition of a real valued matrix A truncated to
rank p.

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

    function cost(X::ProdPoint{Array{GrPoint{Float64},1}})
        U = getValue(getValue(X)[1])
        V = getValue(getValue(X)[2])
        return -0.5 * norm(transpose(U) * A * V)^2
    end

    function egrad(X::ProdPoint{Array{GrPoint{Float64},1}})
        U = getValue(getValue(X)[1])
        V = getValue(getValue(X)[2])
        AV = A*V
        AtU = transpose(A)*U
        return ProdTVector( [ GrTVector(-AV*(transpose(AV)*U)), GrTVector(-AtU*(transpose(AtU)*V))] )
    end

    function ehess(X::ProdPoint{Array{GrPoint{Float64},1}}, H::ProdTVector{Array{GrTVector{Float64},1}})
        U = getValue(getValue(X)[1])
        V = getValue(getValue(X)[2])
        Udot = getValue(getValue(H)[1])
        Vdot = getValue(getValue(H)[2])
        AV = A*V
        AtU = transpose(A)*U
        AVdot = A*Vdot
        AtUdot = transpose(A)*Udot
        return ProdTVector( [
            GrTVector( -(AVdot*transpose(AV)*U + AV*transpose(AVdot)*U + AV*transpose(AV)*Udot)),
            GrTVector( -(AtUdot*transpose(AtU)*V + AtU*transpose(AtUdot)*V + AtU*transpose(AtU)*Vdot))
        ] )
    end

    x = randomMPoint(M)
    X = trustRegions(M, cost, egrad, x, ehess;
        Δ_bar=4*sqrt(2*p),
        debug = [:Iteration, " ", :Cost, "\n", 1, :Stop] 
    )

    U = getValue(getValue(X)[1])
    V = getValue(getValue(X)[2])

    Spp = transpose(U)*A*V
    SVD = svd(Spp)
    U = U*SVD.U
    S = SVD.S
    V = V*SVD.V

    return [U, S, V]
end
