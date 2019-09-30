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

    function egrad(X::Array{Matrix{Float64},1})
        U = X[1]
        V = X[2]
        AV = A*V
        AtU = transpose(A)*U
        return [ -AV*(transpose(AV)*U), -AtU*(transpose(AtU)*V) ];
    end

    function rgrad(M::Product, X::ProdPoint{Array{GrPoint{Float64},1}})
        eG = egrad( getValue.(getValue(X)) )
        return ProdTVector( project.(M.manifolds, getValue(X), eG) )
    end

    function e2rHess(M::Grassmannian{T},x::GrPoint{T},ξ::GrTVector{T},eGrad::Matrix{T},Hess::Matrix{T}) where T<:Union{U, Complex{U}} where U<:AbstractFloat
	    pxHess = getValue(project(M,x,Hess))
        xtGrad = getValue(x)'*eGrad
        ξxtGrad = getValue(ξ)*xtGrad
        return GrTVector(pxHess - ξxtGrad)
    end

    function eHess(X::Array{Matrix{Float64},1}, H::Array{Matrix{Float64},1})
        U = X[1]
        V = X[2]
        Udot = H[1]
        Vdot = H[2]
        AV = A*V
        AtU = transpose(A)*U
        AVdot = A*Vdot
        AtUdot = transpose(A)*Udot
        return [ -(AVdot*transpose(AV)*U + AV*transpose(AVdot)*U + AV*transpose(AV)*Udot),
                 -(AtUdot*transpose(AtU)*V + AtU*transpose(AtUdot)*V + AtU*transpose(AtU)*Vdot)
            ]
    end
    function rhess(M::Product, X::ProdPoint{Array{GrPoint{Float64},1}}, H::ProdTVector{Array{GrTVector{Float64},1}})
        eG = egrad( getValue.(getValue(X)) )
        eH = eHess( getValue.(getValue(X)), getValue.(getValue(H)) )
        return ProdTVector( e2rHess.(M.manifolds, getValue(X), getValue(H), eG, eH) )
    end

    x = randomMPoint(M)
    print("x = $x\n")
    X = trustRegions(M, cost, rgrad, x, rhess;
        Δ_bar=4*sqrt(2*p),
        useRandom=true, # it should not be activated
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

A=[1. 2. 3.; 4. 5. 6.; 7. 8. 9.]

truncated_svd(A,2)
