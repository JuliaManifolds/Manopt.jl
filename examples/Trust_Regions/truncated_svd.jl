#
#   SVD decomposition of a matrix truncated to a rank
#
using Manopt
import LinearAlgebra: norm, svd, Diagonal
export truncated_svd

"""
    truncated_svd(A, p)

return a singular value decomposition of a real valued matrix A truncated to
rank p.

# Input
* `A` – a real-valued matrix A of size mxn
* `p` – an integer p in min { m, n }

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

    M = ProductManifold(Grassmannian(p, m), Grassmannian(p, n))

    function cost(X::ProductRepr)
        return cost([submanifold_components(X)...])
    end
    function cost(X::Array{Matrix{Float64},1})
        return -0.5 * norm(transpose(X[1]) * A * X[2])^2
    end

    function egrad(X::Array{Matrix{Float64},1})
        U = X[1]
        V = X[2]
        AV = A*V
        AtU = transpose(A)*U
        return [ -AV*(transpose(AV)*U), -AtU*(transpose(AtU)*V) ];
    end

    function rgrad(M::ProductManifold, X::ProductRepr)
        eG = egrad([submanifold_components(M,X)...])
        x = [submanifold_components(M,X)...]
        return Manifolds.ProductRepr(project.(M.manifolds, x, eG)...)
    end

    function e2rHess(M::Grassmann, x, ξ, eGrad::Matrix{T},Hess::Matrix{T}) where T<:Union{U, Complex{U}} where U<:AbstractFloat
        pxHess = project(M,x,Hess)
        xtGrad = x'*eGrad
        ξxtGrad = ξ*xtGrad
        return pxHess - ξxtGrad
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

    function rhess(M::ProductManifold, X::ProductRepr, H::ProductRepr)
        x = [submanifold_components(M,X)...]
        h = [submanifold_components(M,H)...]
        eG = egrad(x)
        eH = eHess(x,h)
        return Manifolds.ProductRepr(e2rHess.(M.manifolds, x, h, eG, eH)...)
    end

    x = random_point(M)
    print("x = $x\n")
    X = trust_regions(M, cost, rgrad, x, rhess;
        Δ_bar=4*sqrt(2*p),
        debug = [:Iteration, " ", :Cost, " | ", DebugEntry(:Δ), "\n", 1, :Stop]
    )

    U = X[1]
    V = X[2]

    Spp = transpose(U)*A*V
    SVD = svd(Spp)
    U = U*SVD.U
    S = SVD.S
    V = V*SVD.V

    return [U, S, V]
end

A=[1. 2. 3. 4.; 5. 6. 7. 8.; 9. 10. 11. 12.; 13. 14. 15. 16.]

truncated_svd(A,2)
