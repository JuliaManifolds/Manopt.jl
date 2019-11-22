@testset "Manopt Trust-Region" begin
    A=[1. 2. 3.; 4. 5. 6.; 7. 8. 9.]

    prod = [Grassmannian(2, 3), Grassmannian(2, 3)]

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
        Δ_bar=4*sqrt(2*2),
        debug = [:Iteration, " ", :Cost, " | ", DebugEntry(:Δ), "\n", 1, :Stop]
    )

    @test cost(X) - 142.5 ≈ 0 atol=10.0^(-15)
end
