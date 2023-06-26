@doc raw"""
    AdaptiveRegularizzationCubiCCost

```math
    m(X) = <X, g> + .5 <X, H[X]> +  \frac{σ}{3} \lVert η \rVert^3
```

where `g` is a tangent vector (usually a gradient) and `H` is a matrix (usually the Hessian
of some function ``f``).
"""
struct AdaptiveRegularizationCubicCost{F,R}
    gradnorm::R
    ς::R
    Tmatrix::T #submatrix
    y::Y # Solution of of argmin m(s), s= sum y[i]q[i]
end
mutable struct CubicSubCost{Y,T,I,R}
    k::I #number of Lanczos vectors
    gradnorm::R
    σ::R
    Tmatrix::T #submatrix
    y::Y # Solution of of argmin m(s), s= sum y[i]q[i]
end
function (C::CubicSubCost)(::AbstractManifold, y)# Ronny: M is Euclidean (R^k) but p should be y. I can change it to y a just input c.y when computing the subcost
    #C.y[1]*C.gradnorm + 0.5*dot(C.y[1:C.k],@view(C.Tmatrix[1:C.k,1:C.k])*C.y[1:C.k]) + C.σ/3*norm(C.y[1:C.k],2)^3
    return y[1] * C.gradnorm +
           0.5 * dot(y, @view(C.Tmatrix[1:(C.k), 1:(C.k)]) * y) +
           C.σ / 3 * norm(y, 2)^3
end
