@doc raw"""
    AdaptiveRegularizzationCubiCCost

```math
    m(X) = <X, g> + .5 <X, H[X]> +  \frac{σ}{3} \lVert η \rVert^3
```

where `g` is a tangent vector (usually a gradient) and `H` is a matrix (usually the Hessian
of some function ``f``).
"""
struct AdaptiveRegularizzationCubiCCost{F,R}
    gradnorm::R
    ς::R
    Tmatrix::T #submatrix
    y::Y # Solution of of argmin m(s), s= sum y[i]q[i]

end