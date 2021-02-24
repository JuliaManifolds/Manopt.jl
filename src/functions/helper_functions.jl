
@doc raw"""
    reflect(M, f, x)

reflect the point `x` from the manifold `M` at the point `f(x)` of the
function $f\colon \mathcal M → \mathcal M$, i.e.,

````math
    \operatorname{refl}_f(x) = \operatorname{refl}_{f(x)}(x),
````
see also [`reflect`](@ref reflect(M::Manifold, p, x))`(M,p,x)`.
"""

reflect(M::Manifold, pr::Function, x) = reflect(M, pr(x), x)

@doc raw"""
    reflect(M, p, x)

reflect the point `x` from the manifold `M` at point `x`, i.e.

````math
    \operatorname{refl}_p(x) = \exp_p(-\log_p x).
````
where exp and log denote the exponential and logarithmic map on `M`.
"""
reflect(M::Manifold, p, x) = exp(M, p, -log(M, p, x))