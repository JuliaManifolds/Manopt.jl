@doc raw"""
    reflect(M, f, x)
    reflect!(M, q, f, x)

reflect the point `x` from the manifold `M` at the point `f(x)` of the
function ``f: \mathcal M → \mathcal M``, i.e.,

````math
    \operatorname{refl}_f(x) = \operatorname{refl}_{f(x)}(x),
````

Compute the result in `q`.

see also [`reflect`](@ref reflect(M::AbstractManifold, p, x))`(M,p,x)`.
"""
reflect(M::AbstractManifold, pr::Function, x) = reflect(M, pr(x), x)
reflect!(M::AbstractManifold, q, pr::Function, x) = reflect!(M, q, pr(x), x)

@doc raw"""
    reflect(M, p, x)
    reflect!(M, q, p, x)

Reflect the point `x` from the manifold `M` at point `p`, i.e.

````math
    \operatorname{refl}_p(x) = \exp_p(-\log_p x).
````

where exp and log denote the exponential and logarithmic map on `M`.
This can also be done in place of `q`.
"""
reflect(M::AbstractManifold, p, x) = exp(M, p, -log(M, p, x))
reflect!(M::AbstractManifold, q, p, x) = exp!(M, q, p, -log(M, p, x))

function reflect(M::AbstractManifold, p, x;
    rectraction_method=default_retract_method(M),
    inverse_retraction_method=default_inverse_retraction_method(M)
)
    return retract(M, p, -inverse_retract(M, p, x, inverse_retraction_method), rectraction_method)
end
