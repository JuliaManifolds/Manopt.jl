@doc raw"""
    reflect(M, f, x; kwargs...)
    reflect!(M, q, f, x; kwargs...)

reflect the point `x` from the manifold `M` at the point `f(x)` of the
function ``f: \mathcal M → \mathcal M``, i.e.,

````math
    \operatorname{refl}_f(x) = \operatorname{refl}_{f(x)}(x),
````

Compute the result in `q`.

see also [`reflect`](@ref reflect(M::AbstractManifold, p, x))`(M,p,x)`, to which the keywords are also passed to.
"""
reflect(M::AbstractManifold, pr::Function, x; kwargs...) = reflect(M, pr(x), x; kwargs...)
function reflect!(M::AbstractManifold, q, pr::Function, x; kwargs...)
    return reflect!(M, q, pr(x), x; kwargs...)
end

@doc raw"""
    reflect(M, p, x, kwargs...)
    reflect!(M, q, p, x, kwargs...)

Reflect the point `x` from the manifold `M` at point `p`, i.e.

````math
    \operatorname{refl}_p(x) = \operatorname{retr}_p(-\operatorname{retr}^{-1}_p x).
````

where ``\operatorname{retr}`` and ``\operatorname{retr}^{-1}`` denote a retraction and an inverse
retraction, respectively.
This can also be done in place of `q`.

## Keyword arguments
* `retraction_method` - (`default_retraction_metiod(M, typeof(p))`) the retraction to use in the reflection
* `inverse_retraction_method` - (`default_inverse_retraction_method(M, typeof(p))`) the inverse retraction to use within the reflection

and for the `reflect!` additionally

* `X` (`zero_vector(M,p)`) a temporary memory to compute the inverse retraction in place.
  otherwise this is the memory that would be allocated anyways.

Passing `X` to `reflect` will just have no effect.
"""
function reflect(
    M::AbstractManifold,
    p,
    x;
    retraction_method=default_retraction_method(M, typeof(p)),
    inverse_retraction_method=default_inverse_retraction_method(M, typeof(p)),
    X=nothing,
)
    return retract(
        M, p, -inverse_retract(M, p, x, inverse_retraction_method), retraction_method
    )
end
function reflect!(
    M::AbstractManifold,
    q,
    p,
    x;
    retraction_method=default_retraction_method(M),
    inverse_retraction_method=default_inverse_retraction_method(M),
    X=zero_vector(M, p),
)
    inverse_retract!(M, X, p, x, inverse_retraction_method)
    X .*= -1
    return retract!(M, q, p, X, retraction_method)
end
