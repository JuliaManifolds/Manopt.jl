function reflect end
@doc raw"""
    reflect(M, f, x; kwargs...)
    reflect!(M, q, f, x; kwargs...)

reflect the point `x` from the manifold `M` at the point `f(x)` of the
function ``f: \mathcal M → \mathcal M``, given by

````math
    \operatorname{refl}_f(x) = \operatorname{refl}_{f(x)}(x),
````

Compute the result in `q`.

see also [`reflect`](@ref reflect(M::AbstractManifold, p, x))`(M,p,x)`, to which the keywords are also passed to.
"""
reflect(M::AbstractManifold, pr::Function, x; kwargs...)

@doc raw"""
    reflect(M, p, x, kwargs...)
    reflect!(M, q, p, x, kwargs...)

Reflect the point `x` from the manifold `M` at point `p`, given by

````math
    \operatorname{refl}_p(x) = \operatorname{retr}_p(-\operatorname{retr}^{-1}_p x).
````

where ``\operatorname{retr}`` and ``\operatorname{retr}^{-1}`` denote a retraction and an inverse
retraction, respectively.
This can also be done in place of `q`.

## Keyword arguments

* `retraction_method`:         (`default_retraction_metiod(M, typeof(p))`) the retraction to use in the reflection
* `inverse_retraction_method`: (`default_inverse_retraction_method(M, typeof(p))`) the inverse retraction to use within the reflection

and for the `reflect!` additionally

* `X`:                         (`zero_vector(M,p)`) a temporary memory to compute the inverse retraction in place.
  otherwise this is the memory that would be allocated anyways.
"""
reflect(M::AbstractManifold, p::Any, x; kwargs...)
