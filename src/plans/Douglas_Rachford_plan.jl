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

@doc """
    reflect(M, p, x, kwargs...)
    reflect!(M, q, p, x, kwargs...)

Reflect the point `x` from the manifold `M` at point `p`, given by

```math
$(_tex(:reflect))_p(q) = $(_tex(:retr))_p(-$(_tex(:invretr))_p q),
```
where ``$(_tex(:retr))`` and ``$(_tex(:invretr))`` denote a retraction and an inverse retraction, respectively.

This can also be done in place of `q`.

## Keyword arguments

$(_var(:Keyword, :retraction_method))
$(_var(:Keyword, :inverse_retraction_method))

and for the `reflect!` additionally

* `X=zero_vector(M,p)`: a temporary memory to compute the inverse retraction in place.
  otherwise this is the memory that would be allocated anyways.
"""
reflect(M::AbstractManifold, p::Any, x; kwargs...)
