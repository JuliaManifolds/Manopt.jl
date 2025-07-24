"""
    max_stepsize(M::TangentBundle, p)

Tangent bundle has injectivity radius of either infinity (for flat manifolds) or 0
(for non-flat manifolds). This makes a guess of what a reasonable maximum stepsize
on a tangent bundle might be.
"""
function max_stepsize(M::TangentBundle, p)
    return max_stepsize(M.manifold, p[M, :point])
end
function max_stepsize(M::TangentBundle)
    return max_stepsize(M.manifold)
end
"""
    max_stepsize(M::FixedRankMatrices, p)

Return a reasonable guess of maximum step size on `FixedRankMatrices` following
the choice of typical distance in Matlab Manopt, the dimension of `M`. See
[this note](https://github.com/NicolasBoumal/manopt/blob/97b6eb6b185334ab7b3991585ed2c044d69ee905/manopt/manifolds/fixedrank/fixedrankembeddedfactory.m#L76-L78)
"""
function max_stepsize(M::FixedRankMatrices, p)
    return max_stepsize(M)
end
max_stepsize(M::FixedRankMatrices) = manifold_dimension(M)

"""
    max_stepsize(M::Hyperrectangle, p)

The default maximum stepsize for `Hyperrectangle` manifold with corners is maximum
of distances from `p` to each boundary.
"""
function max_stepsize(M::Hyperrectangle, p)
    ms = 0.0
    for i in eachindex(M.lb, p)
        dist_ub = M.ub[i] - p[i]
        if dist_ub > 0
            ms = max(ms, dist_ub)
        end
        dist_lb = p[i] - M.lb[i]
        if dist_lb > 0
            ms = max(ms, dist_lb)
        end
    end
    return ms
end
function max_stepsize(M::Hyperrectangle)
    ms = 0.0
    for i in eachindex(M.lb, M.ub)
        ms = max(ms, M.ub[i] - M.lb[i])
    end
    return ms
end

"""
    mid_point(M, p, q, x)
    mid_point!(M, y, p, q, x)

Compute the mid point between `p` and `q`. If there is more than one mid point
of (not necessarily minimizing) geodesics (for example on the sphere), the one nearest
to `x` is returned (in place of `y`).
"""
mid_point(M::AbstractManifold, p, q, ::Any) = mid_point(M, p, q)
mid_point!(M::AbstractManifold, y, p, q, ::Any) = mid_point!(M, y, p, q)

function mid_point(M::Circle, p, q, x)
    if distance(M, p, q) ≈ π
        X = 0.5 * log(M, p, q)
        Y = log(M, p, x)
        return exp(M, p, (sign(X) == sign(Y) ? 1 : -1) * X)
    end
    return mid_point(M, p, q)
end

function mid_point(M::Sphere, p, q, x)
    if isapprox(M, p, -q)
        X = log(M, p, x) / distance(M, p, x) * π
    else
        X = log(M, p, q)
    end
    return exp(M, p, 0.5 * X)
end
function mid_point!(M::Sphere, y, p, q, x)
    if isapprox(M, p, -q)
        X = log(M, p, x) / distance(M, p, x) * π
    else
        X = log(M, p, q)
    end
    y .= exp(M, p, 0.5 * X)
    return y
end

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
reflect(M::AbstractManifold, pr::Function, x; kwargs...) = reflect(M, pr(x), x; kwargs...)
function reflect!(M::AbstractManifold, q, pr::Function, x; kwargs...)
    return reflect!(M, q, pr(x), x; kwargs...)
end

@doc """
    reflect(M, p, x, kwargs...)
    reflect!(M, q, p, x, kwargs...)

Reflect the point `x` from the manifold `M` at point `p`, given by

```math
$(_tex(:reflect))
```

where ``$(_tex(:retr))`` and ``$(_tex(:invretr))`` denote a retraction and an inverse
retraction, respectively.
This can also be done in place of `q`.

## Keyword arguments

$(_var(:Keyword, :retraction_method))
$(_var(:Keyword, :inverse_retraction_method))

and for the `reflect!` additionally

$(_var(:Keyword, :X))
  as temporary memory to compute the inverse retraction in place.
  otherwise this is the memory that would be allocated anyways.
"""
function reflect(
        M::AbstractManifold,
        p,
        x;
        retraction_method = default_retraction_method(M, typeof(p)),
        inverse_retraction_method = default_inverse_retraction_method(M, typeof(p)),
        X = nothing,
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
        retraction_method = default_retraction_method(M, typeof(p)),
        inverse_retraction_method = default_inverse_retraction_method(M),
        X = zero_vector(M, p),
    )
    inverse_retract!(M, X, p, x, inverse_retraction_method)
    X .*= -1
    return retract!(M, q, p, X, retraction_method)
end
