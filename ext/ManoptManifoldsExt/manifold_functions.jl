"""
    default_point_distance(::DefaultManifold, p)

Following [HagerZhang:2006:2](@cite), the expected distance to the optimal solution from `p`
on `DefaultManifold` is the `Inf` norm of `p`.
"""
Manopt.default_point_distance(::Euclidean, p) = norm(p, Inf)

Manopt.default_vector_norm(::Euclidean, p, X) = norm(p, Inf)

"""
    get_bounds_index(::Hyperrectangle)

Get the bound indices of [`Hyperrectangle`](@ref) `M`. They are the same as the indices of the
lower (or upper) bounds.
"""
Manopt.get_bounds_index(M::Hyperrectangle) = eachindex(M.lb)
"""
    get_bound_t(M::Hyperrectangle, x, d, i)

Get the upper bound on moving in direction `d` from point `p` on [`Hyperrectangle`](@extref) `M`,
for the bound index `i`. There are three cases:

1. If `d[i] > 0`, the formula reads `(M.ub[i] - p[i]) / d[i]`.
2. If `d[i] < 0`, the formula reads `(M.lb[i] - p[i]) / d[i]`.
3. If `d[i] == 0`, the result is `Inf`.
"""
function Manopt.get_bound_t(M::Hyperrectangle, p, d, i)
    if d[i] > 0
        return (M.ub[i] - p[i]) / d[i]
    elseif d[i] < 0
        return (M.lb[i] - p[i]) / d[i]
    else
        return Inf
    end
end

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
function max_stepsize(M::ProbabilitySimplex)
    return 1.0
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

@doc """
    reflect(M, f, x; kwargs...)
    reflect!(M, q, f, x; kwargs...)

reflect the point `x` from the manifold `M` at the point `f(x)` of the
function ``f: $(Manopt._math(:M)) → $(Manopt._math(:M))``, given by

````math
$(Manopt._tex(:operatorname, "refl"))_f(x) = $(Manopt._tex(:operatorname, "refl"))_{f(x)}(x),
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
$(Manopt._tex(:reflect))
```

where ``$(Manopt._tex(:retr))`` and ``$(Manopt._tex(:invretr))`` denote a retraction and an inverse
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

"""
    Manopt.set_bound_t_at_index!(::Hyperrectangle, p_cp, t, d, i)

Advance the point `p_cp` on [`Hyperrectangle`](@extref) along direction `d` by stepsize `t`
only at index `i`. Used while searching for a bound during generalized Cauchy point updates.
"""
function Manopt.set_bound_t_at_index!(::Hyperrectangle, p_cp, t, d, i)
    p_cp[i] += t * d[i]
    return p_cp
end

"""
    Manopt.set_bound_at_index!(M::Hyperrectangle, p_cp, d, i)

Set element of point `p_cp` on [`Hyperrectangle`](@extref) at index `i` to the
corresponding lower or upper bound of `M` depending on the sign of direction `d` and set
that direction entry to 0.
"""
function Manopt.set_bound_at_index!(M::Hyperrectangle, p_cp, d, i)
    p_cp[i] = d[i] > 0 ? M.ub[i] : M.lb[i]
    d[i] = 0
    return p_cp
end

"""
    Manopt.bound_direction_tweak!(::Hyperrectangle, d_out, d, p, p_cp)

Set `d_out` to the difference between `p_cp` and `p`.
"""
function Manopt.bound_direction_tweak!(::Hyperrectangle, d_out, d, p, p_cp)
    return d_out .= p_cp .- p
end

"""
    Manopt.requires_generalized_cauchy_point_computation(::Hyperrectangle)

Returns `true`, as `Hyperrectangle` manifold requires generalized Cauchy point computation in solvers.
"""
Manopt.requires_generalized_cauchy_point_computation(::Hyperrectangle) = true

"""
    Manopt.get_at_bound_index(::Hyperrectangle, X, b)

Extract the element of tangent vector `X` to a point on [`Hyperrectangle`](@extref)
at index `b`.
"""
Manopt.get_at_bound_index(::Hyperrectangle, X, b) = X[b]
