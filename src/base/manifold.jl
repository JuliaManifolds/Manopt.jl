# Helpers for general manifold functions
"""
    UnitVector{TB}

A type representing a unit tangent vector on a `Hyperrectangle`-like manifold with corners,
or a product of it with a standard manifold.
The field `index` stores the index of the element equal to 1.
All other elements are equal to 0.
"""
struct UnitVector{TB}
    index::TB
end

function add_vector!(M::AbstractManifold, X, p, c, basis::AbstractBasis)
    Y = get_vector(M, p, c, basis)
    X .+= Y
    return X
end
function add_vector!(M::ProductManifold, X, p, c, basis::AbstractBasis)
    dims = map(manifold_dimension, M.manifolds)
    @assert length(c) == sum(dims)
    dim_ranges = ManifoldsBase._get_dim_ranges(dims)
    tc = map(dr -> (@inbounds view(c, dr)), dim_ranges)
    ts = ManifoldsBase.ziptuples(
        M.manifolds,
        submanifold_components(M, X),
        submanifold_components(M, p),
        tc,
    )
    map(ts) do t
        return add_vector!(t..., basis)
    end
    return X
end
function add_coordinates!(M::AbstractManifold, c, p, X, basis::AbstractBasis)
    cX = get_coordinates(M, p, X, basis)
    c .+= cX
    return c
end

"""
    has_anisotropic_max_stepsize(M::AbstractManifold)

Return `true` if `M` has a `max_stepsize` that depends on the direction,
for example if `M` is a [`Hyperrectangle`](@extref `Manifolds.Hyperrectangle`)-like manifold with corners, or a product of it
with a standard manifold. Otherwise return `false`.
"""
has_anisotropic_max_stepsize(::AbstractManifold) = false
has_anisotropic_max_stepsize(M::ProductManifold) = any(has_anisotropic_max_stepsize, M.manifolds)

# whether the vector transport `m` is the parallel transport, on product manifolds in every
# component; it is isometric, so for example the cautious quasi-Newton update does not need
# to re-test its stored pairs
_is_parallel_transport(::AbstractVectorTransportMethod) = false
_is_parallel_transport(::ParallelTransport) = true
function _is_parallel_transport(m::ManifoldsBase.ProductVectorTransport)
    return all(_is_parallel_transport, m.methods)
end

_doc_project_truncate = raw"""
    project_truncate(M::AbstractManifold, p, X)
    project_truncate!(M::AbstractManifold, Y, p, X)

Project the tangent vector `X` at `p` onto the set of tangent vectors ``Z`` for which
``\exp_p Z`` satisfies the box constraints of `M`, that is, compute
```math
Y = \log_p P(\exp_p X),
```
where ``P`` denotes the projection onto the box constraints.
The in-place variant can also be called with `Y` and `X` being the same tangent vector.

* On a [`Hyperrectangle`](@extref `Manifolds.Hyperrectangle`) with bounds ``l`` and ``u``,
  this is ``Y = \operatorname{clamp}(p + X, l, u) - p``.
* On a product manifold, `Y` is computed componentwise.
* On manifolds without box constraints, this is ``Y = X``, projected onto the tangent
  space at `p` with `embed_project!` for numerical stability.

The set `X` is projected onto is contained in the tangent cone at `p`. Unlike the tangent
cone, it changes continuously with `p`, and `Y` depends continuously on `p` and `X`.

For example, for the negative gradient ``X = -\operatorname{grad} f(p)``, `Y` is the
projected negative gradient of L-BFGS-B [ByrdLuNocedalZhu:1995](@cite). Its norm vanishes
exactly at the stationary points of the problem and is used as a criticality measure by
[`StopWhenProjectedNegativeGradientNormLess`](@ref) and
[`model_decrease_reference`](@ref).
"""

@doc "$(_doc_project_truncate)"
function project_truncate(M::AbstractManifold, p, X)
    return project_truncate!(M, allocate_result(M, project_truncate, X, p), p, X)
end
@doc "$(_doc_project_truncate)"
function project_truncate!(M::AbstractManifold, Y, p, X)
    embed_project!(M, Y, p, X)
    return Y
end
function project_truncate!(M::ProductManifold, Y, p, X)
    map(
        project_truncate!,
        M.manifolds,
        submanifold_components(M, Y),
        submanifold_components(M, p),
        submanifold_components(M, X),
    )
    return Y
end
