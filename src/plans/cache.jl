#
# A Simple Cache for Objectives
#
@doc raw"""
     SimpleCacheObjective{O<:AbstractManifoldGradientObjective{E,TC,TG}, P, T,C} <: AbstractManifoldGradientObjective{E,TC,TG}

Provide a simple cache for an [`AbstractManifoldGradientObjective`](@ref) that is for a given point `p` this cache
stores a point `p` and a gradient ``\operatorname{grad} f(p)`` in `X` as well as a cost value ``f(p)`` in `c`.

Both `X` and `c` are accompanied by booleans to keep track of their validity.

# Constructor

    SimpleCacheObjective(M::AbstractManifold, obj::AbstractManifoldGradientObjective; kwargs...)

## Keyword
* `p` (`rand(M)`) – a point on the manifold to initialize the cache with
* `X` (`get_gradient(M, obj, p)` or `zero_vector(M,p)`) – a tangent vector to store the gradient in, see also `initialize`
* `c` (`get_cost(M, obj, p)` or `0.0`) – a value to store the cost function in `initialize`
* `initialized` (`true`) – whether to initialize the cached `X` and `c` or not.
"""
mutable struct SimpleCacheObjective{
    E<:AbstractEvaluationType,TC,TG,O<:AbstractManifoldGradientObjective{E,TC,TG},P,T,C
} <: AbstractDecoratedManifoldObjective{E,O}
    objective::O
    p::P # a point
    X::T # a vector
    X_valid::Bool
    c::C # a value for the cost
    c_valid::Bool
end

function SimpleCacheObjective(
    M::AbstractManifold,
    obj::O;
    initialized=true,
    p=rand(M),
    X=initialized ? get_gradient(M, obj, p) : zero_vector(M, p),
    c=initialized ? get_cost(M, obj, p) : 0.0,
) where {E<:AbstractEvaluationType,TC,TG,O<:AbstractManifoldGradientObjective{E,TC,TG}}
    q = copy(M, p)
    return SimpleCacheObjective{E,TC,TG,O,typeof(q),typeof(X),typeof(c)}(
        obj, q, X, initialized, c, initialized
    )
end
# Default implementations
function get_cost(M::AbstractManifold, sco::SimpleCacheObjective, p)
    scop_neq_p = sco.p != p
    if scop_neq_p || !sco.c_valid
        # else evaluate cost, invalidate grad if p changed
        sco.c = get_cost(M, sco.objective, p)
        # if we switched points, invalidate X
        scop_neq_p && (sco.X_valid = false)
        copyto!(M, sco.p, p)
        sco.c_valid = true
    end
    return sco.c
end
get_cost_function(sco::SimpleCacheObjective) = get_cost_function(sco.objective)
function get_gradient(M::AbstractManifold, sco::SimpleCacheObjective, p)
    scop_neq_p = sco.p != p
    if scop_neq_p || !sco.X_valid
        get_gradient!(M, sco.X, sco.objective, p)
        # if we switched points, invalidate c
        scop_neq_p && (sco.c_valid = false)
        copyto!(M, sco.p, p)
        sco.X_valid = true
    end
    return copy(M, p, sco.X)
end
function get_gradient!(M::AbstractManifold, X, sco::SimpleCacheObjective, p)
    scop_neq_p = sco.p != p
    if scop_neq_p || !sco.X_valid
        get_gradient!(M, sco.X, sco.objective, p)
        scop_neq_p && (sco.c_valid = false)
        copyto!(M, sco.p, p)
        # if we switched points, invalidate c
        sco.X_valid = true
    end
    copyto!(M, X, sco.p, sco.X)
    return X
end
get_gradient_function(sco::SimpleCacheObjective) = get_gradient_function(sco.objective)

#
# CostGradImplementation
#
function get_cost(
    M::AbstractManifold,
    sco::SimpleCacheObjective{AllocatingEvaluation,TC,TG,<:ManifoldCostGradientObjective},
    p,
) where {TC,TG}
    scop_neq_p = sco.p != p
    if scop_neq_p || !sco.c_valid
        sco.c, sco.X = sco.objective.costgrad!!(M, p)
        copyto!(M, sco.p, p)
        sco.X_valid = true
        sco.c_valid = true
    end
    return sco.c
end
function get_cost(
    M::AbstractManifold,
    sco::SimpleCacheObjective{InplaceEvaluation,TC,TG,<:ManifoldCostGradientObjective},
    p,
) where {TC,TG}
    scop_neq_p = sco.p != p
    if scop_neq_p || !sco.c_valid
        sco.c, _ = sco.objective.costgrad!!(M, sco.X, p)
        copyto!(M, sco.p, p)
        sco.X_valid = true
        sco.c_valid = true
    end
    return sco.c
end
function get_gradient(
    M::AbstractManifold,
    sco::SimpleCacheObjective{AllocatingEvaluation,TC,TG,<:ManifoldCostGradientObjective},
    p,
) where {TC,TG}
    scop_neq_p = sco.p != p
    if scop_neq_p || !sco.X_valid
        sco.c, sco.X = sco.objective.costgrad!!(M, p)
        copyto!(M, sco.p, p)
        # if we switched points, invalidate c
        sco.X_valid = true
        sco.c_valid = true
    end
    return copy(M, p, sco.X)
end
function get_gradient(
    M::AbstractManifold,
    sco::SimpleCacheObjective{InplaceEvaluation,TC,TG,<:ManifoldCostGradientObjective},
    p,
) where {TC,TG}
    scop_neq_p = sco.p != p
    if scop_neq_p || !sco.X_valid
        sco.c, _ = sco.objective.costgrad!!(M, sco.X, p)
        copyto!(M, sco.p, p)
        # if we switched points, invalidate c
        sco.X_valid = true
        sco.c_valid = true
    end
    return sco.X
end
function get_gradient!(
    M::AbstractManifold,
    X,
    sco::SimpleCacheObjective{AllocatingEvaluation,TC,TG,<:ManifoldCostGradientObjective},
    p,
) where {TC,TG}
    scop_neq_p = sco.p != p
    if scop_neq_p || !sco.X_valid
        sco.c, sco.X = sco.objective.costgrad!!(M, p)
        copyto!(M, sco.p, p)
        sco.X_valid = true
        sco.c_valid = true
    end
    copyto!(M, X, sco.p, sco.X)
    return X
end
function get_gradient!(
    M::AbstractManifold,
    X,
    sco::SimpleCacheObjective{InplaceEvaluation,TC,TG,<:ManifoldCostGradientObjective},
    p,
) where {TC,TG}
    scop_neq_p = sco.p != p
    if scop_neq_p || !sco.X_valid
        sco.c, _ = sco.objective.costgrad!!(M, sco.X, p)
        sco.p = p
        sco.X_valid = true
        sco.c_valid = true
    end
    copyto!(M, X, sco.p, sco.X)
    return X
end

#
# Factory
@doc raw"""
    objective_cache_factory(M::AbstractManifold, o::AbstractManifoldObjective, cache::Symbol)

Generate a cached variant of the [`AbstractManifoldObjective`](@ref) `o`
on the `AbstractManifold M` based on the symbol `cache`.

The following caches are available

* `:Simple` generates a [`SimpleCacheObjective`](@ref)
* `:LRU` generates a [`LRUCacheObjective`](@ref) where you should use the form
  `(:LRU, [:Cost, :Gradient])` to specify what should be cached or
  `(:LRU, [:Cost, :Gradient], 100)` to specify the cache size.
  Here this variant defaults to `(:LRU, [:Cost, :Gradient], 100)`,
  i.e. to cache up to 100 cost and gradient values.
"""
function objective_cache_factory(M, o, cache::Symbol)
    (cache === :Simple) && return SimpleCacheObjective(M, o)
    (cache === :LRU) && return LRUCacheObjective(M, o, [:Cost, :Gradient]; cache_size=100)
    return o
end

@doc raw"""
    objective_cache_factory(M::AbstractManifold, o::AbstractManifoldObjective, cache::Tuple{Symbol, Array, Array})
    objective_cache_factory(M::AbstractManifold, o::AbstractManifoldObjective, cache::Tuple{Symbol, Array})

Generate a cached variant of the [`AbstractManifoldObjective`](@ref) `o`
on the `AbstractManifold M` based on the symbol `cache[1]`,
where the second element `cache[2]` is are further arguments to  the cache and
the optional third is passed down as keyword arguments.

For all available caches see the simpler variant with symbols.
"""
function objective_cache_factory(M, o, cache::Tuple{Symbol,<:AbstractArray,<:AbstractArray})
    (cache[1] === :Simple) && return SimpleCacheObjective(M, o; cache[3]...)
    if (cache[1] === :LRU)
        if (cacge[3] isa Integer)
            return LRUCacheObjective(M, o, cache[2]; cache_size=cache[3])
        else
            return LRUCacheObjective(M, o, cache[2]; cache[3]...)
        end
    end
    return o
end
function objective_cache_factory(M, o, cache::Tuple{Symbol,<:AbstractArray})
    (cache[1] === :Simple) && return SimpleCacheObjective(M, o)
    (cache[1] === :LRU) && return objective_cache_factory(M, o, Val(:LRU), cache[2])
    return o
end
