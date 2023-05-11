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
} <: AbstractManifoldGradientObjective{E,TC,TG}
    objective::O
    p::P # a point
    X::T # a vector
    X_valid::Bool
    c::C # a value for the cost
    c_valid::Bool
end
dispatch_objective_decorator(::SimpleCacheObjective) = Val(true)

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
# A full cache objective for more than one entry and a full possibility for all fields
#
@doc raw"""
    LRUCacheObjective{E,O<:AbstractManifoldObjective{<:E},C<:NamedTuple{}} <: AbstractManifoldObjective{E}

Create a cache for an objective, based on a `NamedTuple` that stores `LRUCaches` for

# Constructor

    LRUCacheObjective(M, o::AbstractManifoldObjective, caches::Vector{Symbol}; kwargs...)

Create a cache for the [`AbstractManifoldObjective`](@ref) where the Symbols in `caches` indicate,
which function evaluations to cache.

# Keyword Arguments
* `p`           - (`rand(M)`) the type of the keys to be used in the caches. Defaults to the default representation on `M`.
* `X`           - (`zero_vector(M,p)`) the type of values to be cached for gradient and Hessian calls.
* `cache`       - (`[:Cost]`) a vector of symbols indicating which function calls should be cached.
* `cache_size`  - (`10`) number of (least recently used) calls to cache
* `cache_sizes` - a named tuple or dictionary specifying the sizes individually for each cache.
"""
struct LRUCacheObjective{E,O<:AbstractManifoldObjective{<:E},C<:NamedTuple{}} <:
       AbstractManifoldObjective{E}
    objective::O
    cache::C
end
function LRUCacheObjective(
    M::AbstractManifold,
    objective::O,
    caches::AbstractVector{<:Symbol}=[:Cost];
    p::P=rand(M),
    v::R=get_cost(M, objective, p),
    X::T=zero_vector(M, p),
    cache_size=10,
    cache_sizes=Dict{Symbol,Int}(),
) where {O<:AbstractManifoldObjective,R,P,T}
    # Initialize Caches
    lru_caches = LRU{P}[]
    for c in caches
        m = get(cache_sizes, c, cache_size)
        # Float cache, e.g. Cost
        (c === :Cost) && push!(lru_caches, LRU{P,R}(; maxsize=m))
        # Tangent Vector cache, e.g. Gradient
        (c === :Gradient) && push!(lru_caches, LRU{P,T}(; maxsize=m))
        # Arbitrary Vector Caches (constraints maybe?)
        # Point caches ?
    end
    return LRUCacheObjective(objective, NamedTuple{Tuple(caches)}(lru_caches))
end
dispatch_objective_decorator(::LRUCacheObjective) = Val(true)

#
# Default implementations - (a) check if field exists (b) try to get cache
function get_cost(M::AbstractManifold, co::LRUCacheObjective, p)
    !(haskey(co.cache, :Cost)) && return get_cost(M, co.objective, p)
    return get!(co.cache[:Cost], p) do
        get_cost(M, co.objective, p)
    end
end
get_cost_function(co::LRUCacheObjective) = get_cost_function(co.objective)

function get_gradient(M::AbstractManifold, co::LRUCacheObjective, p)
    !(haskey(co.cache, :Gradient)) && return get_gradient(M, co.objective, p)
    return get!(co.cache[:Gradient], p) do
        get_gradient(M, co.objective, p)
    end
end
function get_gradient!(M::AbstractManifold, X, co::LRUCacheObjective, p)
    !(haskey(co.cache, :Gradient)) && return get_gradient!(M, X, co.objective, p)
    copyto!(
        M,
        p,
        X,
        get!(co.cache[:Gradient], p) do
            get_gradient!(M, X, co.objective, p)
        end,
    )
    println(X)
    return X
end
get_gradient_function(co::LRUCacheObjective) = get_gradient_function(co.objective)

#
# CostGradImplementation - ToDo

#
# Factory
#
@doc raw"""
    objective_cache_factory(M::AbstractManifold, o::AbstractManifoldObjective, cache::Symbol)

Generate a cached variant of the [`AbstractManifoldObjective`](@ref) `o`
on the `AbstractManifold M` based on the symbol `cache`.

The following caches are available

* `:Simple` generates a [`SimpleCacheObjective`](@ref)
* `:LRU` generates a [`LRUCacheObjective`](@ref) where you should use the form
  `(:LRU, [:Cost, :Gradient])` to specify what should be cached or
  `(:LRU, [:Cost, :Gradient], 100)` to specify the cache size
"""
function objective_cache_factory(M, o, cache::Symbol)
    (cache === :Simple) && return SimpleCacheObjective(M, o)
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
    (cache[1] === :LRU) && return LRUCacheObjective(M, o, cache[2])
    return o
end
