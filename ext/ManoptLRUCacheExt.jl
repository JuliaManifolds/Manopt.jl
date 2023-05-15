module ManoptLRUCache

if isdefined(Base, :get_extension)
        using LRUCache
else
    # imports need to be relative for Requires.jl-based workflows:
    # https://github.com/JuliaArrays/ArrayInterface.jl/pull/387
    using ..LRUCache
end

@doc raw"""
    LRUCacheObjective{E,P,O<:AbstractManifoldObjective{<:E},C<:NamedTuple{}} <: AbstractDecoratedManifoldObjective{E,P}
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
struct LRUCacheObjective{E,P,O<:AbstractManifoldObjective{<:E},C<:NamedTuple{}} <:
       AbstractDecoratedManifoldObjective{E,P}
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
) where {E,O<:AbstractManifoldObjective{E},R,P,T}
    c = init_caches(caches; p=p, v=v, X=X, cache_size=cache_size, cache_sizes=cache_sizes)
    return LRUCacheObjective{E,O,O,typeof(c)}(objective, c)
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
) where {E,O2,O<:AbstractDecoratedManifoldObjective{E,O2},R,P,T}
    c = init_caches(caches; p=p, v=v, X=X, cache_size=cache_size, cache_sizes=cache_sizes)
    return LRUCacheObjective{E,O2,O,typeof(c)}(objective, c)
end
function init_caches(
    caches;
    p::P=rand(M),
    v::R=get_cost(M, objective, p),
    X::T=zero_vector(M, p),
    cache_size=10,
    cache_sizes=Dict{Symbol,Int}(),
) where {P,R,T}
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
    return NamedTuple{Tuple(caches)}(lru_caches)
end
get_gradient_function(co::LRUCacheObjective) = get_gradient_function(co.objective)
get_cost_function(co::LRUCacheObjective) = get_cost_function(co.objective)

#
# Default implementations - (a) check if field exists (b) try to get cache
function get_cost(M::AbstractManifold, co::LRUCacheObjective, p)
    !(haskey(co.cache, :Cost)) && return get_cost(M, co.objective, p)
    return get!(co.cache[:Cost], p) do
        get_cost(M, co.objective, p)
    end
end

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
        X,
        p,
        get!(co.cache[:Gradient], p) do
            get_gradient!(M, X, co.objective, p)
        end,
    )
    return X
end

#
# CostGradImplementation
function get_cost(
    M::AbstractManifold, co::LRUCacheObjective{E,<:ManifoldCostGradientObjective}, p
) where {E<:AbstractEvaluationType}
    #Neither cost not grad cached -> evaluate
    all(.!(haskey.(Ref(co.cache), [:Cost, :Gradient]))) &&
        return get_cost(M, co.objective, p)
    return get!(co.cache[:Cost], p) do
        c, X = get_cost_and_gradient(M, co.objective, p)
        #if this is evaluated, we can also set X
        haskey(co.cache, :Gradient) && setindex!(co.cache[:Gradient], X, p)
        c #but we also set the new cost here
    end
end
function get_gradient(
    M::AbstractManifold, co::LRUCacheObjective{E,<:ManifoldCostGradientObjective}, p
) where {E<:AllocatingEvaluation}
    all(.!(haskey.(Ref(co.cache), [:Cost, :Gradient]))) &&
        return get_gradient(M, co.objective, p)
    return get!(co.cache[:Gradient], p) do
        c, X = get_cost_and_gradient(M, co.objective, p)
        #if this is evaluated, we can also set c
        haskey(co.cache, :Cost) && setindex!(co.cache[:Cost], c, p)
        X #but we also set the new cost here
    end
end
function get_gradient!(
    M::AbstractManifold, X, co::LRUCacheObjective{E,<:ManifoldCostGradientObjective}, p
) where {E}
    All(!(haskey.(Ref(co.cache), [:Cost, :Gradient]))) &&
        return get_gradient!(M, X, co.objective, p)
    return copyto!(
        M,
        X,
        p,
        get!(co.cache[:Gradient], p) do
            c, _ = get_cost_and_gradient!(M, X, co.objective, p)
            #if this is evaluated, we can also set c
            haskey(co.cache, :Cost) && setindex!(co.cache[:Cost], c, p)
            X
        end,
    )
end

#
# Hessian and precon - ToDo

end