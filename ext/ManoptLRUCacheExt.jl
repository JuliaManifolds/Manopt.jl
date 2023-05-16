module ManoptLRUCacheExt

if isdefined(Base, :get_extension)
    using Manopt
    import Manopt: init_caches
    using ManifoldsBase
    using LRUCache
else
    # imports need to be relative for Requires.jl-based workflows:
    # https://github.com/JuliaArrays/ArrayInterface.jl/pull/387
    using ..Manopt
    import ..Manopt: init_caches
    using ..ManifoldsBase
    using ..LRUCache
end

# introduce LRU even as default.
function Manopt.init_caches(M::AbstractManifold, caches::AbstractVector{<:Symbol}, kwargs...)
    return Manopt.init_caches(M, caches, LRUCaches.LRU; kwargs...)
end

"""
    init_caches(caches, T::Type{LRU}; kwargs...)

Given a vector of symbols `caches`, this function sets up the
`NamedTuple` of caches, where `T` is the type of cache to use.

# Keyword arguments

* `p`   - (`rand(M)`) a point on a manifold, to both infere its type for keys and initialize caches
* `v`   - (`0.0`) a value both typing and initialising number-caches, eg. for caching a cost.
* `X`   - (`zero_vector(M, p)` a tangent vector at `p` to both type and initialize tangent vector caches
* `cache_size` - (`10`)  a default cache size to use
* `cache_sizes` – (`Dict{Symbol,Int}()`) a dictionary of sizes for the `caches` to specify different (non-default) sizes
"""
function Manopt.init_caches(
    M,
    caches::AbstractVector{<:Symbol},
    ::Type{LRU};
    p::P=rand(M),
    v::R=0.0,
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

end
