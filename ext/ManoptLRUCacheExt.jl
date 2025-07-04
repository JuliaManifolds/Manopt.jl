module ManoptLRUCacheExt

using Manopt
import Manopt: init_caches
using ManifoldsBase
using LRUCache

# introduce LRU even as default.
function Manopt.init_caches(
    M::AbstractManifold, caches::AbstractVector{<:Symbol}; kwargs...
)
    return Manopt.init_caches(M, caches, LRU; kwargs...)
end

"""
    init_caches(caches, T::Type{LRU}; kwargs...)

Given a vector of symbols `caches`, this function sets up the
`NamedTuple` of caches, where `T` is the type of cache to use.

# Keyword arguments

* `p=`$(Manopt._link(:rand)): a point on a manifold, to both infer its type for keys and initialize caches
* `value=0.0`:
   a value both typing and initialising number-caches, the default is for (Float) values like the cost.
* `X=zero_vector(M, p)`:
  a tangent vector at `p` to both type and initialize tangent vector caches
* `cache_size=10`:
  a default cache size to use
* `cache_sizes=Dict{Symbol,Int}()`:
  a dictionary of sizes for the `caches` to specify different (non-default) sizes
"""
function Manopt.init_caches(
    M::AbstractManifold,
    caches::AbstractVector{<:Symbol},
    ::Type{LRU};
    p::P=rand(M),
    value::R=0.0,
    X::T=zero_vector(M, p),
    cache_size::Int=10,
    cache_sizes::Dict{Symbol,Int}=Dict{Symbol,Int}(),
) where {P,R<:Real,T}
    lru_caches = LRU[]
    for c in caches
        i = length(lru_caches)
        m = get(cache_sizes, c, cache_size)
        # Float cache, like for f
        (c === :Cost) && push!(lru_caches, LRU{P,R}(; maxsize=m))
        (c === :Differential) && push!(lru_caches, LRU{Tuple{P,T},R}(; maxsize=m))
        # vectors, like for Constraints/EqCOnstraints/InEqCOnstraints
        # (a) store whole vectors
        (c === :EqualityConstraints) && push!(lru_caches, LRU{P,Vector{R}}(; maxsize=m))
        (c === :InequalityConstraints) && push!(lru_caches, LRU{P,Vector{R}}(; maxsize=m))
        (c === :Constraints) && push!(lru_caches, LRU{P,Vector{Vector{R}}}(; maxsize=m))
        # (b) store single entries, but with an point-index key
        (c === :EqualityConstraint) && push!(lru_caches, LRU{Tuple{P,Int},R}(; maxsize=m))
        (c === :InequalityConstraint) && push!(lru_caches, LRU{Tuple{P,Int},R}(; maxsize=m))
        # Tangent Vector cache
        # (a) the simple ones, like the gradient or the Hessian
        (c === :Gradient) && push!(lru_caches, LRU{P,T}(; maxsize=m))
        (c === :SubGradient) && push!(lru_caches, LRU{P,T}(; maxsize=m))
        (c === :SubtrahendGradient) && push!(lru_caches, LRU{P,T}(; maxsize=m))
        # (b) indexed by point and vector
        (c === :Hessian) && push!(lru_caches, LRU{Tuple{P,T},T}(; maxsize=m))
        (c === :Preconditioner) && push!(lru_caches, LRU{Tuple{P,T},T}(; maxsize=m))
        # (b) store tangent vectors of components, but with an point-index key
        (c === :GradEqualityConstraint) &&
            push!(lru_caches, LRU{Tuple{P,Int},T}(; maxsize=m))
        (c === :GradInequalityConstraint) &&
            push!(lru_caches, LRU{Tuple{P,Int},T}(; maxsize=m))
        # For the (future) product tangent bundle this might also be just Ts
        (c === :GradEqualityConstraints) &&
            push!(lru_caches, LRU{P,Union{T,Vector{T}}}(; maxsize=m))
        (c === :GradInequalityConstraints) &&
            push!(lru_caches, LRU{P,Union{T,Vector{T}}}(; maxsize=m))
        # (c === :StochasticGradient)
        (c === :StochasticGradient) && push!(lru_caches, LRU{Tuple{P,Int},T}(; maxsize=m))
        (c === :StochasticGradients) && push!(lru_caches, LRU{P,Vector{T}}(; maxsize=m))
        # Point caches
        # (b) proximal point - again use (p, λ, i) as key
        (c === :ProximalMap) && push!(lru_caches, LRU{Tuple{P,R,Int},P}(; maxsize=m))
        # None of the previous cases matched -> unknown cache type
        if length(lru_caches) == i #nothing pushed
            error("""
            A cache for :$c seems to not be supported by LRU caches.
            """)
        end
    end
    return NamedTuple{Tuple(caches)}(lru_caches)
end

end
