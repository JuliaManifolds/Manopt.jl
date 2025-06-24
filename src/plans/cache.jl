#
# A Simple Cache for Objectives
#
@doc """
     SimpleManifoldCachedObjective{O<:AbstractManifoldFirstOrderObjective{E}, P, T,C} <: AbstractDecoratedManifoldObjective{E,O}

Provide a simple cache for an [`AbstractManifoldFirstOrderObjective`](@ref) that is, this cache
stores a point `p` and a gradient ``$(_tex(:grad)) f(p)`` in `X` as well as a cost value ``f(p)`` in `c`.
It can also easily evaluate the differential based on the cached gradient.

Both `X` and `c` are accompanied by booleans to keep track of their validity.

While this does not provide a cache for the differential, it uses the cached gradient
as a help to evaluate the differential, if an up-to-date gradient is available.
It otherwise does call the original differential.

This simple cache does not take into account, that some first order objectives have a
common function for cost & grad. It only caches the function that is actually called.

# Constructor

    SimpleManifoldCachedObjective(M::AbstractManifold, obj::AbstractManifoldFirstOrderObjective; kwargs...)

## Keyword arguments

* `p=`$(Manopt._link(:rand)): a point on the manifold to initialize the cache with
* `X=get_gradient(M, obj, p)` or `zero_vector(M,p)`: a tangent vector to store the gradient in,
  see also `initialize=`
* `c=[`get_cost`](@ref)`(M, obj, p)` or `0.0`: a value to store the cost function in `initialize`
* `initialized=true`: whether to initialize the cached `X` and `c` or not.
"""
mutable struct SimpleManifoldCachedObjective{
    E<:AbstractEvaluationType,O<:AbstractManifoldObjective{E},P,T,C
} <: AbstractDecoratedManifoldObjective{E,O}
    objective::O
    p::P # a point
    X::T # a vector
    X_valid::Bool
    c::C # a value for the cost
    c_valid::Bool
end

function SimpleManifoldCachedObjective(
    M::AbstractManifold,
    obj::O;
    initialized=true,
    p=rand(M),
    X=initialized ? get_gradient(M, obj, p) : zero_vector(M, p),
    c=initialized ? get_cost(M, obj, p) : 0.0,
) where {E<:AbstractEvaluationType,O<:AbstractManifoldObjective{E}}
    q = copy(M, p)
    return SimpleManifoldCachedObjective{E,O,typeof(q),typeof(X),typeof(c)}(
        obj, q, X, initialized, c, initialized
    )
end

# Default implementations
function get_cost(M::AbstractManifold, sco::SimpleManifoldCachedObjective, p)
    scop_neq_p = sco.p != p
    if scop_neq_p || !sco.c_valid
        # else evaluate cost, invalidate grad if p changed
        sco.c = get_cost(M, sco.objective, p)
        # for switched points, invalidate X
        scop_neq_p && (sco.X_valid = false)
        copyto!(M, sco.p, p)
        sco.c_valid = true
    end
    return sco.c
end

function get_cost_and_gradient(M::AbstractManifold, sco::SimpleManifoldCachedObjective, p)
    scop_neq_p = sco.p != p
    if scop_neq_p || !sco.X_valid || !sco.c_valid
        sco.c, sco.X = get_cost_and_gradient(M, sco.objective, p)
        sco.c_valid = true
        sco.X_valid = true
    end
    return (sco.c, copy(M, p, sco.X))
end
function get_cost_and_gradient!(
    M::AbstractManifold, X, sco::SimpleManifoldCachedObjective, p
)
    scop_neq_p = sco.p != p
    if scop_neq_p || !sco.X_valid || !sco.c_valid
        sco.c, _ = get_cost_and_gradient!(M, X, sco.objective, p)
        copyto!(M, sco.X, p, X)
        sco.c_valid = true
        sco.X_valid = true
    end
    return (sco.c, X)
end

function get_cost_function(sco::SimpleManifoldCachedObjective, recursive=false)
    recursive && return get_cost_function(sco.objective, recursive)
    return (M, p) -> get_cost(M, sco, p)
end

function get_differential(M::AbstractManifold, sco::SimpleManifoldCachedObjective, p, X)
    scop_neq_p = sco.p != p
    # Gradient outdated -> just call differenital of the inner objective
    if scop_neq_p || !sco.X_valid
        return get_differential(M, sco.objective, p, X)
    end
    # otherwise use the up to date gradient and inner
    return real(inner(M, p, sco.X, X))
end
function get_differential_function(
    sco::SimpleManifoldCachedObjective{AllocatingEvaluation}, recursive=false
)
    recursive && (return get_differential_function(sco.objective, recursive))
    return (M, p, X) -> get_differential(M, sco, p, X)
end

function get_gradient(M::AbstractManifold, sco::SimpleManifoldCachedObjective, p)
    scop_neq_p = sco.p != p
    if scop_neq_p || !sco.X_valid
        get_gradient!(M, sco.X, sco.objective, p)
        # for switched points, invalidate c
        scop_neq_p && (sco.c_valid = false)
        copyto!(M, sco.p, p)
        sco.X_valid = true
    end
    return copy(M, p, sco.X)
end
function get_gradient!(M::AbstractManifold, X, sco::SimpleManifoldCachedObjective, p)
    scop_neq_p = sco.p != p
    if scop_neq_p || !sco.X_valid
        get_gradient!(M, sco.X, sco.objective, p)
        scop_neq_p && (sco.c_valid = false)
        copyto!(M, sco.p, p)
        # for switched points, invalidate c
        sco.X_valid = true
    end
    copyto!(M, X, sco.p, sco.X)
    return X
end

function get_gradient_function(
    sco::SimpleManifoldCachedObjective{AllocatingEvaluation}, recursive=false
)
    recursive && (return get_gradient_function(sco.objective, recursive))
    return (M, p) -> get_gradient(M, sco, p)
end
function get_gradient_function(
    sco::SimpleManifoldCachedObjective{InplaceEvaluation}, recursive=false
)
    recursive && (return get_gradient_function(sco.objective, recursive))
    return (M, X, p) -> get_gradient!(M, X, sco, p)
end

#
# ManifoldCachedObjective constructor which errors by default
# since LRUCache.jl extension is required
#
@doc raw"""
    ManifoldCachedObjective{E,P,O<:AbstractManifoldObjective{<:E},C<:NamedTuple{}} <: AbstractDecoratedManifoldObjective{E,P}

Create a cache for an objective, based on a `NamedTuple` that stores some kind of cache.

# Constructor

    ManifoldCachedObjective(M, o::AbstractManifoldObjective, caches::Vector{Symbol}; kwargs...)

Create a cache for the [`AbstractManifoldObjective`](@ref) where the Symbols in `caches` indicate,
which function evaluations to cache.

# Supported symbols

| Symbol                      | Caches calls to (incl. `!` variants)            | Comment
| :-------------------------- | :---------------------------------------------- | :------------------------ |
| `:Cost`                     | [`get_cost`](@ref)                              |                           |
| `:Differential`             | [`get_differential`](@ref)`(M, p, X)`.          |                           |
| `:EqualityConstraint`       | [`get_equality_constraint`](@ref)`(M, p, i)`    |                           |
| `:EqualityConstraints`      | [`get_equality_constraint`](@ref)`(M, p, :)`    |                           |
| `:GradEqualityConstraint`   | [`get_grad_equality_constraint`](@ref)          | tangent vector per (p,i)  |
| `:GradInequalityConstraint` | [`get_inequality_constraint`](@ref)             | tangent vector per (p,i)  |
| `:Gradient`                 | [`get_gradient`](@ref)`(M,p)`                   | tangent vectors           |
| `:Hessian`                  | [`get_hessian`](@ref)                           | tangent vectors           |
| `:InequalityConstraint`     | [`get_inequality_constraint`](@ref)`(M, p, j)`  |                           |
| `:InequalityConstraints`    | [`get_inequality_constraint`](@ref)`(M, p, :)`  |                           |
| `:Preconditioner`           | [`get_preconditioner`](@ref)                    | tangent vectors           |
| `:ProximalMap`              | [`get_proximal_map`](@ref)                      | point per `(p,λ,i)`       |
| `:StochasticGradients`      | [`get_gradients`](@ref)                         | vector of tangent vectors |
| `:StochasticGradient`       | [`get_gradient`](@ref)`(M, p, i)`               | tangent vector per (p,i)  |
| `:SubGradient`              | [`get_subgradient`](@ref)                       | tangent vectors           |
| `:SubtrahendGradient`       | [`get_subtrahend_gradient`](@ref)               | tangent vectors           |

# Keyword arguments

* `p=rand(M)`:
  the type of the keys to be used in the caches. Defaults to the default representation on `M`.
* `value=get_cost(M, objective, p)`:
  the type of values for numeric values in the cache
* `X=zero_vector(M,p)`:
  the type of values to be cached for gradient and Hessian calls.
* `cache=[:Cost]`:
  a vector of symbols indicating which function calls should be cached.
* `cache_size=10`:
  number of (least recently used) calls to cache
* `cache_sizes=Dict{Symbol,Int}()`:
  a named tuple or dictionary specifying the sizes individually for each cache.
"""
struct ManifoldCachedObjective{E,P,O<:AbstractManifoldObjective{<:E},C<:NamedTuple{}} <:
       AbstractDecoratedManifoldObjective{E,P}
    objective::O
    cache::C
    cache_all::Bool
end
function ManifoldCachedObjective(
    M::AbstractManifold,
    objective::O,
    caches::AbstractVector{<:Symbol}=[:Cost];
    p::P=rand(M),
    value::R=get_cost(M, objective, p),
    X::T=zero_vector(M, p),
    cache_size::Int=10,
    cache_sizes::Dict{Symbol,Int}=Dict{Symbol,Int}(),
    cache_all=true,
) where {E,O<:AbstractManifoldObjective{E},R<:Real,P,T}
    c = init_caches(
        M, caches; p=p, value=value, X=X, cache_size=cache_size, cache_sizes=cache_sizes
    )
    return ManifoldCachedObjective{E,O,O,typeof(c)}(objective, c, cache_all)
end
function ManifoldCachedObjective(
    M::AbstractManifold,
    objective::O,
    caches::AbstractVector{<:Symbol}=[:Cost];
    p::P=rand(M),
    value::R=get_cost(M, objective, p),
    X::T=zero_vector(M, p),
    cache_size::Int=10,
    cache_sizes::Dict{Symbol,Int}=Dict{Symbol,Int}(),
    cache_all=true,
) where {E,O2,O<:AbstractDecoratedManifoldObjective{E,O2},R<:Real,P,T}
    c = init_caches(
        M, caches; p=p, value=value, X=X, cache_size=cache_size, cache_sizes=cache_sizes
    )
    return ManifoldCachedObjective{E,O2,O,typeof(c)}(objective, c, cache_all)
end

"""
    init_caches(M::AbstractManifold, caches, T; kwargs...)

Given a vector of symbols `caches`, this function sets up the
`NamedTuple` of caches for points/vectors on `M`,
where `T` is the type of cache to use.
"""
function init_caches(M::AbstractManifold, caches, T=Nothing; kwargs...)
    return throw(DomainError(
        T,
        """
        No function `init_caches` available for a caches $T.
        For a good default load `LRUCache.jl`, for example:
        Enter `using LRUCache`.
        """,
    ))
end

#
# Default implementations: if field exists -> try to get cache
function get_cost(M::AbstractManifold, co::ManifoldCachedObjective, p)
    !(haskey(co.cache, :Cost)) && return get_cost(M, co.objective, p) #No Cost cache
    # If so, check whether we should cache
    return get!(co.cache[:Cost], copy(M, p)) do
        get_cost(M, co.objective, p)
    end
end

function get_cost_function(co::ManifoldCachedObjective, recursive=false)
    recursive && (return get_cost_function(co.objective, recursive))
    return (M, p) -> get_cost(M, co, p)
end

function get_differrential(M::AbstractManifold, co::ManifoldCachedObjective, p, X)
    # No Differential Cache
    !(haskey(co.cache, :Differential)) && return get_differential(M, co.objective, p, X)
    # If so, check whether we should cache
    return get!(co.cache[:Differential], (copy(M, p), copy(M, p, X))) do
        get_differential(M, co.objective, p, X)
    end
end

function get_differential_function(mco::ManifoldCachedObjective, recursive=false)
    recursive && (return get_differential_function(mco.objective, recursive))
    return (M, p, X) -> get_differential(M, mco, p, X)
end

function get_gradient(M::AbstractManifold, co::ManifoldCachedObjective, p)
    !(haskey(co.cache, :Gradient)) && return get_gradient(M, co.objective, p)
    return copy(
        M,
        p,
        get!(co.cache[:Gradient], copy(M, p)) do
            get_gradient(M, co.objective, p)
        end,
    )
end
function get_gradient!(M::AbstractManifold, X, co::ManifoldCachedObjective, p)
    !(haskey(co.cache, :Gradient)) && return get_gradient!(M, X, co.objective, p)
    copyto!(
        M,
        X,
        p,
        get!(co.cache[:Gradient], copy(M, p)) do
            # This evaluates in place of X
            get_gradient!(M, X, co.objective, p)
            copy(M, p, X) #this creates a copy to be placed in the cache
        end, #and copy the values back to X
    )
    return X
end

function get_gradient_function(
    sco::ManifoldCachedObjective{AllocatingEvaluation}, recursive=false
)
    recursive && (return get_gradient_function(sco.objective, recursive))
    return (M, p) -> get_gradient(M, sco, p)
end
function get_gradient_function(
    sco::ManifoldCachedObjective{InplaceEvaluation}, recursive=false
)
    recursive && (return get_gradient_function(sco.objective, recursive))
    return (M, X, p) -> get_gradient!(M, X, sco, p)
end

function get_cost_and_gradient(M::AbstractManifold, mco::ManifoldCachedObjective, p)
    #Neither cost not grad cached -> evaluate normally
    all(.!(haskey.(Ref(mco.cache), [:Cost, :Gradient]))) &&
        return get_cost_and_gradient(M, mco.objective, p)
    # Otherwise -> check whether any of the does not have this index:
    # No cost case or no grad case
    nc = !haskey(mco.cache, :Cost) || !haskey(mco.cache[:Cost], p)
    ng = !haskey(mco.cache, :Gradient) || !haskey(mco.cache[:Gradient], p)
    if nc || ng # one of them does not exist, either full cache or entry -> eval
        c, X = get_cost_and_gradient(M, mco.objective, p)
        # Cache if cache present
        haskey(mco.cache, :Cost) && setindex!(mco.cache[:Cost], c, copy(M, p))
        haskey(mco.cache, :Gradient) &&
            setindex!(mco.cache[:Gradient], copy(M, p, X), copy(M, p))
        return c, X
    else # both exist and are cached, return them
        return get(mco.cache[:Cost], p), copy(M, p, get(mco.cache[:Gradient], p))
    end
end
function get_cost_and_gradient!(M::AbstractManifold, X, mco::ManifoldCachedObjective, p)
    #Neither cost not grad cached -> evaluate normally
    all(.!(haskey.(Ref(mco.cache), [:Cost, :Gradient]))) &&
        return get_cost_and_gradient!(M, X, mco.objective, p)
    # Otherwise -> check whether any of the does not have this index:
    # No cost case or no grad case
    nc = !haskey(mco.cache, :Cost) || !haskey(mco.cache[:Cost], p)
    ng = !haskey(mco.cache, :Gradient) || !haskey(mco.cache[:Gradient], p)
    if nc || ng # one of them does not exist, either full cache or entry -> eval
        c, _ = get_cost_and_gradient!(M, X, mco.objective, p)
        # Cache if cache present
        haskey(mco.cache, :Cost) && setindex!(mco.cache[:Cost], c, copy(M, p))
        haskey(mco.cache, :Gradient) &&
            setindex!(mco.cache[:Gradient], copy(M, p, X), copy(M, p))
        return c, X
    else # both exist and are cached, return them
        copyto!(M, X, p, get(mco.cache[:Gradient], p))
        return get(mco.cache[:Cost], p), X
    end
end

#
# Constraints
function get_equality_constraint(
    M::AbstractManifold, co::ManifoldCachedObjective, p, j::Integer
)
    (!haskey(co.cache, :EqualityConstraint)) &&
        return get_equality_constraint(M, co.objective, p, j)
    return copy(# Return a copy of the version in the cache
        get!(co.cache[:EqualityConstraint], (copy(M, p), j)) do
            get_equality_constraint(M, co.objective, p, j)
        end,
    )
end
function get_equality_constraint(
    M::AbstractManifold, co::ManifoldCachedObjective, p, i::Colon
)
    (!haskey(co.cache, :EqualityConstraints)) &&
        return get_equality_constraint(M, co.objective, p, i)
    return copy(# Return a copy of the version in the cache
        get!(co.cache[:EqualityConstraints], copy(M, p)) do
            get_equality_constraint(M, co.objective, p, i)
        end,
    )
end
function get_equality_constraint(M::AbstractManifold, co::ManifoldCachedObjective, p, i)
    key = copy(M, p)
    if haskey(co.cache, :EqualityConstraints) # full constraints are stored
        if haskey(co.cache[:EqualityConstraints], key)
            return co.cache[:EqualityConstraints][key][i]
            #but caching is not possible here, since that requires evaluating all
        end
    end
    if haskey(co.cache, :EqualityConstraint) # storing the index constraints
        return [
            copy(
                get!(co.cache[:EqualityConstraint], (key, j)) do
                    get_equality_constraint(M, co.objective, p, j)
                end,
            ) for j in _to_iterable_indices(1:equality_constraints_length(co.objective), i)
        ]
    end # neither cache: pass down to objective
    return get_equality_constraint(M, co.objective, p, i)
end
function get_inequality_constraint(
    M::AbstractManifold, co::ManifoldCachedObjective, p, i::Integer
)
    (!haskey(co.cache, :InequalityConstraint)) &&
        return get_inequality_constraint(M, co.objective, p, i)
    return copy(# Return a copy of the version in the cache
        get!(co.cache[:InequalityConstraint], (copy(M, p), i)) do
            get_inequality_constraint(M, co.objective, p, i)
        end,
    )
end
function get_inequality_constraint(
    M::AbstractManifold, co::ManifoldCachedObjective, p, i::Colon
)
    (!haskey(co.cache, :InequalityConstraints)) &&
        return get_inequality_constraint(M, co.objective, p, i)
    return copy(# Return a copy of the version in the cache
        get!(co.cache[:InequalityConstraints], copy(M, p)) do
            get_inequality_constraint(M, co.objective, p, i)
        end,
    )
end
function get_inequality_constraint(M::AbstractManifold, co::ManifoldCachedObjective, p, i)
    key = copy(M, p)
    if haskey(co.cache, :InequalityConstraints) # full constraints are stored
        if haskey(co.cache[:InequalityConstraints], key)
            return co.cache[:InequalityConstraints][key][i]
            #but caching is not possible here, since that requires evaluating all
        end
    end
    if haskey(co.cache, :InequalityConstraint) # storing the index constraints
        return [
            copy(
                get!(co.cache[:InequalityConstraint], (key, j)) do
                    get_inequality_constraint(M, co.objective, p, j)
                end,
            ) for
            j in _to_iterable_indices(1:inequality_constraints_length(co.objective), i)
        ]
    end # neither cache: pass down to objective
    return get_inequality_constraint(M, co.objective, p, i)
end

#
#
# Gradients of Equality Constraints
function get_grad_equality_constraint(
    M::AbstractManifold,
    co::ManifoldCachedObjective,
    p,
    j::Integer,
    range::Union{AbstractPowerRepresentation,Nothing}=nothing,
)
    !(haskey(co.cache, :GradEqualityConstraint)) &&
        return get_grad_equality_constraint(M, co.objective, p, j)
    return copy(# Return a copy of the version in the cache
        M,
        p,
        get!(co.cache[:GradEqualityConstraint], (copy(M, p), j)) do
            get_grad_equality_constraint(M, co.objective, p, j)
        end,
    )
end
function get_grad_equality_constraint(
    M::AbstractManifold,
    co::ManifoldCachedObjective{E,<:ConstrainedManifoldObjective},
    p,
    j::Colon,
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
) where {E}
    !(haskey(co.cache, :GradEqualityConstraints)) &&
        return get_grad_equality_constraint(M, co.objective, p, j)
    pM = PowerManifold(M, range, length(get_objective(co, true).equality_constraints))
    P = fill(p, pM)
    return copy(# Return a copy of the version in the cache
        pM,
        P,
        get!(co.cache[:GradEqualityConstraints], (copy(M, p))) do
            get_grad_equality_constraint(M, co.objective, p, j)
        end,
    )
end
function get_grad_equality_constraint(
    M::AbstractManifold,
    co::ManifoldCachedObjective,
    p,
    i,
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
)
    key = copy(M, p)
    n = _vgf_index_to_length(i, equality_constraints_length(co.objective))
    pM = PowerManifold(M, range, n)
    rep_size = representation_size(M)
    P = fill(p, pM)
    if haskey(co.cache, :GradEqualityConstraints) # full constraints are stored
        if haskey(co.cache[:GradEqualityConstraints], key)
            return co.cache[:GradEqualityConstraints][key][i]
            #but caching is not possible here, since that requires evaluating all
        end
    end
    if haskey(co.cache, :GradEqualityConstraint) # storing the index constraints
        # allocate a tangent vector
        X = zero_vector(pM, P)
        # access is subsampled with j, result linear in k
        for (k, j) in
            zip(1:n, _to_iterable_indices(1:equality_constraints_length(co.objective), i))
            copyto!(
                M,
                _write(pM, rep_size, X, (k,)),
                p,
                get!(co.cache[:GradEqualityConstraint], (key, j)) do
                    get_grad_equality_constraint(M, co.objective, p, j)
                end,
            )
        end
        return X
    end # neither cache: pass down to objective
    return get_grad_equality_constraint(M, co.objective, p, i)
end
function get_grad_equality_constraint!(
    M::AbstractManifold,
    X,
    co::ManifoldCachedObjective,
    p,
    j::Integer,
    range::Union{AbstractPowerRepresentation,Nothing}=nothing,
)
    !(haskey(co.cache, :GradEqualityConstraint)) &&
        return get_grad_equality_constraint!(M, X, co.objective, p, j)
    copyto!(
        M,
        X,
        p,
        get!(co.cache[:GradEqualityConstraint], (copy(M, p), j)) do
            # This evaluates in place of X
            get_grad_equality_constraint!(M, X, co.objective, p, j)
            copy(M, p, X) #this creates a copy to be placed in the cache
        end, #and copy the values back to X
    )
    return X
end
function get_grad_equality_constraint!(
    M::AbstractManifold,
    X,
    co::ManifoldCachedObjective{E,<:ConstrainedManifoldObjective},
    p,
    i::Colon,
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
) where {E}
    !(haskey(co.cache, :GradEqualityConstraints)) &&
        return get_grad_equality_constraint!(M, X, co.objective, p, i)
    pM = PowerManifold(M, range, length(get_objective(co, true).equality_constraints))
    P = fill(p, pM)
    copyto!(
        pM,
        X,
        P,
        get!(co.cache[:GradEqualityConstraints], (copy(M, p))) do
            # This evaluates in place of X
            get_grad_equality_constraint!(M, X, co.objective, p, i)
            copy(pM, P, X) #this creates a copy to be placed in the cache
        end, #and copy the values back to X
    )
    return X
end
function get_grad_equality_constraint!(
    M::AbstractManifold,
    X,
    co::ManifoldCachedObjective,
    p,
    i,
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
)
    key = copy(M, p)
    n = _vgf_index_to_length(i, equality_constraints_length(co.objective))
    pM = PowerManifold(M, range, n)
    rep_size = representation_size(M)
    if haskey(co.cache, :GradEqualityConstraints) # full constraints are stored
        if haskey(co.cache[:GradEqualityConstraints], key)
            # access is subsampled with j, result linear in k
            for (k, j) in zip(
                1:n, _to_iterable_indices(1:equality_constraints_length(co.objective), i)
            )
                copyto!(
                    M,
                    _write(pM, rep_size, X, (k,)),
                    p,
                    co.cache[:GradEqualityConstraints][key][j],
                )
            end
            return X
            #but caching is not possible here, since that requires evaluating all
        end
    end
    if haskey(co.cache, :GradEqualityConstraint) # store the index constraints
        # allocate a tangent vector
        # access is subsampled with j, result linear in k
        for (k, j) in
            zip(1:n, _to_iterable_indices(1:equality_constraints_length(co.objective), i))
            copyto!(
                M,
                _write(pM, rep_size, X, (k,)),
                p,
                get!(co.cache[:GradEqualityConstraint], (key, j)) do
                    get_grad_equality_constraint(M, co.objective, p, j)
                end,
            )
        end
        return X
    end # neither cache: pass down to objective
    return get_grad_equality_constraint!(M, X, co.objective, p, i)
end

#
#
# Inequality Constraint
function get_grad_inequality_constraint(
    M::AbstractManifold,
    co::ManifoldCachedObjective,
    p,
    i::Integer,
    range::Union{AbstractPowerRepresentation,Nothing}=nothing,
)
    !(haskey(co.cache, :GradInequalityConstraint)) &&
        return get_grad_inequality_constraint(M, co.objective, p, i)
    return copy(
        M,
        p,
        get!(co.cache[:GradInequalityConstraint], (copy(M, p), i)) do
            get_grad_inequality_constraint(M, co.objective, p, i)
        end,
    )
end
function get_grad_inequality_constraint(
    M::AbstractManifold,
    co::ManifoldCachedObjective{E,<:ConstrainedManifoldObjective},
    p,
    i::Colon,
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
) where {E}
    !(haskey(co.cache, :GradInequalityConstraints)) &&
        return get_grad_inequality_constraint(M, co.objective, p, i)
    pM = PowerManifold(M, range, length(get_objective(co, true).inequality_constraints))
    P = fill(p, pM)
    return copy(# Return a copy of the version in the cache
        pM,
        P,
        get!(co.cache[:GradInequalityConstraints], (copy(M, p))) do
            get_grad_inequality_constraint(M, co.objective, p, i)
        end,
    )
end
function get_grad_inequality_constraint(
    M::AbstractManifold,
    co::ManifoldCachedObjective,
    p,
    i,
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
)
    key = copy(M, p)
    n = _vgf_index_to_length(i, inequality_constraints_length(co.objective))
    pM = PowerManifold(M, range, n)
    rep_size = representation_size(M)
    P = fill(p, pM)
    if haskey(co.cache, :GradInequalityConstraints) # full constraints are stored
        if haskey(co.cache[:GradInequalityConstraints], key)
            return co.cache[:GradInequalityConstraints][key][i]
            #but caching is not possible here, since that requires evaluating all
        end
    end
    if haskey(co.cache, :GradInequalityConstraint) # storing the index constraints
        # allocate a tangent vector
        X = zero_vector(pM, P)
        # access is subsampled with j, result linear in k
        for (k, j) in
            zip(1:n, _to_iterable_indices(1:equality_constraints_length(co.objective), i))
            copyto!(
                M,
                _write(pM, rep_size, X, (k,)),
                p,
                get!(co.cache[:GradInequalityConstraint], (key, j)) do
                    get_grad_inequality_constraint(M, co.objective, p, j)
                end,
            )
        end
        return X
    end # neither cache: pass down to objective
    return get_grad_inequality_constraint(M, co.objective, p, i)
end
function get_grad_inequality_constraint!(
    M::AbstractManifold,
    X,
    co::ManifoldCachedObjective,
    p,
    i::Integer,
    range::Union{AbstractPowerRepresentation,Nothing}=nothing,
)
    !(haskey(co.cache, :GradInequalityConstraint)) &&
        return get_grad_inequality_constraint!(M, X, co.objective, p, i)
    copyto!(
        M,
        X,
        p,
        get!(co.cache[:GradInequalityConstraint], (copy(M, p), i)) do
            # This evaluates in place of X
            get_grad_inequality_constraint!(M, X, co.objective, p, i)
            copy(M, p, X) #this creates a copy to be placed in the cache
        end, #and copy the values back to X
    )
    return X
end
function get_grad_inequality_constraint!(
    M::AbstractManifold,
    X,
    co::ManifoldCachedObjective{E,<:ConstrainedManifoldObjective},
    p,
    j::Colon,
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
) where {E}
    !(haskey(co.cache, :GradInequalityConstraints)) &&
        return get_grad_inequality_constraint!(M, X, co.objective, p, j)
    pM = PowerManifold(M, range, length(get_objective(co, true).inequality_constraints))
    P = fill(p, pM)
    copyto!(
        pM,
        X,
        P,
        get!(co.cache[:GradInequalityConstraints], (copy(M, p))) do
            # This evaluates in place of X
            get_grad_inequality_constraint!(M, X, co.objective, p, j)
            copy(pM, P, X) #this creates a copy to be placed in the cache
        end, #and copy the values back to X
    )
    return X
end
function get_grad_inequality_constraint!(
    M::AbstractManifold,
    X,
    co::ManifoldCachedObjective,
    p,
    i,
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
)
    key = copy(M, p)
    n = _vgf_index_to_length(i, inequality_constraints_length(co.objective))
    pM = PowerManifold(M, range, n)
    rep_size = representation_size(M)
    if haskey(co.cache, :GradInequalityConstraints) # full constraints are stored
        if haskey(co.cache[:GradInequalityConstraints], key)
            # access is subsampled with j, result linear in k
            for (k, j) in zip(
                1:n, _to_iterable_indices(1:equality_constraints_length(co.objective), i)
            )
                copyto!(
                    M,
                    _write(pM, rep_size, X, (k,)),
                    p,
                    co.cache[:GradInequalityConstraints][key][j],
                )
            end
            return X
            #but caching is not possible here, since that requires evaluating all
        end
    end
    if haskey(co.cache, :GradInequalityConstraint) # storing the index constraints
        # access is subsampled with j, result linear in k
        for (k, j) in
            zip(1:n, _to_iterable_indices(1:equality_constraints_length(co.objective), i))
            copyto!(
                M,
                _write(pM, rep_size, X, (k,)),
                p,
                get!(co.cache[:GradInequalityConstraint], (key, j)) do
                    get_grad_inequality_constraint(M, co.objective, p, j)
                end,
            )
        end
        return X
    end # neither cache: pass down to objective
    return get_grad_inequality_constraint!(M, X, co.objective, p, i)
end

#
# Hessian
function get_hessian(M::AbstractManifold, co::ManifoldCachedObjective, p, X)
    !(haskey(co.cache, :Hessian)) && return get_hessian(M, co.objective, p, X)
    return copy(
        M,
        p,
        get!(co.cache[:Hessian], (copy(M, p), copy(M, p, X))) do
            get_hessian(M, co.objective, p, X)
        end,
    )
end
function get_hessian!(M::AbstractManifold, Y, co::ManifoldCachedObjective, p, X)
    !(haskey(co.cache, :Hessian)) && return get_hessian!(M, Y, co.objective, p, X)
    copyto!(
        M,
        Y,
        p, # perform an in-place cache evaluation, see also `get_gradient!`
        get!(co.cache[:Hessian], (copy(M, p), copy(M, p, X))) do
            get_hessian!(M, Y, co.objective, p, X)
            copy(M, p, Y) #store a copy of Y
        end,
    )
    return Y
end

function get_hessian_function(
    emo::ManifoldCachedObjective{AllocatingEvaluation}, recursive::Bool=false
)
    recursive && (return get_hessian_function(emo.objective, recursive))
    return (M, p, X) -> get_hessian(M, emo, p, X)
end
function get_hessian_function(
    emo::ManifoldCachedObjective{InplaceEvaluation}, recursive::Bool=false
)
    recursive && (return get_hessian_function(emo.objective, recursive))
    return (M, Y, p, X) -> get_hessian!(M, Y, emo, p, X)
end
function get_preconditioner(M::AbstractManifold, co::ManifoldCachedObjective, p, X)
    !(haskey(co.cache, :Preconditioner)) && return get_preconditioner(M, co.objective, p, X)
    return copy(
        M,
        p,
        get!(co.cache[:Preconditioner], (copy(M, p), copy(M, p, X))) do
            get_preconditioner(M, co.objective, p, X)
        end,
    )
end
function get_preconditioner!(M::AbstractManifold, Y, co::ManifoldCachedObjective, p, X)
    !(haskey(co.cache, :Preconditioner)) &&
        return get_preconditioner!(M, Y, co.objective, p, X)
    copyto!(
        M,
        Y,
        p, # perform an in-place cache evaluation, see also `get_gradient!`
        get!(co.cache[:Preconditioner], (copy(M, p), copy(M, p, X))) do
            get_preconditioner!(M, Y, co.objective, p, X)
            copy(M, p, Y)
        end,
    )
    return Y
end

#
# Proximal Map
function get_proximal_map(M::AbstractManifold, co::ManifoldCachedObjective, λ, p, i)
    !(haskey(co.cache, :ProximalMap)) && return get_proximal_map(M, co.objective, λ, p, i)
    return copy(
        M,
        get!(co.cache[:ProximalMap], (copy(M, p), λ, i)) do  # use the tuple (p,i) as key
            get_proximal_map(M, co.objective, λ, p, i)
        end,
    )
end
function get_proximal_map!(M::AbstractManifold, q, co::ManifoldCachedObjective, λ, p, i)
    !(haskey(co.cache, :ProximalMap)) &&
        return get_proximal_map!(M, q, co.objective, λ, p, i)
    copyto!(
        M,
        q,
        get!(co.cache[:ProximalMap], (copy(M, p), λ, i)) do
            get_proximal_map!(M, q, co.objective, λ, p, i) #compute in-place of q
            copy(M, q) #store copy of q
        end,
    )
    return q
end
#
# Stochastic Gradient
function get_gradient(M::AbstractManifold, co::ManifoldCachedObjective, p, i)
    !(haskey(co.cache, :StochasticGradient)) && return get_gradient(M, co.objective, p, i)
    return copy(
        M,
        p,
        get!(co.cache[:StochasticGradient], (copy(M, p), i)) do
            get_gradient(M, co.objective, p, i)
        end,
    )
end
function get_gradient!(M::AbstractManifold, X, co::ManifoldCachedObjective, p, i)
    !(haskey(co.cache, :StochasticGradient)) &&
        return get_gradient!(M, X, co.objective, p, i)
    copyto!(
        M,
        X,
        p,
        get!(co.cache[:StochasticGradient], (copy(M, p), i)) do
            # This evaluates in place of X
            get_gradient!(M, X, co.objective, p, i)
            copy(M, p, X) #this creates a copy to be placed in the cache
        end, #and copy the values back to X
    )
    return X
end

function get_gradients(M::AbstractManifold, co::ManifoldCachedObjective, p)
    !(haskey(co.cache, :StochasticGradients)) && return get_gradients(M, co.objective, p)
    return copy.(
        Ref(M),
        Ref(p),
        get!(co.cache[:StochasticGradients], copy(M, p)) do
            get_gradients(M, co.objective, p)
        end,
    )
end
function get_gradients!(M::AbstractManifold, X, co::ManifoldCachedObjective, p)
    !(haskey(co.cache, :StochasticGradients)) && return get_gradients(M, X, co.objective, p)
    copyto!.(
        Ref(M),
        X,
        Ref(p),
        get!(co.cache[:StochasticGradients], copy(M, p)) do
            # This evaluates in place of X
            get_gradients!(M, X, co.objective, p)
            copy.(Ref(M), Ref(p), X) #this creates a copy to be placed in the cache
        end, #and copy the values back to X
    )
    return X
end
#
# Subgradient
function get_subgradient(M::AbstractManifold, co::ManifoldCachedObjective, p)
    !(haskey(co.cache, :SubGradient)) && return get_subgradient(M, co.objective, p)
    return copy(
        M,
        p,
        get!(co.cache[:SubGradient], copy(M, p)) do
            get_subgradient(M, co.objective, p)
        end,
    )
end
function get_subgradient!(M::AbstractManifold, X, co::ManifoldCachedObjective, p)
    !(haskey(co.cache, :SubGradient)) && return get_subgradient!(M, X, co.objective, p)
    copyto!(
        M,
        X,
        p, # perform an in-place cache evaluation, see also `get_gradient!`
        get!(co.cache[:SubGradient], copy(M, p)) do
            get_subgradient!(M, X, co.objective, p)
            copy(M, p, X)
        end,
    )
    return X
end

#
# Subtrahend gradient (from DC)
function get_subtrahend_gradient(M::AbstractManifold, co::ManifoldCachedObjective, p)
    !(haskey(co.cache, :SubtrahendGradient)) &&
        return get_subtrahend_gradient(M, co.objective, p)
    return copy(
        M,
        p,
        get!(co.cache[:SubtrahendGradient], copy(M, p)) do
            get_subtrahend_gradient(M, co.objective, p)
        end,
    )
end
function get_subtrahend_gradient!(M::AbstractManifold, X, co::ManifoldCachedObjective, p)
    !(haskey(co.cache, :SubtrahendGradient)) &&
        return get_subtrahend_gradient!(M, X, co.objective, p)
    copyto!(
        M,
        X,
        p, # perform an in-place cache evaluation, see also `get_gradient!`
        get!(co.cache[:SubtrahendGradient], copy(M, p)) do
            get_subtrahend_gradient!(M, X, co.objective, p)
            copy(M, p, X)
        end,
    )
    return X
end
#
# Factory
#
@doc raw"""
    objective_cache_factory(M::AbstractManifold, o::AbstractManifoldObjective, cache::Symbol)

Generate a cached variant of the [`AbstractManifoldObjective`](@ref) `o`
on the `AbstractManifold M` based on the symbol `cache`.

The following caches are available

* `:Simple` generates a [`SimpleManifoldCachedObjective`](@ref)
* `:LRU` generates a [`ManifoldCachedObjective`](@ref) where you should use the form
  `(:LRU, [:Cost, :Gradient])` to specify what should be cached or
  `(:LRU, [:Cost, :Gradient], 100)` to specify the cache size.
  Here this variant defaults to `(:LRU, [:Cost, :Gradient], 100)`,
  caching up to 100 cost and gradient values.[^1]

[^1]:
    This cache requires [`LRUCache.jl`](https://github.com/JuliaCollections/LRUCache.jl) to be loaded as well.
"""
function objective_cache_factory(M, o, cache::Symbol)
    (cache === :Simple) && return SimpleManifoldCachedObjective(M, o)
    (cache === :LRU) &&
        return ManifoldCachedObjective(M, o, [:Cost, :Gradient]; cache_size=100)
    return o
end

@doc raw"""
    objective_cache_factory(M::AbstractManifold, o::AbstractManifoldObjective, cache::Tuple{Symbol, Array, Array})
    objective_cache_factory(M::AbstractManifold, o::AbstractManifoldObjective, cache::Tuple{Symbol, Array})

Generate a cached variant of the [`AbstractManifoldObjective`](@ref) `o`
on the `AbstractManifold M` based on the symbol `cache[1]`,
where the second element `cache[2]` are further arguments to the cache and
the optional third is passed down as keyword arguments.

For all available caches see the simpler variant with symbols.
"""
function objective_cache_factory(M, o, cache::Tuple{Symbol,<:AbstractArray,I}) where {I}
    (cache[1] === :Simple) && return SimpleManifoldCachedObjective(M, o; cache[3]...)
    if (cache[1] === :LRU)
        if (cache[3] isa Integer)
            return ManifoldCachedObjective(M, o, cache[2]; cache_size=cache[3])
        else
            return ManifoldCachedObjective(M, o, cache[2]; cache[3]...)
        end
    end
    return o
end
function objective_cache_factory(M, o, cache::Tuple{Symbol,<:AbstractArray})
    (cache[1] === :Simple) && return SimpleManifoldCachedObjective(M, o)
    (cache[1] === :LRU) && return ManifoldCachedObjective(M, o, cache[2])
    return o
end
function show(io::IO, smco::SimpleManifoldCachedObjective{E}) where {E}
    return print(io, "SimpleManifoldCachedObjective{$E,$(smco.objective)}")
end
function show(
    io::IO, t::Tuple{<:SimpleManifoldCachedObjective,S}
) where {S<:AbstractManoptSolverState}
    return print(io, "$(t[2])\n\n$(status_summary(t[1]))")
end
function show(io::IO, mco::ManifoldCachedObjective)
    return print(io, "$(status_summary(mco))")
end
function show(
    io::IO, t::Tuple{<:ManifoldCachedObjective,S}
) where {S<:AbstractManoptSolverState}
    return print(io, "$(t[2])\n\n$(status_summary(t[1]))")
end

function status_summary(smco::SimpleManifoldCachedObjective)
    s = """
    ## Cache
    A `SimpleManifoldCachedObjective` to cache one point and one tangent vector for the iterate and gradient, respectively
    """
    s2 = status_summary(smco.objective)
    length(s2) > 0 && (s2 = "\n$(s2)")
    return "$(s)$(s2)"
end
function status_summary(mco::ManifoldCachedObjective)
    s = "## Cache\n"
    s2 = status_summary(mco.objective)
    (length(s2) > 0) && (s2 = "\n$(s2)")
    length(mco.cache) == 0 && return "$(s)    No caches active\n$(s2)"
    longest_key_length = max(length.(["$k" for k in keys(mco.cache)])...)
    cache_strings = [
        "  * :" *
        rpad("$k", longest_key_length, " ") *
        " : $(v.currentsize)/$(v.maxsize) entries of type $(valtype(v)) used" for
        (k, v) in zip(keys(mco.cache), values(mco.cache))
    ]
    return "$(s)$(join(cache_strings,"\n"))\n$s2"
end
