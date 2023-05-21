#
# A Simple Cache for Objectives
#
@doc raw"""
     SimpleManifoldCachedObjective{O<:AbstractManifoldGradientObjective{E,TC,TG}, P, T,C} <: AbstractManifoldGradientObjective{E,TC,TG}

Provide a simple cache for an [`AbstractManifoldGradientObjective`](@ref) that is for a given point `p` this cache
stores a point `p` and a gradient ``\operatorname{grad} f(p)`` in `X` as well as a cost value ``f(p)`` in `c`.

Both `X` and `c` are accompanied by booleans to keep track of their validity.

# Constructor

    SimpleManifoldCachedObjective(M::AbstractManifold, obj::AbstractManifoldGradientObjective; kwargs...)

## Keyword
* `p` (`rand(M)`) – a point on the manifold to initialize the cache with
* `X` (`get_gradient(M, obj, p)` or `zero_vector(M,p)`) – a tangent vector to store the gradient in, see also `initialize`
* `c` (`get_cost(M, obj, p)` or `0.0`) – a value to store the cost function in `initialize`
* `initialized` (`true`) – whether to initialize the cached `X` and `c` or not.
"""
mutable struct SimpleManifoldCachedObjective{
    E<:AbstractEvaluationType,TC,TG,O<:AbstractManifoldGradientObjective{E,TC,TG},P,T,C
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
) where {E<:AbstractEvaluationType,TC,TG,O<:AbstractManifoldGradientObjective{E,TC,TG}}
    q = copy(M, p)
    return SimpleManifoldCachedObjective{E,TC,TG,O,typeof(q),typeof(X),typeof(c)}(
        obj, q, X, initialized, c, initialized
    )
end
# Default implementations
function get_cost(M::AbstractManifold, sco::SimpleManifoldCachedObjective, p)
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
get_cost_function(sco::SimpleManifoldCachedObjective) = (M, p) -> get_cost(M, sco, p)
function get_gradient_function(sco::SimpleManifoldCachedObjective)
    return (M, p) -> get_gradient(M, sco, p)
end

function get_gradient(M::AbstractManifold, sco::SimpleManifoldCachedObjective, p)
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
function get_gradient!(M::AbstractManifold, X, sco::SimpleManifoldCachedObjective, p)
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

#
# CostGradImplementation
#
function get_cost(
    M::AbstractManifold,
    sco::SimpleManifoldCachedObjective{
        AllocatingEvaluation,TC,TG,<:ManifoldCostGradientObjective
    },
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
    sco::SimpleManifoldCachedObjective{
        InplaceEvaluation,TC,TG,<:ManifoldCostGradientObjective
    },
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
    sco::SimpleManifoldCachedObjective{
        AllocatingEvaluation,TC,TG,<:ManifoldCostGradientObjective
    },
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
    sco::SimpleManifoldCachedObjective{
        InplaceEvaluation,TC,TG,<:ManifoldCostGradientObjective
    },
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
    sco::SimpleManifoldCachedObjective{
        AllocatingEvaluation,TC,TG,<:ManifoldCostGradientObjective
    },
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
    sco::SimpleManifoldCachedObjective{
        InplaceEvaluation,TC,TG,<:ManifoldCostGradientObjective
    },
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
# ManifoldCachedObjective constructor which errors by default since we can only define init
# for LRU Caches
#
@doc raw"""
    ManifoldCachedObjective{E,P,O<:AbstractManifoldObjective{<:E},C<:NamedTuple{}} <: AbstractDecoratedManifoldObjective{E,P}

Create a cache for an objective, based on a `NamedTuple` that stores some kind of cache.

# Constructor

    ManifoldCachedObjective(M, o::AbstractManifoldObjective, caches::Vector{Symbol}; kwargs...)

Create a cache for the [`AbstractManifoldObjective`](@ref) where the Symbols in `caches` indicate,
which function evaluations to cache.

# Supported Symbols

| Symbol                      | Caches calls to (incl. `!` variants`)  | Comment
| --------------------------- | -------------------------------------- | ------------------------- |
| `:Constraints`              | [`get_constraints`](@ref)              | vector of numbers         |
| `:Cost`                     | [`get_cost`](@ref)                     |                           |
| `:EqualityConstraint`       | [`get_equality_constraint`](@ref)      | numbers per (p,i)         |
| `:EqualityConstraints`      | [`get_equality_constraints`](@ref)     | vector of numbers         |
| `:GradEqualityConstraint`   | [`get_grad_equality_constraint`](@ref) | tangent vector per (p,i)  |
| `:GradEqualityConstraints`  | [`get_grad_equality_constraints`](@ref)| vector of tangent vectors |
| `:GradInequalityConstraint` | [`get_inequality_constraint`](@ref)    | tangent vector per (p,i)  |
| `:GradInequalityConstraints`| [`get_inequality_constraints`](@ref)   | vector of tangent vectors |
| `:Gradient`                 | [`get_gradient`](@ref)`(M,p)`          | tangent vectors           |
| `:Hessian`                  | [`get_hessian`](@ref)                  | tangent vectors           |
| `:InequalityConstraint`     | [`get_inequality_constraint`](@ref)    | numbers per (p,j)         |
| `:InequalityConstraints`    | [`get_inequality_constraints`](@ref)   | vector of numbers         |
| `:Preconditioner`           | [`get_preconditioner`](@ref)           | tangent vectors           |
| `:ProximalMap`              | [`get_proximal_map`](@ref)             | point per `(p,λ,i)`       |
| `:SubGradient`              | [`get_subgradient`](@ref)              | tangent vectors           |
| `:SubtrahendGradient`       | [`get_subtrahend_gradient`](@ref)      | tangent vectors           |

# Keyword Arguments

* `p`           - (`rand(M)`) the type of the keys to be used in the caches. Defaults to the default representation on `M`.
* `v`           - (`get_cost(M, objective, p)`) the type of values for numeric values in the cache, e.g. the cost
* `X`           - (`zero_vector(M,p)`) the type of values to be cached for gradient and Hessian calls.
* `cache`       - (`[:Cost]`) a vector of symbols indicating which function calls should be cached.
* `cache_size`  - (`10`) number of (least recently used) calls to cache
* `cache_sizes` – (`Dict{Symbol,Int}()`) a named tuple or dictionary specifying the sizes individually for each cache.


"""
struct ManifoldCachedObjective{E,P,O<:AbstractManifoldObjective{<:E},C<:NamedTuple{}} <:
       AbstractDecoratedManifoldObjective{E,P}
    objective::O
    cache::C
end
function ManifoldCachedObjective(
    M::AbstractManifold,
    objective::O,
    caches::AbstractVector{<:Symbol}=[:Cost];
    p::P=rand(M),
    v::R=get_cost(M, objective, p),
    X::T=zero_vector(M, p),
    cache_size=10,
    cache_sizes=Dict{Symbol,Int}(),
) where {E,O<:AbstractManifoldObjective{E},R,P,T}
    c = init_caches(
        M, caches; p=p, v=v, X=X, cache_size=cache_size, cache_sizes=cache_sizes
    )
    return ManifoldCachedObjective{E,O,O,typeof(c)}(objective, c)
end
function ManifoldCachedObjective(
    M::AbstractManifold,
    objective::O,
    caches::AbstractVector{<:Symbol}=[:Cost];
    p::P=rand(M),
    v::R=get_cost(M, objective, p),
    X::T=zero_vector(M, p),
    cache_size=10,
    cache_sizes=Dict{Symbol,Int}(),
) where {E,O2,O<:AbstractDecoratedManifoldObjective{E,O2},R,P,T}
    c = init_caches(
        M, caches; p=p, v=v, X=X, cache_size=cache_size, cache_sizes=cache_sizes
    )
    return ManifoldCachedObjective{E,O2,O,typeof(c)}(objective, c)
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
# Default implementations - (a) check if field exists (b) try to get cache
function get_cost(M::AbstractManifold, co::ManifoldCachedObjective, p)
    !(haskey(co.cache, :Cost)) && return get_cost(M, co.objective, p)
    return get!(co.cache[:Cost], copy(M, p)) do
        get_cost(M, co.objective, p)
    end
end
get_cost_function(co::ManifoldCachedObjective) = (M, p) -> get_cost(M, co, p)
get_gradient_function(co::ManifoldCachedObjective) = (M, p) -> get_gradient(M, co, p)

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
        end, #and we copy the values back to X
    )
    return X
end

#
# CostGradImplementation
function get_cost(
    M::AbstractManifold, co::ManifoldCachedObjective{E,<:ManifoldCostGradientObjective}, p
) where {E<:AbstractEvaluationType}
    #Neither cost not grad cached -> evaluate
    all(.!(haskey.(Ref(co.cache), [:Cost, :Gradient]))) &&
        return get_cost(M, co.objective, p)
    return get!(co.cache[:Cost], copy(M, p)) do
        c, X = get_cost_and_gradient(M, co.objective, p)
        #if this is evaluated, we can also set X
        haskey(co.cache, :Gradient) && setindex!(co.cache[:Gradient], X, copy(M, p))
        c #but we also set the new cost here
    end
end
function get_gradient(
    M::AbstractManifold, co::ManifoldCachedObjective{E,<:ManifoldCostGradientObjective}, p
) where {E<:AllocatingEvaluation}
    all(.!(haskey.(Ref(co.cache), [:Cost, :Gradient]))) &&
        return get_gradient(M, co.objective, p)
    return copy(
        M,
        p,
        get!(co.cache[:Gradient], copy(M, p)) do
            c, X = get_cost_and_gradient(M, co.objective, p)
            #if this is evaluated, we can also set c
            haskey(co.cache, :Cost) && setindex!(co.cache[:Cost], c, copy(M, p))
            X #but we also set the new cost here
        end,
    )
end
function get_gradient!(
    M::AbstractManifold,
    X,
    co::ManifoldCachedObjective{E,<:ManifoldCostGradientObjective},
    p,
) where {E}
    all(.!(haskey.(Ref(co.cache), [:Cost, :Gradient]))) &&
        return get_gradient!(M, X, co.objective, p)
    return copyto!(
        M,
        X,
        p,
        get!(co.cache[:Gradient], copy(M, p)) do
            c, _ = get_cost_and_gradient!(M, X, get_objective(co.objective), p)
            #if this is evaluated, we can also set c
            haskey(co.cache, :Cost) && setindex!(co.cache[:Cost], c, copy(M, p))
            X
        end,
    )
end

#
# Constraints
function get_constraints(M::AbstractManifold, co::ManifoldCachedObjective, p)
    all(.!(haskey.(Ref(co.cache), [:Constraints]))) &&
        return get_constraints(M, co.objective, p)
    return copy(
        get!(co.cache[:Constraints], copy(M, p)) do
            get_constraints(M, co.objective, p)
        end,
    )
end
function get_equality_constraints(M::AbstractManifold, co::ManifoldCachedObjective, p)
    all(.!(haskey.(Ref(co.cache), [:EqualityConstraints]))) &&
        return get_equality_constraints(M, co.objective, p)
    return copy(
        get!(co.cache[:EqualityConstraints], copy(M, p)) do
            get_equality_constraints(M, co.objective, p)
        end,
    )
end
function get_equality_constraint(M::AbstractManifold, co::ManifoldCachedObjective, p, i)
    all(.!(haskey.(Ref(co.cache), [:EqualityConstraint]))) &&
        return get_equality_constraint(M, co.objective, p, i)
    return copy(
        get!(co.cache[:EqualityConstraint], (copy(M, p), i)) do
            get_equality_constraint(M, co.objective, p, i)
        end,
    )
end
function get_inequality_constraints(M::AbstractManifold, co::ManifoldCachedObjective, p)
    all(.!(haskey.(Ref(co.cache), [:InequalityConstraints]))) &&
        return get_inequality_constraints(M, co.objective, p)
    return copy(
        get!(co.cache[:InequalityConstraints], copy(M, p)) do
            get_inequality_constraints(M, co.objective, p)
        end,
    )
end
function get_inequality_constraint(M::AbstractManifold, co::ManifoldCachedObjective, p, i)
    all(.!(haskey.(Ref(co.cache), [:InequalityConstraint]))) &&
        return get_inequality_constraint(M, co.objective, p, i)
    return copy(
        get!(co.cache[:InequalityConstraints], (copy(M, p), i)) do
            get_inequality_constraint(M, co.objective, p, i)
        end,
    )
end
#
# Gradients of Constraints
function get_grad_equality_constraint(
    M::AbstractManifold, co::ManifoldCachedObjective, p, j
)
    !(haskey(co.cache, :GradEqualityConstraint)) &&
        return get_grad_equality_constraint(M, co.objective, p, j)
    return copy(
        M,
        p,
        get!(co.cache[:GradEqualityConstraint], (copy(M, p), j)) do
            get_grad_equality_constraint(M, co.objective, p, j)
        end,
    )
end
function get_grad_equality_constraint!(
    M::AbstractManifold, X, co::ManifoldCachedObjective, p, j
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
        end, #and we copy the values back to X
    )
    return X
end

function get_grad_equality_constraints(M::AbstractManifold, co::ManifoldCachedObjective, p)
    !(haskey(co.cache, :GradEqualityConstraints)) &&
        return get_grad_equality_constraints(M, co.objective, p)
    return copy.(
        Ref(M),
        Ref(p),
        get!(co.cache[:GradEqualityConstraints], copy(M, p)) do
            get_grad_equality_constraints(M, co.objective, p)
        end,
    )
end
function get_grad_equality_constraints!(
    M::AbstractManifold, X, co::ManifoldCachedObjective, p
)
    !(haskey(co.cache, :GradEqualityConstraints)) &&
        return get_grad_equality_constraints!(M, X, co.objective, p)
    copyto.(
        Ref(M),
        X,
        Ref(p),
        get!(co.cache[:GradEqualityConstraints], copy(M, p)) do
            # This evaluates in place of X
            get_grad_equality_constraints!(M, X, co.objective, p)
            copy.(Ref(M), Ref(p), X) #this creates a copy to be placed in the cache
        end, #and we copy the values back to X
    )
    return X
end

function get_grad_inequality_constraint(
    M::AbstractManifold, co::ManifoldCachedObjective, p, j
)
    !(haskey(co.cache, :GradInequalityConstraint)) &&
        return get_grad_inequality_constraint(M, co.objective, p, j)
    return copy(
        M,
        p,
        get!(co.cache[:GradInequalityConstraint], (copy(M, p), j)) do
            get_grad_inequality_constraint(M, co.objective, p, j)
        end,
    )
end
function get_grad_inequality_constraint!(
    M::AbstractManifold, X, co::ManifoldCachedObjective, p, j
)
    !(haskey(co.cache, :GradInequalityConstraint)) &&
        return get_grad_inequality_constraint!(M, X, co.objective, p, j)
    copyto!(
        M,
        X,
        p,
        get!(co.cache[:GradInequalityConstraint], (copy(M, p), j)) do
            # This evaluates in place of X
            get_grad_inequality_constraint!(M, X, co.objective, p, j)
            copy(M, p, X) #this creates a copy to be placed in the cache
        end, #and we copy the values back to X
    )
    return X
end

function get_grad_inequality_constraints(
    M::AbstractManifold, co::ManifoldCachedObjective, p
)
    !(haskey(co.cache, :GradEqualityConstraints)) &&
        return get_grad_inequality_constraints(M, co.objective, p)
    return copy.(
        Ref(M),
        Ref(p),
        get!(co.cache[:GradInequalityConstraints], copy(M, p)) do
            get_grad_inequality_constraints(M, co.objective, p)
        end,
    )
end
function get_grad_inequality_constraints!(
    M::AbstractManifold, X, co::ManifoldCachedObjective, p
)
    !(haskey(co.cache, :GradInequalityConstraints)) &&
        return get_grad_inequality_constraints!(M, X, co.objective, p)
    copyto.(
        Ref(M),
        X,
        Ref(p),
        get!(co.cache[:GradInequalityConstraints], copy(M, p)) do
            # This evaluates in place of X
            get_grad_inequality_constraints!(M, X, co.objective, p)
            copy.(Ref(M), Ref(p), X) #this creates a copy to be placed in the cache
        end, #and we copy the values back to X
    )
    return X
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
        p, #for the tricks performed here see get_gradient!
        get!(co.cache[:Hessian], (copy(M, p), copy(M, p, X))) do
            get_hessian!(M, Y, co.objective, p, X)
            copy(M, p, Y) #store a copy of Y
        end,
    )
    return Y
end

#
# Preconditioner
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
        p, #for the tricks performed here see get_gradient!
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
        get!(co.cache[:ProximalMap], (copy(M, p), λ, i)) do #use the tuple (p,i) as key
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
            get_proximal_map!(M, q, co.objective, λ, p, i) #compute inplace of q
            copy(M, q) #store copy of q
        end,
    )
    return q
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
        p, #for the tricks performed here see get_gradient!
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
        p, #for the tricks performed here see get_gradient!
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
  i.e. to cache up to 100 cost and gradient values.[^1]

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
where the second element `cache[2]` is are further arguments to  the cache and
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
