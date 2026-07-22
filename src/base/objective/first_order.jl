@doc """
    AbstractManifoldFirstOrderObjective{F,G} <: AbstractManifoldCostObjective{F}

An abstract type for all objectives that provide

* a cost – reflected by the type `F`
* first order information, so either a (full) gradient or a differential,
  or a subgradient – reflected by the parameter `G`.
"""
abstract type AbstractManifoldFirstOrderObjective{F, G} <:
AbstractManifoldCostObjective{F} end

"""
    DirectionUpdateRule

A general functor, that handles direction update rules. It's fields are usually
only a [`StoreStateAction`](@ref) by default initialized to the fields required
for the specific coefficient, but can also be replaced by a (common, global)
individual one that provides these values.
"""
abstract type DirectionUpdateRule end

@doc """
    (c, d) = get_cost_and_differential(M, objective::AbstractManifoldFirstOrderObjective, p, X; Y=nothing)

Evaluate the cost and the differential of a [`AbstractManifoldFirstOrderObjective`] `objective` at a point `p` in direction `X`.

The keyword argument `Y` can be used to provide memory for the evaluation of the gradient,
in case no separate differential is provided.
"""
function get_cost_and_differential end

#Undecorate decorators by default
function get_cost_and_differential(
        M::AbstractManifold, objective::AbstractDecoratedManifoldObjective, p, X; kwargs...
    )
    return get_cost_and_differential(M, get_objective(objective, false), p, X; kwargs...)
end

@doc """
    (c, d) = get_cost_and_differential(problem::AbstractManoptProblem, p, X; kwargs...)

Evaluate the cost and the differential of an objective and the manifold inside the [`AbstractManoptProblem`](@ref) `problem`.

The keyword arguments are passed down to the objective evaluation.
"""
function get_cost_and_differential(amp::AbstractManoptProblem, p, X; kwargs...)
    return get_cost_and_differential(get_manifold(amp), get_objective(amp), p, X; kwargs...)
end

function get_cost_and_gradient! end
@doc """
    (c, X) = get_cost_and_gradient(M, objective::AbstractManifoldFirstOrderObjective, p)
    (c, X) = get_cost_and_gradient!(M, X, objective::AbstractManifoldFirstOrderObjective, p)

Evaluate the cost and the gradient of a [`AbstractManifoldFirstOrderObjective`] `objective` at a point `p`
simultaneously. The gradient part can be evaluated in-place of `X`
"""
function get_cost_and_gradient(M, objective::AbstractManifoldFirstOrderObjective, p)
    X = zero_vector(M, p)
    return get_cost_and_gradient!(M, objective, objective, p)
end
# TODO USe the same 3 line signature in the other places of this file as well.
"""
     get_differential(amp::AbstractManoptProblem, p, X; kwargs...)
     get_differential(M::AbstractManifold, amfo:AbstractManifoldFirstOrderObjective, p, X; kwargs...)
     get_differential(M::AbstractManifold, amfo:AbstractDecoratedManifoldObjective, p, X; kwargs...)

Evaluate the differential ``Df(p)[X]`` of the function ``f`` represented by
the [`AbstractManifoldFirstOrderObjective`](@ref).
For [`AbstractManoptProblem`](@ref) the inner manifold and objectives are used,
similarly, any objective decorator would “pass though” to its inner objective.
By default this falls back to ``Df(p)[X] = ⟨$(_tex(:grad))f(p), X⟩``.

# Keyword arguments
* `gradient=nothing` – pass a tangent vector to be used internally as interims memory,
  e.g. in the default variant to evaluate the gradient in-place in.
* `evaluated=false` – indicate whether `gradient` is just memory (`false`, default) or
  already contains the evaluated gradient (`true`).
"""
function get_differential(
        M::AbstractManifold, objective::AbstractManifoldFirstOrderObjective, p, X;
        gradient = nothing, evaluated::Bool = false,
    )
    isnothing(gradient) && (return real(inner(M, p, get_gradient(M, objective, p), X)))
    # if it is not nothing call in-place
    (!evaluated) && (get_gradient!(M, gradient, objective, p))
    return real(inner(M, p, gradient, X))
end
function get_differential(
        M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p, X; kwargs...
    )
    return get_differential(M, get_objective(admo, false), p, X; kwargs...)
end

@doc """
    d = get_differential(get_differential(problem::AbstractManoptProblem, p, X; kwargs...)

Evaluate the differential of an objective and the manifold inside the [`AbstractManoptProblem`](@ref) `problem`.

The keyword arguments are passed down to the objective evaluation.
"""
function get_differential(problem::AbstractManoptProblem, p, X; kwargs...)
    return get_differential(get_manifold(problem), get_objective(problem), p, X; kwargs...)
end


@doc """
    get_gradient(M, objective::AbstractManifoldFirstOrderObjective, p)
    get_gradient!(M, X, objective::AbstractManifoldFirstOrderObjective, p)

Evaluate the gradient of a [`AbstractManifoldFirstOrderObjective`] `objective` at a point `p`.
This can be evaluated in-place of `X`.
"""
function get_gradient(M, objective::AbstractManifoldFirstOrderObjective, p)
    X = zero_vector(M, p)
    return get_gradient!(M, X, objective, p)
end

# Decorator case
function get_gradient(M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p)
    return get_gradient(M, get_objective(admo, false), p)
end
function get_gradient!(M::AbstractManifold, X, admo::AbstractDecoratedManifoldObjective, p)
    return get_gradient!(M, X, get_objective(admo, false), p)
end
