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

"""
    AbstractGradientGroupDirectionRule <: DirectionUpdateRule

A generic type for all options [`DirectionUpdateRule`](@ref) working with certain splittiungs of the overall gradient in the direction processing.
"""
abstract type AbstractGradientGroupDirectionRule <: DirectionUpdateRule end


@doc """
    (c, d) = get_cost_and_differential(problem::AbstractManoptProblem, p, X; kwargs...)
    (c, d) = get_cost_and_differential(M, objective::AbstractManifoldFirstOrderObjective, p, X; kwargs...)

Evaluate the cost and the differential of a [`AbstractManifoldFirstOrderObjective`] `objective` at a point `p` in direction `X`.
For an [`AbstractManoptProblem`](@ref) `problem` the inner manifold and objectives are used.
Similarly, any objective decorator would “pass though” to its inner objective.

Keyword arguments are passed down.
"""
function get_cost_and_differential end

function get_cost_and_differential(
        M::AbstractManifold, objective::AbstractDecoratedManifoldObjective, p, X; kwargs...
    )
    return get_cost_and_differential(M, get_objective(objective, false), p, X; kwargs...)
end

function get_cost_and_differential(amp::AbstractManoptProblem, p, X; kwargs...)
    return get_cost_and_differential(get_manifold(amp), get_objective(amp), p, X; kwargs...)
end

function get_cost_and_gradient! end
@doc """
    (c, X) = get_cost_and_gradient(problem::AbstractManoptProblem, p)
    (c, X) = get_cost_and_gradient(M, objective::AbstractManifoldFirstOrderObjective, p)
    (c, X) = get_cost_and_gradient!(M, X, objective::AbstractManifoldFirstOrderObjective, p)

Evaluate the cost and the gradient of a [`AbstractManifoldFirstOrderObjective`] `objective` at a point `p`
simultaneously.
For an [`AbstractManoptProblem`](@ref) `problem` the inner manifold and objectives are used.
Similarly, any objective decorator would “pass though” to its inner objective.

The gradient part can be evaluated in-place of `X`.
"""
function get_cost_and_gradient(M, objective::AbstractManifoldFirstOrderObjective, p)
    X = zero_vector(M, p)
    return get_cost_and_gradient!(M, X, objective, p)
end
function get_cost_and_gradient(problem::AbstractManoptProblem, p)
    return get_cost_and_gradient(get_manifold(problem), get_objective(problem), p)
end
function get_cost_and_gradient(
        M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p
    )
    return get_cost_and_gradient(M, get_objective(admo, false), p)
end
function get_cost_and_gradient(
        M::AbstractManifold, mfo::AbstractManifoldFirstOrderObjective, p
    )
    X = zero_vector(M, p)
    return get_cost_and_gradient!(M, X, mfo, p)
end
function get_cost_and_gradient!(
        M::AbstractManifold, X, admo::AbstractDecoratedManifoldObjective, p
    )
    return get_cost_and_gradient!(M, X, get_objective(admo, false), p)
end

"""
     get_differential(amp::AbstractManoptProblem, p, X; kwargs...)
     get_differential(M::AbstractManifold, amfo:AbstractManifoldFirstOrderObjective, p, X; kwargs...)
     get_differential(M::AbstractManifold, amfo:AbstractDecoratedManifoldObjective, p, X; kwargs...)

Evaluate the differential ``Df(p)[X]`` of the function ``f`` represented by
the [`AbstractManifoldFirstOrderObjective`](@ref).
For an [`AbstractManoptProblem`](@ref) `problem` the inner manifold and objectives are used.
Similarly, any objective decorator would “pass though” to its inner objective.
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
    d = get_differential(problem::AbstractManoptProblem, p, X; kwargs...)

Evaluate the differential of an objective and the manifold inside the [`AbstractManoptProblem`](@ref) `problem`.

The keyword arguments are passed down to the objective evaluation.
"""
function get_differential(problem::AbstractManoptProblem, p, X; kwargs...)
    return get_differential(get_manifold(problem), get_objective(problem), p, X; kwargs...)
end

# Differential function - pass-through

function get_differential_function end

@doc """
     get_differential_function(objective::AbstractManifoldFirstOrderObjective, recursive::Bool=false)

Return the function to evaluate (just) the differential ``Df(p)[X]``.
For a decorated objective, the `recursive` positional parameter determines whether to
directly call this function on the next decorator or whether to get the “most inner” objective.
"""
get_differential_function(::AbstractManifoldFirstOrderObjective; recursive::Bool = false)

function get_differential_function(
        objective::AbstractDecoratedManifoldObjective, recursive = false
    )
    return get_differential_function(get_objective(objective, recursive))
end

@doc """
    get_gradient(problem::AbstractManoptProblem, p)
    get_gradient(M, objective::AbstractManifoldFirstOrderObjective, p)
    get_gradient!(problem::AbstractManoptProblem, X, p)
    get_gradient!(M, X, objective::AbstractManifoldFirstOrderObjective, p)

Evaluate the gradient of a [`AbstractManifoldFirstOrderObjective`] `objective` on an $(_link(:AbstractManifold)) `M` at a point `p`.
This can be evaluated in-place of `X` and also when passing and [`AbstractManoptProblem`](@ref) `problem`.
"""
function get_gradient(M::AbstractManifold, objective::AbstractManifoldFirstOrderObjective, p)
    X = zero_vector(M, p)
    return get_gradient!(M, X, objective, p)
end

### Decorator
function get_gradient(M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p)
    return get_gradient(M, get_objective(admo, false), p)
end
function get_gradient!(M::AbstractManifold, X, admo::AbstractDecoratedManifoldObjective, p)
    return get_gradient!(M, X, get_objective(admo, false), p)
end

function get_gradient_function end
@doc """
    get_gradient_function(amgo::AbstractManifoldFirstOrderObjective, recursive=false; evaluation=AllocatingEvaluation())

return the function to evaluate (just) the gradient ``$(_tex(:grad)) f(p)``,
where either the gradient function using the decorator or without the decorator is used.

By default `recursive` is set to `false`, since usually to just pass the gradient function
somewhere, one still wants for example the cached one or the one that still counts calls.
"""
get_gradient_function(::AbstractManifoldFirstOrderObjective, recursive = false; evaluation::AbstractEvaluationType = AllocatingEvaluation())

function get_gradient_function(admo::AbstractDecoratedManifoldObjective, recursive = false; evaluation::AbstractEvaluationType = AllocatingEvaluation())
    return get_gradient_function(get_objective(admo, recursive); evaluation = evaluation)
end

"""
    X = get_subgradient(M::AbstractManifold, sgo::AbstractManifoldFirstOrderObjective, p)
    get_subgradient!(M::AbstractManifold, X, sgo::AbstractManifoldFirstOrderObjective, p)

Evaluate the subgradient, which for the case of a objective having a gradient, means evaluating the
gradient itself.

While in general, the result might not be deterministic, for this case it is.
"""
function get_subgradient(M::AbstractManifold, agmo::AbstractManifoldFirstOrderObjective, p)
    X = zero_vector(M, p)
    return get_subgradient!(M, X, agmo, p)
end
function get_subgradient!(
        M::AbstractManifold, X, agmo::AbstractManifoldFirstOrderObjective, p
    )
    return get_gradient!(M, X, agmo, p)
end
