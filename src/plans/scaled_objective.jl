@doc """
    ScaledManifoldObjective{E, O2, O1<:AbstractManifoldObjective{E},F} <:
       AbstractDecoratedManifoldObjective{E,O2}

Declare an objective to be defined as a scaled version of an existing objective.

This rescales all involved functions.

For now the functions rescaled are

* the cost
* the gradient
* the Hessian

# Fields

* `objective`: the objective that is defined in the embedding
* `scale=1`: the scaling applied

# Constructors

    ScaledManifoldObjective(objective, scale::Real=1)
    -objective
    scale*objective

Generate a scaled manifold objective based on `objective` with `scale` being `1` by default
in the first, `scale=-1` in the second case. The multiplication from the left with a scalar
is also overloaded.
"""
struct ScaledManifoldObjective{
        E <: AbstractEvaluationType, O2, O1 <: AbstractManifoldObjective{E}, F,
    } <: AbstractDecoratedManifoldObjective{E, O2}
    objective::O1
    scale::F
end
function ScaledManifoldObjective(
        objective::O, scale::F = 1
    ) where {E <: AbstractEvaluationType, O <: AbstractManifoldObjective{E}, F <: Real}
    return ScaledManifoldObjective{E, O, O, F}(objective, scale)
end
function ScaledManifoldObjective(
        objective::O1, scale::F = 1
    ) where {
        F <: Real,
        E <: AbstractEvaluationType,
        O2 <: AbstractManifoldObjective,
        O1 <: AbstractDecoratedManifoldObjective{E, O2},
    }
    return ScaledManifoldObjective{E, O2, O1, F}(objective, scale)
end
Base.:-(objective::AbstractManifoldObjective) = ScaledManifoldObjective(objective, -1)
function Base.:*(scale::Real, objective::AbstractManifoldObjective)
    return ScaledManifoldObjective(objective, scale)
end

@doc """
    get_cost(M::AbstractManifold, scaled_objective::ScaledManifoldObjective, p)

Evaluate the scaled objective. ``s*f(p)``
"""
function get_cost(M::AbstractManifold, scaled_objective::ScaledManifoldObjective, p)
    return scaled_objective.scale * get_cost(M, scaled_objective.objective, p)
end

function get_cost_function(scaled_objective::ScaledManifoldObjective, recursive::Bool = false)
    recursive && (return get_cost_function(scaled_objective.objective, recursive))
    return (M, p) -> get_cost(M, scaled_objective, p)
end
@doc """
    get_gradient(M::AbstractManifold, scaled_objective::ScaledManifoldObjective, p)
    get_gradient!(M::AbstractManifold, X, scaled_objective::ScaledManifoldObjective, p)

Evaluate the scaled gradient. ``s*$(_tex(:grad))f(p)``
"""
function get_gradient(M::AbstractManifold, scaled_objective::ScaledManifoldObjective, p)
    return scaled_objective.scale * get_gradient(M, scaled_objective.objective, p)
end
function get_gradient!(M::AbstractManifold, X, scaled_objective::ScaledManifoldObjective, p)
    get_gradient!(M, X, scaled_objective.objective, p)
    X .= scaled_objective.scale .* X
    return X
end

function get_gradient_function(
        scaled_objective::ScaledManifoldObjective{AllocatingEvaluation}, recursive::Bool = false
    )
    recursive && (return get_gradient_function(scaled_objective.objective, recursive))
    return (M, p) -> get_gradient(M, scaled_objective, p)
end
function get_gradient_function(
        scaled_objective::ScaledManifoldObjective{InplaceEvaluation}, recursive::Bool = false
    )
    recursive && (return get_gradient_function(scaled_objective.objective, recursive))
    return (M, X, p) -> get_gradient!(M, X, scaled_objective, p)
end
#
# Hessian
#
@doc """
    get_hessian(M::AbstractManifold, scaled_objective::ScaledManifoldObjective, p, X)
    get_hessian!(M::AbstractManifold, Y, scaled_objective::ScaledManifoldObjective, p, X)

Evaluate the scaled Hessian ``s*$(_tex(:Hess))f(p)``
"""
function get_hessian(M::AbstractManifold, scaled_objective::ScaledManifoldObjective, p, X)
    return scaled_objective.scale * get_hessian(M, scaled_objective.objective, p, X)
end
function get_hessian!(
        M::AbstractManifold, Y, scaled_objective::ScaledManifoldObjective, p, X
    )
    get_hessian!(M, Y, scaled_objective.objective, p, X)
    Y .= scaled_objective.scale .* Y
    return Y
end

function get_hessian_function(
        scaled_objective::ScaledManifoldObjective{AllocatingEvaluation}, recursive::Bool = false
    )
    recursive && (return get_hessian_function(scaled_objective.objective, recursive))
    return (M, p, X) -> get_hessian(M, scaled_objective, p, X)
end
function get_hessian_function(
        scaled_objective::ScaledManifoldObjective{InplaceEvaluation}, recursive::Bool = false
    )
    recursive && (return get_hessian_function(scaled_objective.objective, recursive))
    return (M, Y, p, X) -> get_hessian!(M, Y, scaled_objective, p, X)
end

function show(io::IO, scaled_objective::ScaledManifoldObjective{P, T}) where {P, T}
    return print(
        io,
        "ScaledManifoldObjective based on a $(scaled_objective.objective) with scale $(scaled_objective.scale)",
    )
end
