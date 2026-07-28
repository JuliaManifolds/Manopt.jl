@doc """
    AbstractManoptProblem{M<:AbstractManifold}

Describe a Riemannian optimization problem with all static (not-changing) properties.

The most prominent features that should always be stated here are

* the [`AbstractManifold`](@extref `ManifoldsBase.AbstractManifold`) ``$(_math(:Manifold))``
* the cost function ``f:  $(_math(:Manifold)) → ℝ``

Usually the cost should be within an [`AbstractManifoldObjective`](@ref).
"""
abstract type AbstractManoptProblem{M <: AbstractManifold} end

function Base.show(io::IO, ::MIME"text/plain", amp::AbstractManoptProblem)
    multiline = get(io, :multiline, true)
    return multiline ? status_summary(io, amp) : show(io, amp)
end

"""
    evaluation_type(mp::AbstractManoptProblem)

Get the [`AbstractEvaluationType`](@ref) of the objective in [`AbstractManoptProblem`](@ref)
`mp`.
"""
evaluation_type(amp::AbstractManoptProblem) = evaluation_type(get_objective(amp))

@doc """
    get_preconditioner(amp::AbstractManoptProblem, p, X)

evaluate the symmetric, positive definite preconditioner (approximation of the
inverse of the Hessian of the cost function `f`) of a
[`AbstractManoptProblem`](@ref) `amp`s objective at the point `p` applied to a
tangent vector `X`.
"""
function get_preconditioner(amp::AbstractManoptProblem, p, X)
    return get_preconditioner(get_manifold(amp), get_objective(amp), p, X)
end
function get_preconditioner!(amp::AbstractManoptProblem, Y, p, X)
    return get_preconditioner!(get_manifold(amp), Y, get_objective(amp), p, X)
end

@doc """
    Y = get_hessian(amp::AbstractManoptProblem{T}, p, X)
    get_hessian!(amp::AbstractManoptProblem{T}, Y, p, X)

evaluate the Hessian of an [`AbstractManoptProblem`](@ref) `amp` at `p`
applied to a tangent vector `X`, computing ``$(_tex(:Hess))f(q)[X]``,
which can also happen in-place of `Y`.
"""
function get_hessian(amp::AbstractManoptProblem, p, X)
    return get_hessian(get_manifold(amp), get_objective(amp), p, X)
end
function get_hessian!(amp::AbstractManoptProblem, Y, p, X)
    return get_hessian!(get_manifold(amp), Y, get_objective(amp), p, X)
end

function get_manifold end

@doc """
    get_manifold(amp::AbstractManoptProblem)

return the manifold stored within an [`AbstractManoptProblem`](@ref)
"""
get_manifold(::AbstractManoptProblem)

function get_objective end

@doc """
    get_objective(mp::AbstractManoptProblem, recursive=false)

return the objective [`AbstractManifoldObjective`](@ref) stored within an [`AbstractManoptProblem`](@ref).
If `recursive is set to true, it additionally unwraps all decorators of the objective`
"""
get_objective(::AbstractManoptProblem)

@doc """
    get_cost(amp::AbstractManoptProblem, p)

evaluate the cost function `f` stored within the [`AbstractManifoldObjective`](@ref) of an
[`AbstractManoptProblem`](@ref) `amp` at the point `p`.
"""
function get_cost(amp::AbstractManoptProblem, p)
    return get_cost(get_manifold(amp), get_objective(amp), p)
end

function get_gradient(mp::AbstractManoptProblem, p)
    return get_gradient(get_manifold(mp), get_objective(mp), p)
end
function get_gradient!(mp::AbstractManoptProblem, X, p)
    return get_gradient!(get_manifold(mp), X, get_objective(mp), p)
end
function get_gradient(mp::AbstractManoptProblem, p, k)
    return get_gradient(get_manifold(mp), get_objective(mp), p, k)
end
function get_gradient!(mp::AbstractManoptProblem, X, p, k)
    return get_gradient!(get_manifold(mp), X, get_objective(mp), p, k)
end

function get_gradients(mp::AbstractManoptProblem, p)
    return get_gradients(get_manifold(mp), get_objective(mp), p)
end
function get_gradients!(mp::AbstractManoptProblem, X, p)
    return get_gradients!(get_manifold(mp), X, get_objective(mp), p)
end

"""
    set_parameter!(ams::AbstractManoptProblem, element::Symbol, field::Symbol , value)

Set a certain field/element from the [`AbstractManoptProblem`](@ref) `ams` to `value`.
This function usually dispatches on `Val(element)`.
Instead of a single field, also a chain of elements can be provided, allowing to access
encapsulated parts of the problem.

Main values for `element` are `:Manifold` and `:Objective`.
"""
set_parameter!(amp::AbstractManoptProblem, e::Symbol, args...)

function set_parameter!(amp::AbstractManoptProblem, ::Val{:Manifold}, args...)
    set_parameter!(get_manifold(amp), args...)
    return amp
end
function set_parameter!(amp::AbstractManoptProblem, ::Val{:Objective}, args...)
    set_parameter!(get_objective(amp), args...)
    return amp
end
