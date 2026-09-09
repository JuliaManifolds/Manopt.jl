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

@doc """
    get_preconditioner(amp::AbstractManoptProblem, p, X)

Evaluate the preconditioner of the objective of the [`AbstractManoptProblem`](@ref) `amp`
at the point `p`, applied to the tangent vector `X`.

It usually is a symmetric, positive definite approximation of the inverse of the Hessian of the cost function `f`.
"""
function get_preconditioner(amp::AbstractManoptProblem, p, X)
    return get_preconditioner(get_manifold(amp), get_objective(amp), p, X)
end
function get_preconditioner!(amp::AbstractManoptProblem, Y, p, X)
    return get_preconditioner!(get_manifold(amp), Y, get_objective(amp), p, X)
end

@doc """
    Y = get_hessian(amp::AbstractManoptProblem, p, X)
    get_hessian!(amp::AbstractManoptProblem, Y, p, X)

Evaluate the Hessian of an [`AbstractManoptProblem`](@ref) `amp` at `p`
applied to a tangent vector `X`, computing ``$(_tex(:Hess))f(p)[X]``,
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

Return the manifold stored within an [`AbstractManoptProblem`](@ref).
"""
get_manifold(::AbstractManoptProblem)

function get_objective end

@doc """
    get_objective(mp::AbstractManoptProblem, recursive=false)

Return the objective [`AbstractManifoldObjective`](@ref) stored within an [`AbstractManoptProblem`](@ref).
If `recursive` is set to `true`, it additionally unwraps all decorators of the objective.

This default assumes that the objective is stored in the field `objective`.
A problem that stores it elsewhere has to implement this method itself.
"""
function get_objective(mp::AbstractManoptProblem, recursive = false)
    return recursive ? get_objective(mp.objective, true) : mp.objective
end

@doc """
    get_cost(amp::AbstractManoptProblem, p)

Evaluate the cost function `f` stored within the [`AbstractManifoldObjective`](@ref) of an
[`AbstractManoptProblem`](@ref) `amp` at the point `p`.
"""
function get_cost(amp::AbstractManoptProblem, p)
    return get_cost(get_manifold(amp), get_objective(amp), p)
end


_doc_get_gradient_amp = """
    get_gradient(amp::AbstractManoptProblem, p)
    get_gradient!(amp::AbstractManoptProblem, X, p)

Evaluate the gradient of an [`AbstractManoptProblem`](@ref) `amp` at the point `p`.

This can also be computed in-place of `X` for the `!`-variant.
"""
@doc "$(_doc_get_gradient_amp)"
function get_gradient(mp::AbstractManoptProblem, p)
    return get_gradient(get_manifold(mp), get_objective(mp), p)
end
@doc "$(_doc_get_gradient_amp)"
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

_doc_get_subtrahend_gradient = """
    X = get_subtrahend_gradient(amp, p)
    get_subtrahend_gradient!(amp, X, p)

Evaluate the (sub)gradient of the subtrahend `h` from within the
[`ManifoldDifferenceOfConvexObjective`](@ref) of an [`AbstractManoptProblem`](@ref) `amp`
at the point `p`.

The evaluation is done in place of `X` for the `!`-variant.
An objective using [`AllocatingEvaluation`](@ref) might still allocate memory within.
When the non-mutating variant is called with an [`InplaceEvaluation`](@ref),
memory for the result is allocated.
"""

@doc "$(_doc_get_subtrahend_gradient)"
function get_subtrahend_gradient(amp::AbstractManoptProblem, p)
    return get_subtrahend_gradient(get_manifold(amp), get_objective(amp), p)
end
@doc "$(_doc_get_subtrahend_gradient)"
function get_subtrahend_gradient!(amp::AbstractManoptProblem, X, p)
    get_subtrahend_gradient!(get_manifold(amp), X, get_objective(amp), p)
    return X
end

"""
    set_parameter!(amp::AbstractManoptProblem, element::Symbol, field::Symbol, value)

Set a certain field/element from the [`AbstractManoptProblem`](@ref) `amp` to `value`.
This function usually dispatches on `Val(element)`.
Instead of a single field, also a chain of elements can be provided, allowing access to
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
