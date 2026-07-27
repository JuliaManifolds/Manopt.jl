"""
    AbstractManifoldSubObjective{O<:AbstractManifoldObjective} <: AbstractManifoldObjective

An abstract type for objectives of sub problems within a solver but still store the
original objective internally to generate generic objectives for sub solvers.
"""
abstract type AbstractManifoldSubObjective{O <: AbstractManifoldObjective} <: AbstractManifoldObjective end

function get_gradient_function(amso::AbstractManifoldSubObjective)
    return (M, p) -> get_gradient(M, get_objective(amso), p)
end

@doc """
    get_objective(amso::AbstractManifoldSubObjective)

Return the (original) objective stored the sub objective is build on.
"""
get_objective(amso::AbstractManifoldSubObjective)

@doc """
    get_objective_cost(M, amso::AbstractManifoldSubObjective, p)

Evaluate the cost of the (original) objective stored within the sub objective.
"""
function get_objective_cost(M::AbstractManifold, amso::AbstractManifoldSubObjective, p)
    return get_cost(M, get_objective(amso), p)
end

@doc """
    X = get_objective_gradient(M, amso::AbstractManifoldSubObjective, p)
    get_objective_gradient!(M, X, amso::AbstractManifoldSubObjective, p)

Evaluate the gradient of the (original) objective stored within the sub objective `amso`.
"""
function get_objective_gradient(M::AbstractManifold, amso::AbstractManifoldSubObjective, p)
    return get_gradient(M, get_objective(amso), p)
end
function get_objective_gradient!(M::AbstractManifold, X, amso::AbstractManifoldSubObjective, p)
    return get_gradient!(M, X, get_objective(amso), p)
end

@doc """
    Y = get_objective_Hessian(M, amso::AbstractManifoldSubObjective, p, X)
    get_objective_Hessian!(M, Y, amso::AbstractManifoldSubObjective, p, X)

Evaluate the Hessian of the (original) objective stored within the sub objective `amso`.
"""
function get_objective_hessian(M::AbstractManifold, amso::AbstractManifoldSubObjective, p, X)
    return get_hessian(M, get_objective(amso), p, X)
end
function get_objective_hessian!(M::AbstractManifold, Y, amso::AbstractManifoldSubObjective, p, X)
    return get_hessian!(M, Y, get_objective(amso), p, X)
end

@doc """
    Y = get_objective_preconditioner(M, amso::AbstractManifoldSubObjective, p, X)
    get_objective_preconditioner(M, Y, amso::AbstractManifoldSubObjective, p, X)

Evaluate the Hessian of the (original) objective stored within the sub objective `amso`.
"""
function get_objective_preconditioner(M::AbstractManifold, amso::AbstractManifoldSubObjective, p, X)
    return get_preconditioner(M, get_objective(amso), p, X)
end
function get_objective_preconditioner!(M::AbstractManifold, Y, amso::AbstractManifoldSubObjective, p, X)
    return get_preconditioner!(M, Y, get_objective(amso), p, X)
end

"""
    AbstractLinearSurrogateObjective{O <: AbstractManifoldObjective} <: AbstractManifoldSubObjective

Provide a linear surrogate model for the given [`AbstractManifoldObjective`](@ref) `O` of the form

```math
μ_p(X) = $(_tex(:frac, "1", "2"))$(_tex(:norm, _tex(:Cal, "L") * "(X) + y"; index = "2"))^2
  + $(_tex(:frac, "λ", "2"))$(_tex(:norm, "X"; index = "p"))^2,
  $(_tex(:qquad))$(_tex(:text, " for "))X ∈ $(_math(:TangentSpace)), λ ≥ 0,
```

where ``$(_tex(:Cal, "L"))`` is a linear operator on the tangent space at a point ``p ∈ M``
that maps into some vector space ``V`` and ``y ∈ V`` is a fixed vector in that space
and ``$(_tex(:norm, "⋅"))`` is a norm on ``V``.

Both ``$(_tex(:Cal, "L"))`` and ``y`` are derived from the objective `O` and usually depend
on the base point ``p ∈ M``.

Besides the usual methods defined for [`AbstractManifoldObjective`](@ref) that may be implemented
like [`get_cost`](@ref) and [`get_gradient`](@ref), the following methods should be implemented
for a concrete subtype of `AbstractLinearSurrogateObjective`

* [`get_linear_operator`](@ref) to compute/evaluate the linear operator ``$(_tex(:Cal, "L"))``
* [`get_vector_field`](@ref) to compute/evaluate the vector ``y``
* [`get_objective`](@ref) to provide access to the underlying objective `O`

See also the [`NormalEquationsObjective`](@ref) for the corresponding normal equations.
"""
abstract type AbstractLinearSurrogateObjective{O <: AbstractManifoldObjective} <: AbstractManifoldSubObjective{O} end

"""
    get_objective(also::AbstractLinearSurrogateObjective)

Return the objective `O` associated with the linear surrogate model `also`.
By default, this returns `also.objective`.
"""
get_objective(also::AbstractLinearSurrogateObjective) = also.objective

function get_linear_operator end
"""
    get_linear_operator(M::AbstractManifold, lsmo::AbstractLinearSurrogateObjective, p)
    get_linear_operator(M::AbstractManifold, lsmo::AbstractLinearSurrogateObjective, p, B::AbstractBasis)
    get_linear_operator(M::AbstractManifold, lsmo::AbstractLinearSurrogateObjective, p, X)
    get_linear_operator!(M::AbstractManifold, L, lsmo::AbstractLinearSurrogateObjective, p, B::AbstractBasis)
    get_linear_operator!(M::AbstractManifold, Y, lsmo::AbstractLinearSurrogateObjective, p, X)

Return/Evaluate the linear operator ``$(_tex(:Cal, "L"))`` of the linear surrogate model `lsmo` at the point ``p ∈ M``.

If a tangent vector `X` is provided, evaluate ``$(_tex(:Cal, "L"))(X)``.
If a basis `B` is provided, return the matrix representation of ``$(_tex(:Cal, "L"))`` with respect to that basis.
Otherwise return the operator as a function `(TpM, X) -> Y`.
"""
get_linear_operator(M::AbstractManifold, lsmo::AbstractLinearSurrogateObjective, p, X = nothing)

function get_vector_field end
"""
    get_vector_field(M::AbstractManifold, lsmo::AbstractLinearSurrogateObjective, p)
    get_vector_field!(M::AbstractManifold, y, lsmo::AbstractLinearSurrogateObjective, p)

Return the vector `y` of the linear surrogate model `lsmo` at the point ``p ∈ M``.
"""
get_vector_field(M::AbstractManifold, lsmo::AbstractLinearSurrogateObjective, p)

function get_normal_linear_operator end
"""
    get_normal_linear_operator(M::AbstractManifold, lsmo::AbstractLinearSurrogateObjective, p)
    get_normal_linear_operator(M::AbstractManifold, lsmo::AbstractLinearSurrogateObjective, p, B::AbstractBasis)
    get_normal_linear_operator(M::AbstractManifold, lsmo::AbstractLinearSurrogateObjective, p, X)
    get_normal_linear_operator!(M::AbstractManifold, N, lsmo::AbstractLinearSurrogateObjective, p, B::AbstractBasis)
    get_normal_linear_operator!(M::AbstractManifold, Y, lsmo::AbstractLinearSurrogateObjective, p, X)

Return/Evaluate the normal operator ``$(_tex(:Cal, "L"))^* $(_tex(:Cal, "L"))`` of the linear surrogate model `lsmo` at the point ``p ∈ M``.

If a tangent vector `X` is provided, evaluate ``$(_tex(:Cal, "L"))^* $(_tex(:Cal, "L"))(X)``.
If a basis `B` is provided, return the matrix representation of ``$(_tex(:Cal, "L"))^* $(_tex(:Cal, "L"))`` with respect to that basis.
Otherwise return the operator as a function `(TpM, X) -> Y`.
"""
get_normal_linear_operator(M::AbstractManifold, lsmo::AbstractLinearSurrogateObjective, p, X = nothing)

function get_normal_vector_field end
"""
    get_normal_vector_field(M::AbstractManifold, lsmo::AbstractLinearSurrogateObjective, p)
    get_normal_vector_field!(M::AbstractManifold, y, lsmo::AbstractLinearSurrogateObjective, p)

Return the normal vector ``$(_tex(:Cal, "L"))^*(y)`` of the linear surrogate model `lsmo` at the point ``p ∈ M``.
"""
get_normal_vector_field(M::AbstractManifold, lsmo::AbstractLinearSurrogateObjective, p)
