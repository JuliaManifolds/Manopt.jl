#
#
# A linear surrogate model for use with certain objectives.

"""
    AbstractLinearSurrogateObjective{E <: AbstractEvaluationType, O <: AbstractManifoldObjective{E}} <: AbstractManifoldObjective{E}

Provide a linear surrogate model for the given [`AbstractManifoldObjective`](@ref) `O` of the form

```math
σ_p(X) = $(_tex(:frac, "1", "2"))$(_tex(:norm, _tex(:Cal, "L") * "(X) + y"))^2
  + $(_tex(:frac, "λ", "2"))$(_tex(:norm, "X"; index = "p"))^2,
  $(_tex(:qquad))for X ∈ $(_math(:TangentSpace)), λ ≥ 0,
```

where ``$(_tex(:Cal, "L"))`` is a linear operator on the tangent space at a point ``p ∈ M``
that maps into some vector space ``V`` and ``y ∈ V`` is a fixed vector in that space
and ``$(_tex(:norm, "?"))`` is a norm on ``V``.

Both ``$(_tex(:Cal, "L"))`` and ``y`` are derived from the objective `O` and usually depend
on the base point ``p ∈ M``.

Besides the usual methods defined for [`AbstractManifoldObjective`](@ref) that may be implemented
like [`get_cost`](@ref) and [`get_gradient`](@ref), the following methods should be implemented
for a concrete subtype of `AbstractLinearSurrogateObjective`

* [`linear_operator`](@ref) to compute/evaluate the linear operator ``$(_tex(:Cal, "L"))``
* [`vector_field`](@ref) to compute/evaluate the vector ``y``
* [`linear_normal_operator`](@ref) to compute/evaluate the normal operator ``$(_tex(:Cal, "L"))^* $(_tex(:Cal, "L"))``
* [`normal_vector_field`](@ref) to compute/evaluate the normal vector ``$(_tex(:Cal, "L"))^*(y)``
* [`get_objective`](@ref) to provide access to the underlying objective `O`
"""
abstract type AbstractLinearSurrogateObjective{E <: AbstractEvaluationType, O <: AbstractManifoldObjective{E}} <: AbstractManifoldObjective{E} end

"""
    get_objective(lso::AbstractLinearSurrogateObjective)

Return the objective `O` associated with the linear surrogate model `lso`.
"""
get_objective(lso::AbstractLinearSurrogateObjective)

function linear_operator end
"""
    linear_operator(M::AbstractManifold, lsmo::AbstractLinearSurrogateObjective, p)
    linear_operator(M::AbstractManifold, lsmo::AbstractLinearSurrogateObjective, p, B::AbstractBasis)
    linear_operator(M::AbstractManifold, lsmo::AbstractLinearSurrogateObjective, p, X)
    linear_operator!(M::AbstractManifold, L, lsmo::AbstractLinearSurrogateObjective, p, B::AbstractBasis)
    linear_operator!(M::AbstractManifold, Y, lsmo::AbstractLinearSurrogateObjective, p, X)

Return/Evaluate the linear operator ``$(_tex(:Cal, "L"))`` of the linear surrogate model `lsmo` at the point ``p ∈ M``.

If a tangent vector `X` is provided, evaluate ``$(_tex(:Cal, "L"))(X)``.
If a basis `B` is provided, return the matrix representation of ``$(_tex(:Cal, "L"))`` with respect to that basis.
Otherwise return the operator as a function `(TpM, X) -> Y`.
"""
linear_operator(M::AbstractManifold, lsmo::AbstractLinearSurrogateObjective, p, X = nothing)

function vector_field end
"""
    vector_field(M::AbstractManifold, lsmo::AbstractLinearSurrogateObjective, p)
    vector_field!(M::AbstractManifold, y, lsmo::AbstractLinearSurrogateObjective, p)

Return the vector `y` of the linear surrogate model `lsmo` at the point ``p ∈ M``.
"""
vector_field(M::AbstractManifold, lsmo::AbstractLinearSurrogateObjective, p)

function linear_normal_operator end
"""
    linear_normal_operator(M::AbstractManifold, lsmo::AbstractLinearSurrogateObjective, p)
    linear_normal_operator(M::AbstractManifold, lsmo::AbstractLinearSurrogateObjective, p, B::AbstractBasis)
    linear_normal_operator(M::AbstractManifold, lsmo::AbstractLinearSurrogateObjective, p, X)
    linear_normal_operator!(M::AbstractManifold, N, lsmo::AbstractLinearSurrogateObjective, p, B::AbstractBasis)
    linear_normal_operator!(M::AbstractManifold, Y, lsmo::AbstractLinearSurrogateObjective, p, X)

Return/Evaluate the normal operator ``$(_tex(:Cal, "L"))^* $(_tex(:Cal, "L"))`` of the linear surrogate model `lsmo` at the point ``p ∈ M``.

If a tangent vector `X` is provided, evaluate ``$(_tex(:Cal, "L"))^* $(_tex(:Cal, "L"))(X)``.
If a basis `B` is provided, return the matrix representation of ``$(_tex(:Cal, "L"))^* $(_tex(:Cal, "L"))`` with respect to that basis.
Otherwise return the operator as a function `(TpM, X) -> Y`.
"""
linear_normal_operator(M::AbstractManifold, lsmo::AbstractLinearSurrogateObjective, p, X = nothing)

function normal_vector_field end
"""
    normal_vector_field(M::AbstractManifold, lsmo::AbstractLinearSurrogateObjective, p)
    normal_vector_field!(M::AbstractManifold, y, lsmo::AbstractLinearSurrogateObjective, p)

Return the normal vector ``$(_tex(:Cal, "L"))^*(y)`` of the linear surrogate model `lsmo` at the point ``p ∈ M``.
"""
normal_vector_field(M::AbstractManifold, lsmo::AbstractLinearSurrogateObjective, p)

#
#
# Wrapper to symmetrix linear systems where the normal eq becomes the linear one
"""
    SymmetricLinearSystem{E <: AbstractEvaluationType, O<: AbstractLinearSurrogateObjective{E}} <: AbstractSymmetricLinearSystemObjective{E}

A wrapper type to turn an
[`AbstractLinearSurrogateObjective`](@ref) `O` into a
[`AbstractSymmetricLinearSystemObjective`](@ref) by interpreting the normal equations of `O`
as the symmetric linear system.
"""
struct SymmetricLinearSystem{E <: AbstractEvaluationType, O <: AbstractLinearSurrogateObjective{E}} <: AbstractSymmetricLinearSystemObjective{E}
    objective::O
end

get_objective(slsmo::SymmetricLinearSystem) = slsmo.objective

# set parameter just passes down to the inner objective
set_parameter!(slsmo::SymmetricLinearSystem, name::Symbol, value) = set_parameter!(get_objective(slsmo), name, value)

function linear_operator(M::AbstractManifold, slso::SymmetricLinearSystem, p, X)
    return linear_normal_operator(M, slso.objective, p, X)
end
function linear_operator!(M::AbstractManifold, Y, slso::SymmetricLinearSystem, p, X)
    return linear_normal_operator!(M, Y, slso.objective, p, X)
end
function vector_field(M::AbstractManifold, slso::SymmetricLinearSystem, p)
    return normal_vector_field(M, slso.objective, p)
end
function vector_field!(M::AbstractManifold, Y, slso::SymmetricLinearSystem, p)
    return normal_vector_field!(M, Y, slso.objective, p)
end
