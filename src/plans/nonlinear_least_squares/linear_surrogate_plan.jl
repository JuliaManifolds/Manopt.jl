#
#
# A linear surrogate model for use with certain objectives.
"""
    AbstractLinearSurrogateObjective{E <: AbstractEvaluationType, O <: AbstractManifoldObjective{E}} <: AbstractManifoldObjective{E}

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

See also [`NormalEquations`](@ref) for the corresponding normal equations.
"""
abstract type AbstractLinearSurrogateObjective{E <: AbstractEvaluationType, O <: AbstractManifoldObjective{E}} <: AbstractManifoldObjective{E} end

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

#
# A wrapper to model the linear operator and vector field of the normal equations of the surrogate
"""
    NormalEquationsObjective{E <: AbstractEvaluationType, O<: AbstractLinearSurrogateObjective{E}} <: AbstractSymmetricLinearSystemObjective{E}

A [`AbstractLinearSurrogateObjective`](@ref) might be overdetermined, and it usually is overdetermined,
e.g. for the case of the [`LevenbergMarquardt`](@ref) algorithm.
For this case, one considers the [normal equations](https://en.wikipedia.org/wiki/Non-linear_least_squares).

This wrapper provides the same three functions as the wrapped surrogate

* [`get_linear_operator`](@ref) to compute/evaluate the linear operator ``$(_tex(:Cal, "L"))``
* [`get_vector_field`](@ref) to compute/evaluate the vector ``y``
* [`get_objective`](@ref) to provide access to the underlying surrogate

so that we obtain a symmetric linear system of equations, that can be
* solved with an iterative method like [`cojugate_gradient`](@ref) or [`conjugate_residual`](@ref)
* solved as a linear system in a basis of the corresponding tangent space.
"""
struct NormalEquationsObjective{E <: AbstractEvaluationType, O <: AbstractLinearSurrogateObjective{E}} <: AbstractSymmetricLinearSystemObjective{E}
    objective::O
end

function show(io::IO, neo::NormalEquationsObjective{E}) where {E}
    print(io, "NormalEquationsObjective(")
    print(io, neo.objective)
    return print(io, ")")
end

get_objective(slsmo::NormalEquationsObjective) = slsmo.objective

# set parameter just passes down to the inner objective
for NT in [Val, Val{:Cost}, Val{:Gradient}, Val{:SubGradient}]
    @eval function set_parameter!(neo::NormalEquationsObjective, name::$NT, value)
        set_parameter!(neo.objective, name, value)
        return neo
    end
end
function set_parameter!(neo::NormalEquationsObjective, name::Symbol, value)
    set_parameter!(neo.objective, name, value)
    return neo
end
