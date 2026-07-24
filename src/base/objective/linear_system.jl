_doc_CR_cost = """
```math
f(X) = $(_tex(:frac, 1, 2)) $(_tex(:norm, _tex(:Cal, "A") * "[X] + b"; index = "p"))^2,$(_tex(:qquad)) X ∈ $(_math(:TangentSpace)),
```
"""

"""
    AbstractSymmetricLinearSystemObjective{E<:AbstractEvaluationType} <: AbstractManifoldObjective{E}

Model the objective

$(_doc_CR_cost)

defined on the tangent space ``$(_math(:TangentSpace))`` at ``p`` on the manifold ``$(_math(:Manifold))``.

In other words this is an objective to solve ``$(_tex(:Cal, "A"))[X] = -b(p)``
for some linear symmetric operator ``$(_tex(:Cal, "A"))`` and a vector function ``b``

Concrete subtypes of this type should/could implement

* [`get_linear_operator`](@ref) to evaluate ``$(_tex(:Cal, "A"))[X]```
* [`get_vector_field`](@ref) to evaluate ``b`` at ``p``.

Then the following functions are available directly

* [`get_cost`](@ref)`(TpM, aslso, X)` to compute/evaluate the objective
* [`get_gradient`](@ref)`(TpM, aslso, X)` to compute/evaluate the objectives gradient at `X`
* [`get_linear_operator`](@ref)`(TpM, aslso, X)` to compute/evaluate the linear operator ``$(_tex(:Cal, "A"))`` at `X`
"""
abstract type AbstractSymmetricLinearSystemObjective <: AbstractManifoldObjective end


@doc """
    get_cost(TpM::TangentSpace, aslso::SymmetricLinearSystemObjective, X)

evaluate the cost

$(_doc_CR_cost)

at `X`.
"""
function get_cost(
        TpM::TangentSpace, aslso::AbstractSymmetricLinearSystemObjective, X
    )
    M = base_manifold(TpM)
    p = base_point(TpM)
    return 0.5 * norm(M, p, get_linear_operator(TpM, aslso, p, X) + get_vector_field(TpM, aslso, p))^2
end
@doc """
    get_gradient(TpM::TangentSpace, aslso::AbstractSymmetricLinearSystemObjective, X)
    get_gradient!(TpM::TangentSpace, Y, aslso::AbstractSymmetricLinearSystemObjective, X)

evaluate the gradient of

$(_doc_CR_cost)

Which is ``$(_tex(:grad)) f(X) = $(_tex(:Cal, "A"))[X]+b``.
This can be computed in-place of `Y`.
"""
function get_gradient(TpM::TangentSpace, aslso::AbstractSymmetricLinearSystemObjective, X)
    M = base_manifold(TpM); p = base_point(TpM)
    return get_linear_operator(M, aslso, p, X) + get_vector_field(M, aslso, p)
end
function get_gradient!(
        TpM::TangentSpace, Y, aslso::AbstractSymmetricLinearSystemObjective, X
    )
    M = base_manifold(TpM)
    p = base_point(TpM)
    W = copy(M, p, Y)
    get_linear_operator!(M, W, aslso, p, X)
    get_vector_field!(M, Y, aslso, p)
    Y .+= W
    return Y
end
@doc """
    get_Hessian(TpM::TangentSpace, aslso::AbstractSymmetricLinearSystemObjective, X, V)
    get_Hessian!(TpM::TangentSpace, W, aslso::AbstractSymmetricLinearSystemObjective, X, V)

evaluate the Hessian of

$(_doc_CR_cost)

Which is ``$(_tex(:Hess)) f(X)[Y] = $(_tex(:Cal, "A"))[V]``. This can be computed in-place of `W`.
Internally this (just) calls the [`get_linear_operator`](@ref) function.
"""
function get_hessian(TpM::TangentSpace, aslso::AbstractSymmetricLinearSystemObjective, X, V)
    M = base_manifold(TpM)
    p = base_point(TpM)
    return get_linear_operator(M, aslso, p, V)
end
function get_hessian!(TpM::TangentSpace, W, aslso::AbstractSymmetricLinearSystemObjective, X, V)
    M = base_manifold(TpM)
    p = base_point(TpM)
    get_linear_operator!(M, W, aslso, p, V)
    return W
end
