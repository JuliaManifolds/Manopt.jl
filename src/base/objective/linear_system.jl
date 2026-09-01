_doc_CR_cost = """
```math
f(X) = $(_tex(:frac, 1, 2)) ⟨X, $(_tex(:Cal, "A"))[X]⟩_p + ⟨b, X⟩_p,$(_tex(:qquad)) X ∈ $(_math(:TangentSpace)),
```
"""

"""
    AbstractSymmetricLinearSystemObjective <: AbstractManifoldObjective

Model the objective

$(_doc_CR_cost)

defined on the tangent space ``$(_math(:TangentSpace))`` at ``p`` on the manifold ``$(_math(:Manifold))``.

In other words this is an objective to solve ``$(_tex(:Cal, "A"))[X] = -b(p)``
for some linear symmetric operator ``$(_tex(:Cal, "A"))`` and a vector function ``b``.

Concrete subtypes of this type should/could implement

* [`get_linear_operator`](@ref) to evaluate ``$(_tex(:Cal, "A"))[X]``
* [`get_vector_field`](@ref) to evaluate ``b`` at ``p``.

Then the following functions are available directly

* [`get_cost`](@ref)`(TpM, aslso, X)` to compute/evaluate the objective
* [`get_gradient`](@ref)`(TpM, aslso, X)` to compute/evaluate the objective's gradient at `X`
* [`get_linear_operator`](@ref)`(M, aslso, p, X)` to compute/evaluate the linear operator ``$(_tex(:Cal, "A"))`` at `X`
"""
abstract type AbstractSymmetricLinearSystemObjective <: AbstractManifoldObjective end


@doc """
    get_cost(TpM::TangentSpace, aslso::AbstractSymmetricLinearSystemObjective, X)

Evaluate the cost

$(_doc_CR_cost)

at `X`.
"""
function get_cost(
        TpM::TangentSpace, aslso::AbstractSymmetricLinearSystemObjective, X
    )
    M = base_manifold(TpM)
    p = base_point(TpM)
    W = 0.5 * get_linear_operator(M, aslso, p, X) + get_vector_field(M, aslso, p)
    return real(inner(M, p, X, W))
end
@doc """
    get_gradient(TpM::TangentSpace, aslso::AbstractSymmetricLinearSystemObjective, X)
    get_gradient!(TpM::TangentSpace, Y, aslso::AbstractSymmetricLinearSystemObjective, X)

Evaluate the gradient of

$(_doc_CR_cost)

This gradient is given by ``$(_tex(:grad)) f(X) = $(_tex(:Cal, "A"))[X]+b``.
It can be computed in-place of `Y`.
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
    get_hessian(TpM::TangentSpace, aslso::AbstractSymmetricLinearSystemObjective, X, V)
    get_hessian!(TpM::TangentSpace, W, aslso::AbstractSymmetricLinearSystemObjective, X, V)

Evaluate the Hessian of

$(_doc_CR_cost)

This Hessian is given by ``$(_tex(:Hess)) f(X)[V] = $(_tex(:Cal, "A"))[V]``. It can be computed in-place of `W`.
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
