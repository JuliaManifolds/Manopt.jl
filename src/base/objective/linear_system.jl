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

In other words this is an objective to solve ``$(_tex(:Cal, "A")) = -b(p)``
for some linear symmetric operator ``$(_tex(:Cal, "A"))`` and a vector function ``b``

Concrete subtypes of this type should/could implement

* [`get_cost`](@ref)`(TpM, aslso, X)` to compute/evaluate the objective
* [`get_gradient`](@ref)`(TpM, aslso, X)` to compute/evaluate the objectives gradient at `X`
* [`get_linear_operator`](@ref)`(TpM, aslso, X)` to compute/evaluate the linear operator ``$(_tex(:Cal, "A"))`` at `X`
"""
abstract type AbstractSymmetricLinearSystemObjective <: AbstractManifoldObjective end
