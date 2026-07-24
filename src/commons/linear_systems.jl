@doc """
    SymmetricLinearSystemObjective{TA,T} <: AbstractSymmetricLinearSystemObjective{E}

Model the objective

$(_doc_CR_cost)

defined on the tangent space ``$(_math(:TangentSpace))`` at ``p`` on the manifold ``$(_math(:Manifold))``.

In other words this is an objective to solve ``$(_tex(:Cal, "A")) = -b(p)``
for some linear symmetric operator and a vector function.
Note the minus on the right hand side, which makes this objective especially tailored
for (iteratively) solving Newton-like equations.

# Fields

* `A!!`: a symmetric, linear operator on the tangent space, see [`get_linear_operator`](@ref)
* `b!!`: a tangent vector function, see [`get_vector_field`](@ref)

where `A!!` is implemented as in-place operator `(M, Y, p, X) -> Y`,
and similarly `b!!` is a function `(M, X, p) -> X` implemented to work in-place of `X`.

# Constructor

    SymmetricLinearSystemObjective(A, b; evaluation=AllocatingEvaluation())

Generate the objective specifying whether the two parts work allocating or in-place.
"""
mutable struct SymmetricLinearSystemObjective{TA, T} <: AbstractSymmetricLinearSystemObjective
    A!!::TA
    b!!::T
end

function set_parameter!(slso::SymmetricLinearSystemObjective, symbol::Symbol, value)
    set_parameter!(slso.A!!, symbol, value)
    set_parameter!(slso.b!!, symbol, value)
    return slso
end

function Base.show(io::IO, slso::SymmetricLinearSystemObjective)
    print(io, "SymmetricLinearSystemObjective(")
    print(io, slso.A!!); print(io, ", "); print(io, slso.b!!)
    return print(io, ")")
end

function status_summary(slso::SymmetricLinearSystemObjective; context::Symbol = :default)
    _is_inline(context) && (return repr(slso))
    return """
    An objective modelling a symmetric linear system Ax=b, i.e. with a symmetric matrix A
    implemented as a function `(M, p, X) -> Y` performing the matrix vector multiplication in the tangent space,
    and a function `b(M,p)` returning the vector on the right hand side in the current tangent space.

    # Fields
    * A: $(slso.A!!)
    * b: $(slso.b!!)"""
end
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
