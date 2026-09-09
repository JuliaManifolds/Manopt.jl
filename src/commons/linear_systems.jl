@doc """
    SymmetricLinearSystemObjective{TA,T} <: AbstractSymmetricLinearSystemObjective

Model the objective

$(_doc_CR_cost)

defined on the tangent space ``$(_math(:TangentSpace))`` at ``p`` on the manifold ``$(_math(:Manifold))``.

In other words this is an objective to solve ``$(_tex(:Cal, "A"))[X] = -b(p)``
for some linear symmetric operator ``$(_tex(:Cal, "A"))`` and a vector function ``b``.
Note the minus on the right hand side, which makes this objective especially tailored
for (iteratively) solving Newton-like equations.

# Fields

* `A!`: a symmetric, linear operator on the tangent space, see [`get_linear_operator`](@ref)
* `b!`: a tangent vector function, see [`get_vector_field`](@ref)

where `A!` is implemented as in-place operator `(M, Y, p, X) -> Y`,
and similarly `b!` is a function `(M, X, p) -> X` implemented to work in-place of `X`.

# Constructor

    SymmetricLinearSystemObjective(A, b; evaluation=AllocatingEvaluation(), p = missing)

Generate the objective specifying whether the two parts work allocating or in-place.

## Keyword Arguments

$(_kwargs(:evaluation))
* `p = missing` provide a point to automatically ensure the functions of the objective “act” on mutating variables.
"""
mutable struct SymmetricLinearSystemObjective{TA, T} <: AbstractSymmetricLinearSystemObjective
    A!::TA
    b!::T
    function SymmetricLinearSystemObjective(A, b; evaluation::AbstractEvaluationType = AllocatingEvaluation(), p = missing)
        A_ = maybe_wrap_function(A, p, evaluation; result = :TangentVector)
        b_ = maybe_wrap_function(b, p, evaluation; result = :TangentVector)
        return new{typeof(A_), typeof(b_)}(A_, b_)
    end
end
function set_parameter!(slso::SymmetricLinearSystemObjective, e::Val, value)
    set_parameter!(slso.A!, e, value)
    set_parameter!(slso.b!, e, value)
    return slso
end

function Base.show(io::IO, slso::SymmetricLinearSystemObjective)
    print(io, "SymmetricLinearSystemObjective(")
    print(io, slso.A!); print(io, ", "); print(io, slso.b!)
    return print(io, ")")
end

function status_summary(slso::SymmetricLinearSystemObjective; context::Symbol = :default)
    _is_inline(context) && (return repr(slso))
    return """
    An objective modeling a symmetric linear system A[X] = -b, i.e. with a symmetric operator A
    implemented as a function `(M, Y, p, X) -> Y` performing the operator application in the tangent space,
    and a function `(M, X, p) -> X` returning the vector on the right hand side in the current tangent space.

    ## Fields
    * A: $(slso.A!)
    * b: $(slso.b!)"""
end

@doc """
    Y = get_linear_operator(M::AbstractManifold, slso::SymmetricLinearSystemObjective, p, X)
    get_linear_operator!(M::AbstractManifold, Y, slso::SymmetricLinearSystemObjective, p, X)

Evaluate the linear operator ``Y = $(_tex(:Cal, "A"))[X]`` from the [`SymmetricLinearSystemObjective`](@ref)
defined on the tangent space at `p` at the tangent vector `X`.

This can be evaluated in-place of `Y`.
"""
function get_linear_operator(
        M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p, X
    )
    return get_linear_operator(M, get_objective(admo, false), p, X)
end
function get_linear_operator(M::AbstractManifold, slso::SymmetricLinearSystemObjective, p, X)
    Y = copy(M, p, X)
    return slso.A!(M, Y, p, X)
end
function get_linear_operator!(
        M::AbstractManifold, Y, admo::AbstractDecoratedManifoldObjective, p, X
    )
    return get_linear_operator!(M, Y, get_objective(admo, false), p, X)
end
function get_linear_operator!(M::AbstractManifold, Y, slso::SymmetricLinearSystemObjective, p, X)
    return slso.A!(M, Y, p, X)
end

_doc_get_vector_field_slso = """
    get_vector_field(M::AbstractManifold, slso::SymmetricLinearSystemObjective, p)
    get_vector_field!(M::AbstractManifold, Y, slso::SymmetricLinearSystemObjective, p)
    get_vector_field(TpM::TangentSpace, slso::SymmetricLinearSystemObjective)
    get_vector_field!(TpM::TangentSpace, Y, slso::SymmetricLinearSystemObjective)

Evaluate the stored value for computing the right hand side ``b`` in ``$(_tex(:Cal, "A"))[X] = -b``,
either providing a tangent space or a manifold and a point.

This can be evaluated in-place of `Y`.
"""

function get_vector_field(M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p)
    return get_vector_field(M, get_objective(admo, false), p)
end
@doc "$(_doc_get_vector_field_slso)"
function get_vector_field(M::AbstractManifold, slso::SymmetricLinearSystemObjective, p)
    Y = zero_vector(M, p)
    return slso.b!(M, Y, p)
end
function get_vector_field!(M::AbstractManifold, Y, admo::AbstractDecoratedManifoldObjective, p)
    return get_vector_field!(M, Y, get_objective(admo, false), p)
end
function get_vector_field!(M::AbstractManifold, Y, slso::SymmetricLinearSystemObjective, p)
    return slso.b!(M, Y, p)
end
# Also on TpM – shortcuts
function get_vector_field(TpM::TangentSpace, admo::AbstractDecoratedManifoldObjective)
    return get_vector_field(TpM, get_objective(admo, false))
end
@doc "$(_doc_get_vector_field_slso)"
function get_vector_field(TpM::TangentSpace, slso::SymmetricLinearSystemObjective)
    M = base_manifold(TpM)
    p = base_point(TpM)
    Y = zero_vector(M, p)
    return slso.b!(M, Y, p)
end
function get_vector_field!(TpM::TangentSpace, Y, admo::AbstractDecoratedManifoldObjective)
    return get_vector_field!(TpM, Y, get_objective(admo, false))
end
@doc "$(_doc_get_vector_field_slso)"
function get_vector_field!(TpM::TangentSpace, Y, slso::SymmetricLinearSystemObjective)
    M = base_manifold(TpM)
    p = base_point(TpM)
    return slso.b!(M, Y, p)
end
