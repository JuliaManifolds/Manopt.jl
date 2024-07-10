@doc raw"""
    SymmetricLinearSystemObjective{E<:AbstractEvaluationType,TA,T} <: AbstractManifoldObjective{E}

Model the objective

```math
f(X) = \frac{1}{2} \lVert \mathcal A[X] - b \rVert_p^2,\qquad X ∈ T_p\mathcal M,
```
defined on the tangent space ``T_p\mathcal M`` at ``p`` on the manifold ``\mathcal M``.

# Fields

* `A!!`: a symmetric, linear operator on the tangent space
* `b`: a gradient function

where ``A!!`` can work as an allocating operator `(M, p, X) -> Y` or an in-place one `(M, Y, p, X) -> Y`,
and similarly ``b`` can either be a function `(M, p) -> X` or `(M, X, p) -> X`

# Constructor

    SymmetricLinearSystemObjective(A, b; evaluation=AllocatingEvaluation())

"""
mutable struct SymmetricLinearSystemObjective{E<:AbstractEvaluationType,TA,T} <:
       AbstractManifoldObjective{E}
    A!!::TA
    b::T
end

function SymmetricLinearSystemObjective(
    A::TA,
    b::T;
    evaluation::E = AllocatingEvaluation()
    ) where {TA,T, E<:AbstractEvaluationType}
    return SymmetricLinearSystemObjective{E,TA,T}(A, b)
end

function set_manopt_parameter!(slso::SymmetricLinearSystemObjective, symbol::Symbol, value)
    set_manopt_parameter!(slso.A!!, symbol, value)
    return slso
end

function set_manopt_parameter!(slso::SymmetricLinearSystemObjective, ::Val{:b}, b)
    slso.b = b
    return slso
end

function get_cost(TpM::TangentSpace, slso::SymmetricLinearSystemObjective, X)
    M = base_manifold(TpM)
    p = base_point(TpM)

    return 0.5 * inner(M, p, X, get_hessian(TpM, slso, p, X) -  inner(M, p, slso.b(M, p), X)
end

function get_gradient(TpM::TangentSpace, slso::SymmetricLinearSystemObjective, X)
    # Evaluate A[X] - b
    return get_hessian(TpM, slso, base_point(TpM), X) - slso.b
end

function get_gradient!(TpM::TangentSpace, Y, slso::SymmetricLinearSystemObjective, X)
    get_hessian!(TpM, Y, base_point(TpM), X) #Evaluate A[X] in-place of Y
    Y -= slso.b
    return Y
end

# evaluate Hessian: ∇²Q(X) = A[X]
function get_hessian(TpM::TangentSpace, slso::SymmetricLinearSystemObjective{AllocatingEvaluation}, X, V)
    return slso.A!!(base_manifold(TpM), base_point(TpM), V)
end
function get_hessian(
    TpM::TangentSpace, slso::SymmetricLinearSystemObjective{InplaceEvaluation}, X, V
)
    M = base_manifold(TpM)
    p = base_point(TpM)
    W = copy(M, p, V)
    slso.A!!(M, W, p, V)
    return W
end
function get_hessian!(
    TpM::TangentSpace, W, slso::SymmetricLinearSystemObjective{AllocatingEvaluation}, X, V
)
    M = base_manifold(TpM)
    p = base_point(TpM)
    copyto!(M, p, W, slso.A!!(M, p, V))
    return W
end
function get_hessian(
    TpM::TangentSpace, W, slso::SymmetricLinearSystemObjective{InplaceEvaluation}, X, V
)
    return slso.A!!(base_manifold(TpM), W, base_point(TpM), V)
end