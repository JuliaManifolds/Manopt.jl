@doc raw"""
    SymmetricLinearSystemObjective{E<:AbstractEvaluationType,TA,T} <: AbstractManifoldObjective{E}

Model the objective

```math
f(X) = \frac{1}{2} \lVert \mathcal A[X] + b \rVert_p^2,\qquad X ∈ T_p\mathcal M,
```

defined on the tangent space ``T_p\mathcal M`` at ``p`` on the manifold ``\mathcal M``.

In other words this is an objective to solve ``\mathcal A(p)[X] = -b(p)``
for some linear symmetric operator and a vector function.
Note the minus on the right hand side, which makes this objective especially taylored
for (iteratively) solving Newton-like equations.

# Fields

* `A!!`: a symmetric, linear operator on the tangent space
* `b!!`: a gradient function

where ``A!!`` can work as an allocating operator `(M, p, X) -> Y` or an in-place one `(M, Y, p, X) -> Y`,
and similarly ``b`` can either be a function `(M, p) -> X` or `(M, X, p) -> X`

# Constructor

    SymmetricLinearSystemObjective(A, b; evaluation=AllocatingEvaluation())

Generate the objective specifying whether the two parts work allocating or in-place.
"""
mutable struct SymmetricLinearSystemObjective{E<:AbstractEvaluationType,TA,T} <:
               AbstractManifoldObjective{E}
    A!!::TA
    b!!::T
end

function SymmetricLinearSystemObjective(
    A::TA, b::T; evaluation::E=AllocatingEvaluation(), kwargs...
) where {TA,T,E<:AbstractEvaluationType}
    return SymmetricLinearSystemObjective{E,TA,T}(A, b)
end

function set_manopt_parameter!(slso::SymmetricLinearSystemObjective, symbol::Symbol, value)
    set_manopt_parameter!(slso.A!!, symbol, value)
    set_manopt_parameter!(slso.b!!, symbol, value)
    return slso
end

function get_cost(
    TpM::TangentSpace, slso::SymmetricLinearSystemObjective{AllocatingEvaluation}, X
)
    M = base_manifold(TpM)
    p = base_point(TpM)
    b = slso.b!!(M, p)
    return 0.5 * norm(M, p, slso.A!!(M, p, X) + slso.b!!(M, p))^2
end
function get_cost(
    TpM::TangentSpace, slso::SymmetricLinearSystemObjective{InplaceEvaluation}, X
)
    M = base_manifold(TpM)
    p = base_point(TpM)
    Y = zero_vector(M, p)
    W = copy(M, p, Y)
    slso.b!!(M, Y, p)
    slso.A!!(M, W, p, X)
    return 0.5 * norm(M, p, W + Y)^2
end

function get_b(
    TpM::TangentSpace, slso::SymmetricLinearSystemObjective{AllocatingEvaluation}, X
)
    return slso.b!!(base_manifold(TpM), base_point(TpM))
end
function get_b(
    TpM::TangentSpace, slso::SymmetricLinearSystemObjective{InplaceEvaluation}, X
)
    M = base_manifold(TpM)
    p = base_point(TpM)
    Y = zero_vector(M, p)
    return slso.b!!(M, Y, p)
end

function get_gradient(TpM::TangentSpace, slso::SymmetricLinearSystemObjective, X)
    p = base_point(TpM)
    return get_hessian(TpM, slso, p, X) + get_b(TpM, slso, X)
end
function get_gradient!(
    TpM::TangentSpace, Y, slso::SymmetricLinearSystemObjective{AllocatingEvaluation}, X
)
    M = base_manifold(TpM)
    p = base_point(TpM)
    # Evaluate A[X] + b
    Y .= slso.A!!(M, p, X) + slso.b!!(M, p)
    return Y
end
function get_gradient!(
    TpM::TangentSpace, Y, slso::SymmetricLinearSystemObjective{InplaceEvaluation}, X
)
    M = base_manifold(TpM)
    p = base_point(TpM)
    W = copy(M, p, Y)
    slso.b!!(M, Y, p)
    slso.A!!(M, W, p, X)
    Y .+= W
    return Y
end

# evaluate Hessian: ∇²Q(X) = A[X]
function get_hessian(
    TpM::TangentSpace, slso::SymmetricLinearSystemObjective{AllocatingEvaluation}, X, V
)
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
    copyto!(M, W, p, slso.A!!(M, p, V))
    return W
end
function get_hessian!(
    TpM::TangentSpace, W, slso::SymmetricLinearSystemObjective{InplaceEvaluation}, X, V
)
    return slso.A!!(base_manifold(TpM), W, base_point(TpM), V)
end
