@doc """
    TwoManifoldProblem{
        MT<:AbstractManifold,NT<:AbstractManifold,O<:AbstractManifoldObjective
    } <: AbstractManoptProblem{MT}

An abstract type for primal-dual-based problems.
"""
struct TwoManifoldProblem{
        MT <: AbstractManifold, NT <: AbstractManifold, S <: AbstractManifoldObjective,
    } <: AbstractManoptProblem{MT}
    first_manifold::MT
    second_manifold::NT
    objective::S
end
get_manifold(tmp::TwoManifoldProblem) = get_manifold(tmp, 1)
get_manifold(tmp::TwoManifoldProblem, i) = _get_manifold(tmp, Val(i))
_get_manifold(tmp::TwoManifoldProblem, ::Val{1}) = tmp.first_manifold
_get_manifold(tmp::TwoManifoldProblem, ::Val{2}) = tmp.second_manifold

get_objective(tmo::TwoManifoldProblem) = tmo.objective

function show(io::IO, tmp::TwoManifoldProblem)
    print(io, "TwoManifoldProblem("); show(io, tmp.first_manifold)
    print(io, ", "); show(io, tmp.second_manifold)
    print(io, ", "); show(io, tmp.objective)
    return print(io, ")")
end
function status_summary(tmp::TwoManifoldProblem; inline = false)
    inline && return "An optimization problem to minimize $(tmp.objective) using a primal manifold $(tmp.first_manifold) and a dual manifold $(tmp.second_manifold)."
    return """
    An optimization problem for Manopt.jl requiring a primal and a dual manifold

    ## Manifolds
    * $(replace(repr(tmp.first_manifold), "\n#" => "\n##", "\n" => "\n$(_MANOPT_INDENT)"))
    * $(replace(repr(tmp.second_manifold), "\n#" => "\n##", "\n" => "\n$(_MANOPT_INDENT)"))

    ## Objective
    $(_MANOPT_INDENT)$(replace(status_summary(tmp.objective, inline = inline), "\n#" => "\n##", "\n" => "\n$(_MANOPT_INDENT)"))"""
end

@doc """
    AbstractPrimalDualManifoldObjective{E<:AbstractEvaluationType,C,P} <: AbstractManifoldCostObjective{E,C}

A common abstract super type for objectives that consider primal-dual problems.
"""
abstract type AbstractPrimalDualManifoldObjective{E <: AbstractEvaluationType, C, P} <:
AbstractManifoldCostObjective{E, C} end

@doc """
    PrimalDualManifoldObjective{T<:AbstractEvaluationType} <: AbstractPrimalDualManifoldObjective{T}

Describes an Objective linearized or exact Chambolle-Pock algorithm, cf. [BergmannHerzogSilvaLouzeiroTenbrinckVidalNunez:2021](@cite), [ChambollePock:2011](@cite)

# Fields

All fields with `!!` can either be in-place or allocating functions, which should be set
depending on the `evaluation=` keyword in the constructor and stored in `T <: AbstractEvaluationType`.

* `cost`:                          ``F + G(Λ(⋅))`` to evaluate interim cost function values
* `linearized_forward_operator!!`: linearized operator for the forward operation in the algorithm ``DΛ``
* `linearized_adjoint_operator!!`: the adjoint differential ``(DΛ)^* : $(_tex(:Cal, "N")) → T$(_math(:Manifold))nifold))nifold))``
* `prox_f!!`:                      the proximal map belonging to ``f``
* `prox_G_dual!!`:                 the proximal map belonging to ``g_n^*``
* `Λ!!`:                           the  forward operator (if given) ``Λ: $(_math(:Manifold))) → $(_tex(:Cal, "N"))``

Either the linearized operator ``DΛ`` or ``Λ`` are required usually.

# Constructor

    PrimalDualManifoldObjective(cost, prox_f, prox_G_dual, adjoint_linearized_operator;
        linearized_forward_operator::Union{Function,Missing}=missing,
        Λ::Union{Function,Missing}=missing,
        evaluation::AbstractEvaluationType=AllocatingEvaluation()
    )

The last optional argument can be used to provide the 4 or 5 functions as allocating or
mutating (in place computation) ones.
Note that the first argument is always the manifold under consideration, the mutated one is
the second.
"""
mutable struct PrimalDualManifoldObjective{
        T <: AbstractEvaluationType, TC, TP, TDP, LFO, ALFO, L,
    } <: AbstractPrimalDualManifoldObjective{T, TC, TP}
    cost::TC
    prox_f!!::TP
    prox_g_dual!!::TDP
    linearized_forward_operator!!::LFO
    adjoint_linearized_operator!!::ALFO
    Λ!!::L
end
function PrimalDualManifoldObjective(
        cost::C,
        prox_f::F,
        prox_g_dual::G,
        adjoint_linearized_operator::A;
        linearized_forward_operator::Union{Function, Missing} = missing,
        Λ::Union{Function, Missing} = missing,
        evaluation::E = AllocatingEvaluation(),
    ) where {E <: AbstractEvaluationType, C, F, G, A}
    return PrimalDualManifoldObjective{
        E, C, F, G, typeof(linearized_forward_operator), A, typeof(Λ),
    }(
        cost, prox_f, prox_g_dual, linearized_forward_operator, adjoint_linearized_operator, Λ,
    )
end

@doc """
    q = get_primal_prox(M::AbstractManifold, p::AbstractPrimalDualManifoldObjective, σ, p)
    get_primal_prox!(M::AbstractManifold, p::AbstractPrimalDualManifoldObjective, q, σ, p)

Evaluate the proximal map of ``F`` stored within [`AbstractPrimalDualManifoldObjective`](@ref)

```math
$(_tex(:prox))_{σF}(x)
```

which can also be computed in place of `y`.
"""
get_primal_prox(::AbstractManifold, ::AbstractPrimalDualManifoldObjective, ::Any...)

function get_primal_prox(tmp::TwoManifoldProblem, σ, p)
    return get_primal_prox(get_manifold(tmp, 1), get_objective(tmp), σ, p)
end
function get_primal_prox!(tmp::TwoManifoldProblem, q, σ, p)
    get_primal_prox!(get_manifold(tmp, 1), q, get_objective(tmp), σ, p)
    return q
end

function get_primal_prox(
        M::AbstractManifold, apdmo::AbstractPrimalDualManifoldObjective{AllocatingEvaluation}, σ, p,
    )
    return apdmo.prox_f!!(M, σ, p)
end
function get_primal_prox(
        M::AbstractManifold, apdmo::AbstractPrimalDualManifoldObjective{InplaceEvaluation}, σ, p
    )
    q = allocate_result(M, get_primal_prox, p)
    return apdmo.prox_f!!(M, q, σ, p)
end
function get_primal_prox(
        M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, σ, p
    )
    return get_primal_prox(M, get_objective(admo, false), σ, p)
end

function get_primal_prox!(
        M::AbstractManifold, q, apdmo::AbstractPrimalDualManifoldObjective{AllocatingEvaluation}, σ, p,
    )
    copyto!(M, q, apdmo.prox_f!!(M, σ, p))
    return q
end
function get_primal_prox!(
        M::AbstractManifold, q, apdmo::AbstractPrimalDualManifoldObjective{InplaceEvaluation}, σ, p,
    )
    apdmo.prox_f!!(M, q, σ, p)
    return q
end
function get_primal_prox!(
        M::AbstractManifold, q, admo::AbstractDecoratedManifoldObjective, σ, p
    )
    return get_primal_prox!(M, q, get_objective(admo, false), σ, p)
end

@doc """
    Y = get_dual_prox(N::AbstractManifold, apdmo::AbstractPrimalDualManifoldObjective, n, τ, X)
    get_dual_prox!(N::AbstractManifold, apdmo::AbstractPrimalDualManifoldObjective, Y, n, τ, X)

Evaluate the proximal map of ``g_n^*`` stored within [`AbstractPrimalDualManifoldObjective`](@ref)

```math
  Y = $(_tex(:prox))}_{τG_n^*}(X)
```

which can also be computed in place of `Y`.
"""
get_dual_prox(::AbstractManifold, ::AbstractPrimalDualManifoldObjective, ::Any...)

function get_dual_prox(tmp::TwoManifoldProblem, n, τ, X)
    return get_dual_prox(get_manifold(tmp, 2), get_objective(tmp), n, τ, X)
end
function get_dual_prox!(tmp::TwoManifoldProblem, Y, n, τ, X)
    get_dual_prox!(get_manifold(tmp, 2), Y, get_objective(tmp), n, τ, X)
    return Y
end

function get_dual_prox(
        M::AbstractManifold, apdmo::AbstractPrimalDualManifoldObjective{AllocatingEvaluation}, n, τ, X,
    )
    return apdmo.prox_g_dual!!(M, n, τ, X)
end
function get_dual_prox(
        M::AbstractManifold, apdmo::AbstractPrimalDualManifoldObjective{InplaceEvaluation}, n, τ, X,
    )
    Y = allocate_result(M, get_dual_prox, X)
    apdmo.prox_g_dual!!(M, Y, n, τ, X)
    return Y
end
function get_dual_prox(
        M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, n, τ, X
    )
    return get_dual_prox(M, get_objective(admo, false), n, τ, X)
end

function get_dual_prox!(
        M::AbstractManifold, Y, apdmo::AbstractPrimalDualManifoldObjective{AllocatingEvaluation}, n, τ, X,
    )
    copyto!(M, Y, apdmo.prox_g_dual!!(M, n, τ, X))
    return Y
end
function get_dual_prox!(
        M::AbstractManifold, Y, apdmo::AbstractPrimalDualManifoldObjective{InplaceEvaluation}, n, τ, X,
    )
    apdmo.prox_g_dual!!(M, Y, n, τ, X)
    return Y
end
function get_dual_prox!(
        M::AbstractManifold, Y, admo::AbstractDecoratedManifoldObjective, n, τ, X
    )
    return get_dual_prox!(M, Y, get_objective(admo, false), n, τ, X)
end

@doc """
    Y = linearized_forward_operator(M::AbstractManifold, N::AbstractManifold, apdmo::AbstractPrimalDualManifoldObjective, m, X, n)
    linearized_forward_operator!(M::AbstractManifold, N::AbstractManifold, Y, apdmo::AbstractPrimalDualManifoldObjective, m, X, n)

Evaluate the linearized operator (differential) ``DΛ(m)[X]`` stored within
the [`AbstractPrimalDualManifoldObjective`](@ref) (in place of `Y`), where `n = Λ(m)`.
"""
linearized_forward_operator(
    ::AbstractManifold, ::AbstractPrimalDualManifoldObjective, ::Any...
)

function linearized_forward_operator(tmp::TwoManifoldProblem, m, X, n)
    return linearized_forward_operator(
        get_manifold(tmp, 1), get_manifold(tmp, 2), get_objective(tmp), m, X, n
    )
end
function linearized_forward_operator!(tmp::TwoManifoldProblem, Y, m, X, n)
    linearized_forward_operator!(
        get_manifold(tmp, 1), get_manifold(tmp, 2), Y, get_objective(tmp), m, X, n
    )
    return Y
end

function linearized_forward_operator(
        M::AbstractManifold, ::AbstractManifold, apdmo::AbstractPrimalDualManifoldObjective{AllocatingEvaluation}, m, X, ::Any,
    )
    return apdmo.linearized_forward_operator!!(M, m, X)
end
function linearized_forward_operator(
        M::AbstractManifold, N::AbstractManifold,
        apdmo::AbstractPrimalDualManifoldObjective{InplaceEvaluation}, m, X, n,
    )
    Y = zero_vector(N, n)
    apdmo.linearized_forward_operator!!(M, Y, m, X)
    return Y
end
function linearized_forward_operator(
        M::AbstractManifold, N::AbstractManifold,
        admo::AbstractDecoratedManifoldObjective, m, X, n,
    )
    return linearized_forward_operator(M, N, get_objective(admo, false), m, X, n)
end

function linearized_forward_operator!(
        M::AbstractManifold, N::AbstractManifold,
        Y, apdmo::AbstractPrimalDualManifoldObjective{AllocatingEvaluation}, m, X, n,
    )
    copyto!(N, Y, n, apdmo.linearized_forward_operator!!(M, m, X))
    return Y
end
function linearized_forward_operator!(
        M::AbstractManifold, ::AbstractManifold,
        Y, apdmo::AbstractPrimalDualManifoldObjective{InplaceEvaluation}, m, X, ::Any,
    )
    apdmo.linearized_forward_operator!!(M, Y, m, X)
    return Y
end
function linearized_forward_operator!(
        M::AbstractManifold, N::AbstractManifold,
        Y, admo::AbstractDecoratedManifoldObjective, m, X, n,
    )
    return linearized_forward_operator!(M, N, Y, get_objective(admo, false), m, X, n)
end

@doc """
    q = forward_operator(M::AbstractManifold, N::AbstractManifold, apdmo::AbstractPrimalDualManifoldObjective, p)
    forward_operator!(M::AbstractManifold, N::AbstractManifold, q, apdmo::AbstractPrimalDualManifoldObjective, p)

Evaluate the forward operator of ``Λ(x)`` stored within the [`TwoManifoldProblem`](@ref)
(in place of `q`).
"""
forward_operator(::AbstractManifold, ::AbstractPrimalDualManifoldObjective, ::Any...)

function forward_operator(tmp::TwoManifoldProblem, p)
    return forward_operator(
        get_manifold(tmp, 1), get_manifold(tmp, 2), get_objective(tmp), p
    )
end
function forward_operator!(tmp::TwoManifoldProblem, q, p)
    return forward_operator!(
        get_manifold(tmp, 1), get_manifold(tmp, 2), q, get_objective(tmp), p
    )
end

function forward_operator(
        M::AbstractManifold, ::AbstractManifold,
        apdmo::AbstractPrimalDualManifoldObjective{AllocatingEvaluation}, p,
    )
    return apdmo.Λ!!(M, p)
end
function forward_operator(
        M::AbstractManifold, N::AbstractManifold,
        apdmo::AbstractPrimalDualManifoldObjective{InplaceEvaluation}, p,
    )
    q = rand(N)
    apdmo.Λ!!(M, q, p)
    return q
end
function forward_operator(
        M::AbstractManifold, N::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p
    )
    return forward_operator(M, N, get_objective(admo, false), p)
end

function forward_operator!(
        M::AbstractManifold, N::AbstractManifold,
        q, apdmo::AbstractPrimalDualManifoldObjective{AllocatingEvaluation}, p,
    )
    copyto!(N, q, apdmo.Λ!!(M, p))
    return q
end
function forward_operator!(
        M::AbstractManifold, ::AbstractManifold,
        q, apdmo::AbstractPrimalDualManifoldObjective{InplaceEvaluation}, p,
    )
    apdmo.Λ!!(M, q, p)
    return q
end
function forward_operator!(
        M::AbstractManifold, N::AbstractManifold, q, admo::AbstractDecoratedManifoldObjective, p
    )
    return forward_operator!(M, N, q, get_objective(admo, false), p)
end

@doc """
    X = adjoint_linearized_operator(N::AbstractManifold, apdmo::AbstractPrimalDualManifoldObjective, m, n, Y)
    adjoint_linearized_operator(N::AbstractManifold, X, apdmo::AbstractPrimalDualManifoldObjective, m, n, Y)

Evaluate the adjoint of the linearized forward operator of ``(DΛ(m))^*[Y]`` stored within
the [`AbstractPrimalDualManifoldObjective`](@ref) (in place of `X`).
Since ``Y∈T_n$(_tex(:Cal, "N"))``, both ``m`` and ``n=Λ(m)`` are necessary arguments, mainly because
the forward operator ``Λ`` might be `missing` in `p`.
"""
adjoint_linearized_operator(
    ::AbstractManifold, ::AbstractPrimalDualManifoldObjective, ::Any...
)

function adjoint_linearized_operator(tmp::TwoManifoldProblem, m, n, Y)
    return adjoint_linearized_operator(
        get_manifold(tmp, 1), get_manifold(tmp, 2), get_objective(tmp), m, n, Y
    )
end
function adjoint_linearized_operator!(tmp::TwoManifoldProblem, X, m, n, Y)
    return adjoint_linearized_operator!(
        get_manifold(tmp, 1), get_manifold(tmp, 2), X, get_objective(tmp), m, n, Y
    )
end
function adjoint_linearized_operator(
        ::AbstractManifold, N::AbstractManifold,
        apdmo::AbstractPrimalDualManifoldObjective{AllocatingEvaluation}, m, n, Y,
    )
    return apdmo.adjoint_linearized_operator!!(N, m, n, Y)
end
function adjoint_linearized_operator(
        M::AbstractManifold, N::AbstractManifold,
        apdmo::AbstractPrimalDualManifoldObjective{InplaceEvaluation}, m, n, Y,
    )
    X = zero_vector(M, m)
    apdmo.adjoint_linearized_operator!!(N, X, m, n, Y)
    return X
end
function adjoint_linearized_operator(
        M::AbstractManifold, N::AbstractManifold,
        admo::AbstractDecoratedManifoldObjective, m, n, Y,
    )
    return adjoint_linearized_operator(M, N, get_objective(admo, false), m, n, Y)
end

function adjoint_linearized_operator!(
        M::AbstractManifold, N::AbstractManifold,
        X, apdmo::AbstractPrimalDualManifoldObjective{AllocatingEvaluation}, m, n, Y,
    )
    copyto!(M, X, apdmo.adjoint_linearized_operator!!(N, m, n, Y))
    return X
end
function adjoint_linearized_operator!(
        ::AbstractManifold, N::AbstractManifold,
        X, apdmo::AbstractPrimalDualManifoldObjective{InplaceEvaluation}, m, n, Y,
    )
    apdmo.adjoint_linearized_operator!!(N, X, m, n, Y)
    return X
end
function adjoint_linearized_operator!(
        M::AbstractManifold, N::AbstractManifold,
        X, admo::AbstractDecoratedManifoldObjective, m, n, Y,
    )
    return adjoint_linearized_operator!(M, N, X, get_objective(admo, false), m, n, Y)
end

function status_summary(pdmo::PrimalDualManifoldObjective; inline = false)
    both_missing = ismissing(pdmo.Λ!!) && ismissing(pdmo.linearized_forward_operator!!)
    inline && ("A primal dual objective with a cost of f+g, a prox for f, a prox for the dual of g, as well as $(!ismissing(pdmo.Λ!!) ? "an operator Λ," : "") $(!ismissing(pdmo.linearized_forward_operator!!) ? "DΛ, " : "")$(!both_missing ? "and " : "")an adjoint D^*Λ")

    maybe_line1 = ismissing(pdmo.Λ!!) ? "" : "\n* Λ:       $(pdmo.Λ!!)"
    maybe_line2 = ismissing(pdmo.linearized_forward_operator!!) ? "" : "\n* DΛ:      $(pdmo.linearized_forward_operator!!)"
    return """
    A primal dual objective with

    * cost:    $(pdmo.cost)
    * prox_f:  $(pdmo.prox_f!!)
    * prox_g*: $(pdmo.prox_g_dual!!)
    * D^*Λ:    $(pdmo.adjoint_linearized_operator!!)$(maybe_line1)$(maybe_line2)"""
end
function show(io::IO, pdmo::PrimalDualManifoldObjective{E}) where {E}
    print(io, "PrimalDualManifoldObjective(")
    show(io, pdmo.cost); print(io, ", ")
    show(io, pdmo.prox_f!!); print(io, ", ")
    show(io, pdmo.prox_g_dual!!); print(io, ", ")
    show(io, pdmo.adjoint_linearized_operator!!); print(io, ";\n$(_MANOPT_INDENT)")
    print(io, _to_kw(E)); print(io, ",")
    !ismissing(pdmo.Λ!!) && (print(io, " Λ = "); show(io, pdmo.Λ!!); print(io, ","))
    !ismissing(pdmo.linearized_forward_operator!!) && (print(io, " linearized_forward_operator = "); show(io, pdmo.linearized_forward_operator!!); print(io, ","))
    print(io, "\n")
    return print(io, ")")
end

@doc """
    AbstractPrimalDualSolverState

A general type for all primal dual based options to be used within primal dual
based algorithms
"""
abstract type AbstractPrimalDualSolverState <: AbstractManoptSolverState end

@doc """
    primal_residual(p, o, x_old, X_old, n_old)

Compute the primal residual at current iterate ``k`` given the necessary values ``x_{k-1},
X_{k-1}``, and ``n_{k-1}`` from the previous iterate.

```math
$(
    _tex(
        :norm,
        "$(_tex(:frac, "1", "σ"))$(_tex(:retr))^{-1}_{x_{k}}x_{k-1} - V_{x_k←m_k} $(_tex(:bigl))( DΛ^*(m_k)$(_tex(:bigl))[V_{n_k← n_{k-1}}X_{k-1} - X_k $(_tex(:bigr))]$(_tex(:bigr)))"
    )
)
```
where ``V_{⋅←⋅}`` is the vector transport used in the [`ChambollePockState`](@ref)
"""
function primal_residual(
        tmp::TwoManifoldProblem, apds::AbstractPrimalDualSolverState, p_old, X_old, n_old
    )
    return primal_residual(
        get_manifold(tmp, 1), get_manifold(tmp, 2), get_objective(tmp), apds, p_old, X_old, n_old,
    )
end
function primal_residual(
        M::AbstractManifold, N::AbstractManifold, apdmo::AbstractPrimalDualManifoldObjective,
        apds::AbstractPrimalDualSolverState, p_old, X_old, n_old,
    )
    return norm(
        M,
        apds.p,
        1 / apds.primal_stepsize *
            inverse_retract(M, apds.p, p_old, apds.inverse_retraction_method) -
            vector_transport_to(
            M, apds.m,
            adjoint_linearized_operator(
                M, N, apdmo, apds.m, apds.n,
                vector_transport_to(N, n_old, X_old, apds.n, apds.vector_transport_method_dual) - apds.X,
            ),
            apds.p, apds.vector_transport_method,
        ),
    )
end
@doc """
    dual_residual(p, o, x_old, X_old, n_old)

Compute the dual residual at current iterate ``k`` given the necessary values ``x_{k-1},
X_{k-1}``, and ``n_{k-1}`` from the previous iterate. The formula is slightly different depending
on the `o.variant` used:

For the `:linearized` it reads
```math
$(
    _tex(
        :norm,
        "$(_tex(:frac, "1", "τ"))$(_tex(:bigl))( V_{n_{k}← n_{k-1}}(X_{k-1}) - X_k $(_tex(:bigr)) ) - DΛ(m_k)$(_tex(:bigl))[ V_{m_k← x_k}$(_tex(:retr))^{-1}_{x_{k}}(x_{k-1})$(_tex(:bigr))]"
    )
)
```

and for the `:exact` variant

```math
$(
    _tex(
        :norm,
        "$(_tex(:frac, "1", "τ")) V_{n_{k}← n_{k-1}}(X_{k-1}) - $(_tex(:retr))^{-1}_{n_{k}}$(_tex(:bigl))( Λ($(_tex(:retr))_{m_{k}}(V_{m_k← x_k}$(_tex(:retr))^{-1}_{x_{k}}x_{k-1}))$(_tex(:bigr)))"
    )
)
```

where in both cases ``V_{⋅←⋅}`` is the vector transport used in the [`ChambollePockState`](@ref).
"""
function dual_residual(
        tmp::TwoManifoldProblem, apds::AbstractPrimalDualSolverState, p_old, X_old, n_old
    )
    return dual_residual(
        get_manifold(tmp, 1), get_manifold(tmp, 2), get_objective(tmp), apds, p_old, X_old, n_old,
    )
end

function dual_residual(
        M::AbstractManifold, N::AbstractManifold, apdmo::AbstractPrimalDualManifoldObjective,
        apds::AbstractPrimalDualSolverState, p_old, X_old, n_old,
    )
    if apds.variant === :linearized
        return norm(
            N,
            apds.n,
            1 / apds.dual_stepsize * (
                vector_transport_to(
                    N, n_old, X_old, apds.n, apds.vector_transport_method_dual
                ) - apds.X
            ) - linearized_forward_operator(
                M, N, apdmo, apds.m,
                vector_transport_to(
                    M, apds.p,
                    inverse_retract(M, apds.p, p_old, apds.inverse_retraction_method),
                    apds.m, apds.vector_transport_method,
                ),
                apds.n,
            ),
        )
    elseif apds.variant === :exact
        return norm(
            N,
            apds.n,
            1 / apds.dual_stepsize * (
                vector_transport_to(
                    N, n_old, X_old, apds.n, apds.vector_transport_method_dual
                ) - apds.n
            ) - inverse_retract(
                N, apds.n,
                forward_operator(
                    M, N, apdmo,
                    retract(
                        M, apds.m,
                        vector_transport_to(
                            M, apds.p,
                            inverse_retract(
                                M, apds.p, p_old, apds.inverse_retraction_method
                            ),
                            apds.m, apds.vector_transport_method,
                        ),
                        apds.retraction_method,
                    ),
                ),
                apds.inverse_retraction_method_dual,
            ),
        )
    else
        throw(
            DomainError(
                apds.variant, "Unknown Chambolle—Pock variant, allowed are `:exact` or `:linearized`.",
            ),
        )
    end
end
#
# Special Debuggers
#
@doc """
    DebugDualResidual <: DebugAction

A Debug action to print the dual residual.
The constructor accepts a printing function and some (shared) storage, which
should at least record `:Iterate`, `:X` and `:n`.

# Constructor
DebugDualResidual(; kwargs...)

# Keyword warguments

* `io=`stdout`: stream to perform the debug to
* `format="\$prefix%s"`: format to print the dual residual, using the
* `prefix="Dual Residual: "`: short form to just set the prefix
* `storage` (a new [`StoreStateAction`](@ref)) to store values for the debug.
"""
mutable struct DebugDualResidual <: DebugAction
    io::IO
    format::String
    storage::StoreStateAction
    function DebugDualResidual(;
            storage::StoreStateAction = StoreStateAction([:Iterate, :X, :n]),
            io::IO = stdout, prefix = "Dual Residual: ", format = "$prefix%s",
        )
        return new(io, format, storage)
    end
    function DebugDualResidual(
            initial_values::Tuple{P, T, Q};
            storage::StoreStateAction = StoreStateAction([:Iterate, :X, :n]),
            io::IO = stdout, prefix = "Dual Residual: ", format = "$prefix%s",
        ) where {P, T, Q}
        update_storage!(
            storage, Dict(k => v for (k, v) in zip((:Iterate, :X, :n), initial_values))
        )
        return new(io, format, storage)
    end
end
function (d::DebugDualResidual)(
        tmp::TwoManifoldProblem, apds::AbstractPrimalDualSolverState, k::Int
    )
    M = get_manifold(tmp, 1)
    N = get_manifold(tmp, 2)
    apdmo = get_objective(tmp)
    if all(has_storage.(Ref(d.storage), [:Iterate, :X, :n])) && k > 0 # all values stored
        #fetch
        p_old = get_storage(d.storage, :Iterate)
        X_old = get_storage(d.storage, :X)
        n_old = get_storage(d.storage, :n)
        Printf.format(
            d.io, Printf.Format(d.format),
            dual_residual(M, N, apdmo, apds, p_old, X_old, n_old),
        )
    end
    return d.storage(tmp, apds, k)
end
@doc """
    DebugPrimalResidual <: DebugAction

A Debug action to print the primal residual.
The constructor accepts a printing function and some (shared) storage, which
should at least record `:Iterate`, `:X` and `:n`.

# Constructor

    DebugPrimalResidual(; kwargs...)

# Keyword warguments

* `io=`stdout`: stream to perform the debug to
* `format="\$prefix%s"`: format to print the dual residual, using the
* `prefix="Primal Residual: "`: short form to just set the prefix
* `storage` (a new [`StoreStateAction`](@ref)) to store values for the debug.
"""
mutable struct DebugPrimalResidual <: DebugAction
    io::IO
    format::String
    storage::StoreStateAction
    function DebugPrimalResidual(;
            storage::StoreStateAction = StoreStateAction([:Iterate, :X, :n]),
            io::IO = stdout, prefix = "Primal Residual: ", format = "$prefix%s",
        )
        return new(io, format, storage)
    end
    function DebugPrimalResidual(
            values::Tuple{P, T, Q};
            storage::StoreStateAction = StoreStateAction([:Iterate, :X, :n]),
            io::IO = stdout, prefix = "Primal Residual: ", format = "$prefix%s",
        ) where {P, T, Q}
        update_storage!(storage, Dict(k => v for (k, v) in zip((:Iterate, :X, :n), values)))
        return new(io, format, storage)
    end
end
function (d::DebugPrimalResidual)(
        tmp::TwoManifoldProblem, apds::AbstractPrimalDualSolverState, k::Int
    )
    M = get_manifold(tmp, 1)
    N = get_manifold(tmp, 2)
    apdmo = get_objective(tmp)
    if all(has_storage.(Ref(d.storage), [:Iterate, :X, :n])) && k > 0 # all values stored
        #fetch
        p_old = get_storage(d.storage, :Iterate)
        X_old = get_storage(d.storage, :X)
        n_old = get_storage(d.storage, :n)
        Printf.format(
            d.io, Printf.Format(d.format),
            primal_residual(M, N, apdmo, apds, p_old, X_old, n_old),
        )
    end
    return d.storage(tmp, apds, k)
end
@doc """
    DebugPrimalDualResidual <: DebugAction

A Debug action to print the primal dual residual.
The constructor accepts a printing function and some (shared) storage, which
should at least record `:Iterate`, `:X` and `:n`.

# Constructor

    DebugPrimalDualResidual()

with the keywords

# Keyword warguments

* `io=`stdout`: stream to perform the debug to
* `format="\$prefix%s"`: format to print the dual residual, using the
* `prefix="PD Residual: "`: short form to just set the prefix
* `storage` (a new [`StoreStateAction`](@ref)) to store values for the debug.
"""
mutable struct DebugPrimalDualResidual <: DebugAction
    io::IO
    format::String
    storage::StoreStateAction
    function DebugPrimalDualResidual(;
            storage::StoreStateAction = StoreStateAction([:Iterate, :X, :n]),
            io::IO = stdout, prefix = "PD Residual: ", format = "$prefix%s",
        )
        return new(io, format, storage)
    end
    function DebugPrimalDualResidual(
            values::Tuple{P, T, Q};
            storage::StoreStateAction = StoreStateAction([:Iterate, :X, :n]),
            io::IO = stdout, prefix = "PD Residual: ", format = "$prefix%s",
        ) where {P, Q, T}
        update_storage!(storage, Dict(k => v for (k, v) in zip((:Iterate, :X, :n), values)))
        return new(io, format, storage)
    end
end
function (d::DebugPrimalDualResidual)(
        tmp::TwoManifoldProblem, apds::AbstractPrimalDualSolverState, k::Int
    )
    M = get_manifold(tmp, 1)
    N = get_manifold(tmp, 2)
    apdmo = get_objective(tmp)
    if all(has_storage.(Ref(d.storage), [:Iterate, :X, :n])) && k > 0 # all values stored
        #fetch
        p_old = get_storage(d.storage, :Iterate)
        X_old = get_storage(d.storage, :X)
        n_old = get_storage(d.storage, :n)
        v = primal_residual(M, N, apdmo, apds, p_old, X_old, n_old) + dual_residual(tmp, apds, p_old, X_old, n_old)
        Printf.format(d.io, Printf.Format(d.format), v / manifold_dimension(M))
    end
    return d.storage(tmp, apds, k)
end

#
# Debugs
#
"""
    DebugPrimalChange(opts...)

Print the change of the primal variable by using [`DebugChange`](@ref),
see their constructors for detail.
"""
function DebugPrimalChange(;
        storage::StoreStateAction = StoreStateAction([:Iterate]), prefix = "Primal Change: ", kwargs...,
    )
    return DebugChange(; storage = storage, prefix = prefix, kwargs...)
end

"""
    DebugPrimalIterate(opts...;kwargs...)

Print the change of the primal variable by using [`DebugIterate`](@ref),
see their constructors for detail.
"""
DebugPrimalIterate(opts...; kwargs...) = DebugIterate(opts...; kwargs...)

"""
    DebugDualIterate(e)

Print the dual variable by using [`DebugEntry`](@ref),
see their constructors for detail.
This method is further set display `o.X`.
"""
DebugDualIterate(opts...; kwargs...) = DebugEntry(:X, opts...; kwargs...)

"""
    DebugDualChange(opts...)

Print the change of the dual variable, similar to [`DebugChange`](@ref),
see their constructors for detail, but with a different calculation of the change,
since the dual variable lives in (possibly different) tangent spaces.
"""
mutable struct DebugDualChange <: DebugAction
    io::IO
    format::String
    storage::StoreStateAction
    function DebugDualChange(;
            storage::StoreStateAction = StoreStateAction([:X, :n]),
            io::IO = stdout, prefix = "Dual Change: ", format = "$prefix%s",
        )
        return new(io, format, storage)
    end
    function DebugDualChange(
            values::Tuple{T, P};
            storage::StoreStateAction = StoreStateAction([:X, :n]),
            io::IO = stdout, prefix = "Dual Change: ", format = "$prefix%s",
        ) where {P, T}
        update_storage!(
            storage, Dict{Symbol, Any}(k => v for (k, v) in zip((:X, :n), values))
        )
        return new(io, format, storage)
    end
end
function (d::DebugDualChange)(
        tmp::TwoManifoldProblem, apds::AbstractPrimalDualSolverState, k::Int
    )
    N = get_manifold(tmp, 2)
    if all(has_storage.(Ref(d.storage), [:X, :n])) && k > 0 # all values stored
        #fetch
        X_old = get_storage(d.storage, :X)
        n_old = get_storage(d.storage, :n)
        v = norm(
            N, apds.n,
            vector_transport_to(
                N, n_old, X_old, apds.n, apds.vector_transport_method_dual
            ) - apds.X,
        )
        Printf.format(d.io, Printf.Format(d.format), v)
    end
    return d.storage(tmp, apds, k)
end

"""
    DebugDualBaseIterate(io::IO=stdout)

Print the dual base variable by using [`DebugEntry`](@ref),
see their constructors for detail.
This method is further set display `o.n`.
"""
DebugDualBaseIterate(; kwargs...) = DebugEntry(:n; kwargs...)

"""
    DebugDualChange(; storage=StoreStateAction([:n]), io::IO=stdout)

Print the change of the dual base variable by using [`DebugEntryChange`](@ref),
see their constructors for detail, on `o.n`.
"""
function DebugDualBaseChange(;
        storage::StoreStateAction = StoreStateAction([:n]), prefix = "Dual Base Change:", kwargs...
    )
    return DebugEntryChange(
        :n, (p, o, x, y) -> distance(get_manifold(p, 2), x, y, o.inverse_retraction_method_dual);
        storage = storage, prefix = prefix, kwargs...,
    )
end

"""
    DebugPrimalBaseIterate()

Print the primal base variable by using [`DebugEntry`](@ref),
see their constructors for detail.
This method is further set display `o.m`.
"""
DebugPrimalBaseIterate(opts...; kwargs...) = DebugEntry(:m, opts...; kwargs...)

"""
    DebugPrimalBaseChange(a::StoreStateAction=StoreStateAction([:m]),io::IO=stdout)

Print the change of the primal base variable by using [`DebugEntryChange`](@ref),
see their constructors for detail, on `o.n`.
"""
function DebugPrimalBaseChange(opts...; prefix = "Primal Base Change:", kwargs...)
    return DebugEntryChange(
        :m, (p, o, x, y) -> distance(get_manifold(p, 1), x, y),
        opts...; prefix = prefix, kwargs...,
    )
end

#
# Records
#

# For primal changes just use the usual change and record functors
"""
    RecordPrimalChange(a)

Create an [`RecordAction`](@ref) that records the primal value change,
[`RecordChange`](@ref), to record the change of `o.x`.
"""
RecordPrimalChange() = RecordChange()

"""
    RecordDualBaseIterate(x)

Create an [`RecordAction`](@ref) that records the dual base point,
an [`RecordIterate`](@ref) of `o.x`.
"""
RecordPrimalIterate(p) = RecordIterate(p)

"""
    RecordDualIterate(X)

Create an [`RecordAction`](@ref) that records the dual base point,
an [`RecordEntry`](@ref) of `o.X`.
"""
RecordDualIterate(X) = RecordEntry(X, :X)

"""
    RecordDualChange()

Create the action either with a given (shared) Storage, which can be set to the
`values` Tuple, if that is provided).
"""
function RecordDualChange()
    return RecordEntryChange(:X, (p, o, x, y) -> distance(get_manifold(p, 2), x, y))
end

"""
    RecordDualBaseIterate(n)

Create an [`RecordAction`](@ref) that records the dual base point,
an [`RecordEntry`](@ref) of `o.n`.
"""
RecordDualBaseIterate(n) = RecordEntry(n, :n)

"""
    RecordDualBaseChange(e)

Create an [`RecordAction`](@ref) that records the dual base point change,
an [`RecordEntryChange`](@ref) of `o.n` with distance to the last value to store a value.
"""
function RecordDualBaseChange()
    return RecordEntryChange(:n, (p, o, x, y) -> distance(get_manifold(p, 2), x, y))
end

"""
    RecordPrimalBaseIterate(x)

Create an [`RecordAction`](@ref) that records the primal base point,
an [`RecordEntry`](@ref) of `o.m`.
"""
RecordPrimalBaseIterate(m) = RecordEntry(m, :m)

"""
    RecordPrimalBaseChange()

Create an [`RecordAction`](@ref) that records the primal base point change,
an [`RecordEntryChange`](@ref) of `o.m` with distance to the last value to store a value.
"""
function RecordPrimalBaseChange()
    return RecordEntryChange(:m, (p, o, x, y) -> distance(get_manifold(p, 1), x, y))
end
