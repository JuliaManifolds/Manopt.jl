function get_proximal_map end

function get_proximal_map(amp::AbstractManoptProblem, λ, p, i)
    return get_proximal_map(get_manifold(amp), get_objective(amp), λ, p, i)
end
function get_proximal_map!(amp::AbstractManoptProblem, q, λ, p, i)
    return get_proximal_map!(get_manifold(amp), q, get_objective(amp), λ, p, i)
end
function get_proximal_map(amp::AbstractManoptProblem, λ, p)
    return get_proximal_map(get_manifold(amp), get_objective(amp), λ, p)
end
function get_proximal_map!(amp::AbstractManoptProblem, q, λ, p)
    return get_proximal_map!(get_manifold(amp), q, get_objective(amp), λ, p)
end
function get_proximal_map(M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, λ, p)
    return get_proximal_map(M, get_objective(admo, false), λ, p)
end
function get_proximal_map(M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, λ, p, i)
    return get_proximal_map(M, get_objective(admo, false), λ, p, i)
end
function get_proximal_map!(
        M::AbstractManifold, q, admo::AbstractDecoratedManifoldObjective, λ, p
    )
    return get_proximal_map!(M, q, get_objective(admo, false), λ, p)
end
function get_proximal_map!(
        M::AbstractManifold, q, admo::AbstractDecoratedManifoldObjective, λ, p, i
    )
    return get_proximal_map!(M, q, get_objective(admo, false), λ, p, i)
end

@doc """
    get_subgradient(amp::AbstractManoptProblem, p)
    get_subgradient!(amp::AbstractManoptProblem, X, p)

evaluate the subgradient of an [`AbstractManoptProblem`](@ref) `amp` at point `p`.

The evaluation is done in place of `X` for the `!`-variant.
The result might not be deterministic, _one_ element of the subdifferential is returned.
"""
function get_subgradient(amp::AbstractManoptProblem, p)
    return get_subgradient(get_manifold(amp), get_objective(amp), p)
end
function get_subgradient!(amp::AbstractManoptProblem, X, p)
    return get_subgradient!(get_manifold(amp), X, get_objective(amp), p)
end
function get_subgradient(M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p)
    return get_subgradient(M, get_objective(admo, false), p)
end

function get_subgradient!(
        M::AbstractManifold, X, admo::AbstractDecoratedManifoldObjective, p
    )
    return get_subgradient!(M, X, get_objective(admo, false), p)
end
function get_subgradient_function(
        admo::AbstractDecoratedManifoldObjective, recursive = false; evaluation::AbstractEvaluationType = AllocatingEvaluation()
    )
    return get_subgradient_function(get_objective(admo, recursive); evaluation = evaluation)
end

#
#
# ---
@doc """
    AbstractPrimalDualManifoldObjective{C,P} <: AbstractManifoldCostObjective{C}

A common abstract super type for objectives that consider primal-dual problems.
"""
abstract type AbstractPrimalDualManifoldObjective{C, P} <: AbstractManifoldCostObjective{C} end

function adjoint_linearized_operator end

@doc """
    X = adjoint_linearized_operator(N::AbstractManifold, apdmo::AbstractPrimalDualManifoldObjective, m, n, Y)
    adjoint_linearized_operator(N::AbstractManifold, X, apdmo::AbstractPrimalDualManifoldObjective, m, n, Y)

Evaluate the adjoint of the linearized forward operator of ``(DΛ(m))^*[Y]`` stored within
the [`AbstractPrimalDualManifoldObjective`](@ref) (in place of `X`).
Since ``Y∈$(_math(:TangentSpace; p = "n", M = "N"))``, both ``m`` and ``n=Λ(m)`` are necessary arguments, mainly because
the forward operator ``Λ`` might be `missing` in `p`.
"""
adjoint_linearized_operator(::AbstractManifold, ::AbstractPrimalDualManifoldObjective, ::Any...)

function adjoint_linearized_operator(
        M::AbstractManifold, N::AbstractManifold,
        apdmo::AbstractPrimalDualManifoldObjective, m, n, Y,
    )
    X = zero_vector(M, m)
    apdmo.adjoint_linearized_operator!(N, X, m, n, Y)
    return X
end
function adjoint_linearized_operator(
        M::AbstractManifold, N::AbstractManifold,
        admo::AbstractDecoratedManifoldObjective, m, n, Y,
    )
    return adjoint_linearized_operator(M, N, get_objective(admo, false), m, n, Y)
end

function adjoint_linearized_operator!(
        ::AbstractManifold, N::AbstractManifold,
        X, apdmo::AbstractPrimalDualManifoldObjective, m, n, Y,
    )
    apdmo.adjoint_linearized_operator!(N, X, m, n, Y)
    return X
end
function adjoint_linearized_operator!(
        M::AbstractManifold, N::AbstractManifold,
        X, admo::AbstractDecoratedManifoldObjective, m, n, Y,
    )
    return adjoint_linearized_operator!(M, N, X, get_objective(admo, false), m, n, Y)
end

function forward_operator end

@doc """
    q = forward_operator(M::AbstractManifold, N::AbstractManifold, apdmo::AbstractPrimalDualManifoldObjective, p)
    forward_operator!(M::AbstractManifold, N::AbstractManifold, q, apdmo::AbstractPrimalDualManifoldObjective, p)

Evaluate the forward operator of ``Λ(x)`` stored within the [`TwoManifoldProblem`](@ref)
(in place of `q`).
"""
forward_operator(::AbstractManifold, ::AbstractPrimalDualManifoldObjective, ::Any...)

function forward_operator(
        M::AbstractManifold, N::AbstractManifold, apdmo::AbstractPrimalDualManifoldObjective, p,
    )
    q = rand(N)
    apdmo.Λ!(M, q, p)
    return q
end
function forward_operator(
        M::AbstractManifold, N::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p
    )
    return forward_operator(M, N, get_objective(admo, false), p)
end
function forward_operator!(
        M::AbstractManifold, ::AbstractManifold, q, apdmo::AbstractPrimalDualManifoldObjective, p,
    )
    apdmo.Λ!(M, q, p)
    return q
end
function forward_operator!(
        M::AbstractManifold, N::AbstractManifold, q, admo::AbstractDecoratedManifoldObjective, p
    )
    return forward_operator!(M, N, q, get_objective(admo, false), p)
end

function get_dual_prox end
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

function get_dual_prox(M::AbstractManifold, apdmo::AbstractPrimalDualManifoldObjective, n, τ, X)
    Y = allocate_result(M, get_dual_prox, X)
    apdmo.prox_g_dual!(M, Y, n, τ, X)
    return Y
end
function get_dual_prox(M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, n, τ, X)
    return get_dual_prox(M, get_objective(admo, false), n, τ, X)
end
function get_dual_prox!(M::AbstractManifold, Y, apdmo::AbstractPrimalDualManifoldObjective, n, τ, X)
    apdmo.prox_g_dual!(M, Y, n, τ, X)
    return Y
end
function get_dual_prox!(M::AbstractManifold, Y, admo::AbstractDecoratedManifoldObjective, n, τ, X)
    return get_dual_prox!(M, Y, get_objective(admo, false), n, τ, X)
end

function get_primal_prox end
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

function get_primal_prox(
        M::AbstractManifold, apdmo::AbstractPrimalDualManifoldObjective, σ, p
    )
    q = allocate_result(M, get_primal_prox, p)
    return apdmo.prox_f!(M, q, σ, p)
end
function get_primal_prox(
        M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, σ, p
    )
    return get_primal_prox(M, get_objective(admo, false), σ, p)
end
function get_primal_prox!(
        M::AbstractManifold, q, apdmo::AbstractPrimalDualManifoldObjective, σ, p,
    )
    apdmo.prox_f!(M, q, σ, p)
    return q
end
function get_primal_prox!(
        M::AbstractManifold, q, admo::AbstractDecoratedManifoldObjective, σ, p
    )
    return get_primal_prox!(M, q, get_objective(admo, false), σ, p)
end

function linearized_forward_operator end

@doc """
    Y = linearized_forward_operator(M::AbstractManifold, N::AbstractManifold, apdmo::AbstractPrimalDualManifoldObjective, m, X, n)
    linearized_forward_operator!(M::AbstractManifold, N::AbstractManifold, Y, apdmo::AbstractPrimalDualManifoldObjective, m, X, n)

Evaluate the linearized operator (differential) ``DΛ(m)[X]`` stored within
the [`AbstractPrimalDualManifoldObjective`](@ref) (in place of `Y`), where `n = Λ(m)`.
"""
linearized_forward_operator(::AbstractManifold, ::AbstractPrimalDualManifoldObjective, ::Any...)

function linearized_forward_operator(
        M::AbstractManifold, N::AbstractManifold, apdmo::AbstractPrimalDualManifoldObjective, m, X, n
    )
    Y = zero_vector(N, n)
    apdmo.linearized_forward_operator!(M, Y, m, X)
    return Y
end
function linearized_forward_operator(
        M::AbstractManifold, N::AbstractManifold, admo::AbstractDecoratedManifoldObjective, m, X, n,
    )
    return linearized_forward_operator(M, N, get_objective(admo, false), m, X, n)
end
function linearized_forward_operator!(
        M::AbstractManifold, ::AbstractManifold, Y, apdmo::AbstractPrimalDualManifoldObjective, m, X, ::Any,
    )
    apdmo.linearized_forward_operator!(M, Y, m, X)
    return Y
end
function linearized_forward_operator!(
        M::AbstractManifold, N::AbstractManifold, Y, admo::AbstractDecoratedManifoldObjective, m, X, n,
    )
    return linearized_forward_operator!(M, N, Y, get_objective(admo, false), m, X, n)
end
