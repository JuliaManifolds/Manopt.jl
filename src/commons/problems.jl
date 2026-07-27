@doc """
    ConstrainedManoptProblem{
        TM <: AbstractManifold,
        O <: AbstractManifoldObjective
        HR<:Union{AbstractPowerRepresentation,Nothing},
        GR<:Union{AbstractPowerRepresentation,Nothing},
        HHR<:Union{AbstractPowerRepresentation,Nothing},
        GHR<:Union{AbstractPowerRepresentation,Nothing},
    } <: AbstractManoptProblem{TM}

A constrained problem might feature different ranges for the
(vectors of) gradients of the equality and inequality constraints.

The ranges are required in a few places to allocate memory and access elements
correctly, they work as follows:

Assume the objective is
```math
\\begin{aligned}
 $(_tex(:argmin))_{p ∈ $(_math(:Manifold))} & f(p)\\\\
 $(_tex(:text, "subject to ")) & g_i(p) ≤ 0 $(_tex(:quad)) $(_tex(:text, " for all ")) i=1,…,m,\\
 $(_tex(:quad)) & h_j(p)=0 $(_tex(:quad)) $(_tex(:text, " for all ")) j=1,…,n.
\\end{aligned}
```

then the gradients can (classically) be considered as vectors of the
components gradients, for example
``$(_tex(:bigl))($(_tex(:grad)) g_1(p), $(_tex(:grad)) g_2(p), …, $(_tex(:grad)) g_m(p) $(_tex(:bigr)))``.

In another interpretation, this can be considered a point on the tangent space
at ``P = (p,…,p) ∈ $(_math(:Manifold))^m``, so in the tangent space to the [`PowerManifold`](@extref `ManifoldsBase.PowerManifold`) ``$(_math(:Manifold))^m``.
The case where this is a [`NestedPowerRepresentation`](@extref `ManifoldsBase.NestedPowerRepresentation`) this agrees with the
interpretation from before, but on power manifolds, more efficient representations exist.

To then access the elements, the range has to be specified. That is what this
problem is for.

# Constructor
    ConstrainedManoptProblem(
        M::AbstractManifold,
        co::ConstrainedManifoldObjective;
        range=NestedPowerRepresentation(),
        gradient_equality_range=range,
        gradient_inequality_range=range
        hessian_equality_range=range,
        hessian_inequality_range=range
    )

Creates a constrained Manopt problem specifying an [`AbstractPowerRepresentation`](@extref `ManifoldsBase.AbstractPowerRepresentation`)
for both the `gradient_equality_range` and the `gradient_inequality_range`, respectively.
"""
struct ConstrainedManoptProblem{
        TM <: AbstractManifold, O <: AbstractManifoldObjective,
        HR <: Union{AbstractPowerRepresentation, Nothing}, GR <: Union{AbstractPowerRepresentation, Nothing},
        HHR <: Union{AbstractPowerRepresentation, Nothing}, GHR <: Union{AbstractPowerRepresentation, Nothing},
    } <: AbstractManoptProblem{TM}
    manifold::TM
    grad_equality_range::HR
    grad_inequality_range::GR
    hess_equality_range::HHR
    hess_inequality_range::GHR
    objective::O
end

function ConstrainedManoptProblem(
        M::TM, objective::O;
        range::AbstractPowerRepresentation = NestedPowerRepresentation(),
        gradient_equality_range::HR = range, gradient_inequality_range::GR = range,
        hessian_equality_range::HHR = range, hessian_inequality_range::GHR = range,
    ) where {
        TM <: AbstractManifold, O <: AbstractManifoldObjective,
        GR <: Union{AbstractPowerRepresentation, Nothing}, HR <: Union{AbstractPowerRepresentation, Nothing},
        GHR <: Union{AbstractPowerRepresentation, Nothing}, HHR <: Union{AbstractPowerRepresentation, Nothing},
    }
    return ConstrainedManoptProblem{TM, O, HR, GR, HHR, GHR}(
        M, gradient_equality_range, gradient_inequality_range,
        hessian_equality_range, hessian_inequality_range, objective,
    )
end
function get_grad_equality_constraint(
        amp::AbstractManoptProblem, p, j = :,
        range::AbstractPowerRepresentation = NestedPowerRepresentation(),
    )
    return get_grad_equality_constraint(get_manifold(amp), get_objective(amp), p, j, range)
end
function get_grad_equality_constraint(cmp::ConstrainedManoptProblem, p, j = :)
    return get_grad_equality_constraint(
        get_manifold(cmp), get_objective(cmp), p, j, cmp.grad_equality_range
    )
end
function get_grad_equality_constraint!(cmp::ConstrainedManoptProblem, X, p, j = :)
    return get_grad_equality_constraint!(
        get_manifold(cmp), X, get_objective(cmp), p, j, cmp.grad_equality_range
    )
end
function get_grad_inequality_constraint(
        amp::AbstractManoptProblem, p, j = :, range::AbstractPowerRepresentation = NestedPowerRepresentation(),
    )
    return get_grad_inequality_constraint(get_manifold(amp), get_objective(amp), p, j, range)
end
function get_grad_inequality_constraint(cmp::ConstrainedManoptProblem, p, j = :)
    return get_grad_inequality_constraint(
        get_manifold(cmp), get_objective(cmp), p, j, cmp.grad_inequality_range
    )
end
function get_grad_inequality_constraint!(amp::AbstractManoptProblem, X, p, j)
    return get_grad_inequality_constraint!(get_manifold(amp), X, get_objective(amp), p, j)
end
function get_grad_inequality_constraint!(cmp::ConstrainedManoptProblem, X, p, j)
    return get_grad_inequality_constraint!(
        get_manifold(cmp), X, get_objective(cmp), p, j, cmp.grad_inequality_range
    )
end
function get_hess_equality_constraint!(
        amp::AbstractManoptProblem, Y, p, X, j = :,
        range::AbstractPowerRepresentation = NestedPowerRepresentation(),
    )
    return get_hess_equality_constraint!(
        get_manifold(amp), Y, get_objective(amp), p, X, j, range
    )
end
function get_hess_equality_constraint!(cmp::ConstrainedManoptProblem, Y, p, X, j = :)
    return get_hess_equality_constraint!(
        get_manifold(cmp), Y, get_objective(cmp), p, X, j, cmp.hess_equality_range
    )
end
function get_hess_equality_constraint(amp::AbstractManoptProblem, p, X, j = :)
    return get_hess_equality_constraint(get_manifold(amp), get_objective(amp), p, X, j)
end
function get_hess_equality_constraint(cmp::ConstrainedManoptProblem, p, X, j = :)
    return get_hess_equality_constraint(
        get_manifold(cmp), get_objective(cmp), p, X, j, cmp.hess_equality_range
    )
end
function get_hess_inequality_constraint(
        amp::AbstractManoptProblem, p, X, j = :,
        range::AbstractPowerRepresentation = NestedPowerRepresentation(),
    )
    return get_hess_inequality_constraint(
        get_manifold(amp), get_objective(amp), p, X, j, range
    )
end
function get_hess_inequality_constraint(cmp::ConstrainedManoptProblem, p, X, j = :)
    return get_hess_inequality_constraint(
        get_manifold(cmp), get_objective(cmp), p, X, j, cmp.hess_inequality_range
    )
end
function get_hess_inequality_constraint!(
        amp::AbstractManoptProblem, Y, p, X, j = :,
        range::AbstractPowerRepresentation = NestedPowerRepresentation(),
    )
    return get_hess_inequality_constraint!(
        get_manifold(amp), Y, get_objective(amp), p, X, j, range
    )
end
function get_hess_inequality_constraint!(cmp::ConstrainedManoptProblem, Y, p, X, j = :)
    return get_hess_inequality_constraint!(
        get_manifold(cmp), Y, get_objective(cmp), p, X, j, cmp.hess_inequality_range
    )
end

get_manifold(cmp::ConstrainedManoptProblem) = cmp.manifold
get_objective(cmp::ConstrainedManoptProblem) = cmp.objective


function show(io::IO, cmp::ConstrainedManoptProblem)
    print(io, "ConstrainedManoptProblem(", cmp.manifold, ", ", cmp.objective, ";")
    print(io, " gradient_equality_range = ", cmp.grad_equality_range)
    print(io, ", gradient_inequality_range = ", cmp.grad_inequality_range)
    print(io, ", hessian_equality_range = ", cmp.hess_equality_range)
    print(io, ", hessian_inequality_range = ", cmp.hess_inequality_range)
    return print(io, ")")
end

function status_summary(cmp::ConstrainedManoptProblem; context::Symbol = :default)
    _is_inline(context) && return "A constrained optimization problem to minimize $(cmp.objective) on the manifold $(cmp.manifold)"
    return """
    A constrained optimization problem for Manopt.jl

    ## Manifold
    $(_in_str(repr(cmp.manifold); indent = 1, headers = 0))

    ## Objective
    $(_in_str(status_summary(cmp.objective, context = context); indent = 1))

    ## Ranges
    * gradient equality range: $(_MANOPT_INDENT)$(cmp.grad_equality_range)
    * gradient inequality range: $(_MANOPT_INDENT)$(cmp.grad_inequality_range)
    * hessian equality range: $(_MANOPT_INDENT)$(cmp.hess_equality_range)
    * hessian inequality range: $(_MANOPT_INDENT)$(cmp.hess_inequality_range)
    """
end

@doc """
    DefaultManoptProblem{TM <: AbstractManifold, Objective <: AbstractManifoldObjective}

Model a default manifold problem, that (just) consists of the domain of optimisation,
that is an `AbstractManifold` and an [`AbstractManifoldObjective`](@ref)
"""
struct DefaultManoptProblem{TM <: AbstractManifold, O <: AbstractManifoldObjective} <:
    AbstractManoptProblem{TM}
    manifold::TM
    objective::O
end

function show(io::IO, dmp::DefaultManoptProblem)
    print(io, "DefaultManoptProblem(")
    show(io, dmp.manifold)
    print(io, ", ")
    show(io, dmp.objective)
    return print(io, ")")
end

function status_summary(dmp::DefaultManoptProblem; context::Symbol = :default)
    _is_inline(context) && return "An optimization problem to minimize $(dmp.objective) on the manifold $(dmp.manifold)"
    return """
    An optimization problem for Manopt.jl

    ## Manifold
    $(_MANOPT_INDENT)$(replace(repr(dmp.manifold), "\n#" => "\n$(_MANOPT_INDENT)##", "\n" => "\n$(_MANOPT_INDENT)"))

    ## Objective
    $(_in_str(status_summary(dmp.objective, context = context); indent = 1))"""
end

get_manifold(amp::DefaultManoptProblem) = amp.manifold

function get_objective(amp::DefaultManoptProblem, recursive = false)
    return recursive ? get_objective(amp.objective, true) : amp.objective
end

#
#
# ---
@doc """
    TwoManifoldProblem{
        MT<:AbstractManifold,NT<:AbstractManifold,O<:AbstractManifoldObjective
    } <: AbstractManoptProblem{MT}

An abstract type for problems that require two manifolds, for example the primal-dual-based problems.
"""
struct TwoManifoldProblem{
        MT <: AbstractManifold, NT <: AbstractManifold, S <: AbstractManifoldObjective,
    } <: AbstractManoptProblem{MT}
    first_manifold::MT
    second_manifold::NT
    objective::S
end

function adjoint_linearized_operator(tmp::TwoManifoldProblem, m, n, Y)
    return adjoint_linearized_operator(get_manifold(tmp, 1), get_manifold(tmp, 2), get_objective(tmp), m, n, Y)
end
function adjoint_linearized_operator!(tmp::TwoManifoldProblem, X, m, n, Y)
    return adjoint_linearized_operator!(get_manifold(tmp, 1), get_manifold(tmp, 2), X, get_objective(tmp), m, n, Y)
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

function forward_operator(tmp::TwoManifoldProblem, p)
    return forward_operator(get_manifold(tmp, 1), get_manifold(tmp, 2), get_objective(tmp), p)
end
function forward_operator!(tmp::TwoManifoldProblem, q, p)
    return forward_operator!(get_manifold(tmp, 1), get_manifold(tmp, 2), q, get_objective(tmp), p)
end


get_manifold(tmp::TwoManifoldProblem) = get_manifold(tmp, 1)
get_manifold(tmp::TwoManifoldProblem, i) = _get_manifold(tmp, Val(i))
_get_manifold(tmp::TwoManifoldProblem, ::Val{1}) = tmp.first_manifold
_get_manifold(tmp::TwoManifoldProblem, ::Val{2}) = tmp.second_manifold

get_objective(tmo::TwoManifoldProblem) = tmo.objective

function get_dual_prox(tmp::TwoManifoldProblem, n, τ, X)
    return get_dual_prox(get_manifold(tmp, 2), get_objective(tmp), n, τ, X)
end
function get_dual_prox!(tmp::TwoManifoldProblem, Y, n, τ, X)
    get_dual_prox!(get_manifold(tmp, 2), Y, get_objective(tmp), n, τ, X)
    return Y
end

function get_primal_prox(tmp::TwoManifoldProblem, σ, p)
    return get_primal_prox(get_manifold(tmp, 1), get_objective(tmp), σ, p)
end
function get_primal_prox!(tmp::TwoManifoldProblem, q, σ, p)
    get_primal_prox!(get_manifold(tmp, 1), q, get_objective(tmp), σ, p)
    return q
end

function linearized_forward_operator(tmp::TwoManifoldProblem, m, X, n)
    return linearized_forward_operator(get_manifold(tmp, 1), get_manifold(tmp, 2), get_objective(tmp), m, X, n)
end
function linearized_forward_operator!(tmp::TwoManifoldProblem, Y, m, X, n)
    linearized_forward_operator!(get_manifold(tmp, 1), get_manifold(tmp, 2), Y, get_objective(tmp), m, X, n)
    return Y
end

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

function show(io::IO, tmp::TwoManifoldProblem)
    print(io, "TwoManifoldProblem("); show(io, tmp.first_manifold)
    print(io, ", "); show(io, tmp.second_manifold)
    print(io, ", "); show(io, tmp.objective)
    return print(io, ")")
end
function status_summary(tmp::TwoManifoldProblem; context::Symbol = :default)
    _is_inline(context) && return "An optimization problem to minimize $(tmp.objective) using a primal manifold $(tmp.first_manifold) and a dual manifold $(tmp.second_manifold)."
    return """
    An optimization problem for Manopt.jl requiring a primal and a dual manifold

    ## Manifolds
    * $(_in_str(repr(tmp.first_manifold); indent = 1))
    * $(_in_str(repr(tmp.second_manifold); indent = 1))

    ## Objective
    $(_in_str(status_summary(tmp.objective, context = context); indent = 1))"""
end
