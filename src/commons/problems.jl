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
