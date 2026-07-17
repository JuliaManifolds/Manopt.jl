#
# Define a global problem and its constructors
#
# ---

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

@doc """
    get_cost(M::AbstractManifold, obj::AbstractManifoldObjective, p)

evaluate the cost function `f` defined on `M` stored within the [`AbstractManifoldObjective`](@ref) at the point `p`.
"""
get_cost(::AbstractManifold, ::AbstractManifoldObjective, p)

function set_parameter!(TpM::TangentSpace, ::Union{Val{:Basepoint}, Val{:p}}, p)
    copyto!(TpM.manifold, TpM.point, p)
    return TpM
end
