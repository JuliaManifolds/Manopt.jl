"""
    ConstrainedSetObjective{E, MO, PF, IF} <: AbstractManifoldObjective{E}

Model a constrained objective restricted to a set

```math
$(_tex(:argmin))_{p ∈ $(_tex(:Cal,"C"))} f(p)
```

where ``$(_tex(:Cal,"C")) ⊂ $(_math(:M))`` is a convex closed subset.

# Fields

* `objective::AbstractManifoldObjective` the (unconstrained) objective, which
  contains ``f`` and for example ist gradient ``$(_tex(:grad)) f``.
* `project!!::PF` a projection function ``$(_tex(:proj))_{$(_tex(:Cal,"C"))}: $(_math(:M)) → $(_tex(:Cal,"C"))`` that projects onto the set ``$(_tex(:Cal,"C"))``.
* `indicator::IF` the indicator function ``ι_{$(_tex(:Cal,"C"))}(p) = $(_tex(:cases, "0 &"*_tex(:text, " for ")*"p∈"*_tex(:Cal,"C"), "∞ &"*_tex(:text, " else.")))

# Constructor

    ConstrainedSetObjective(f, grad_f, project!!; kwargs...)

Generate the constrained objective for a given function `f` its gradient `grad_f` and a projection `project!!` ``$(_tex(:proj))_{$(_tex(:Cal,"C"))}``.

## Keyword arguments

$(_var(:Keyword, :evaluation))
* `indicator=nothing`: the indicator function ``ι_{$(_tex(:Cal,"C"))}(p)``. If not provided a test, whether the projection yields the same point is performed.
  For the [`InplaceEvaluation`](@ref) this required one allocation.
"""
struct ConstrainedSetObjective{
    E<:AbstractEvaluationType,MO<:AbstractManifoldObjective,PF,IF
} <: AbstractManifoldObjective{E}
    objective::MO
    project!!::PF
    indicator::IF
end

function ConstrainedSetObjective(
    f, grad_f, project!!::PF; evaluation::E=AllocatingEvaluation(), indicator=nothing
) where {PF,E<:AbstractEvaluationType}
    obj = ManifoldGradientObjective(f, grad_f; evaluation=evaluation)
    if isnothing(indicator)
        if evaluation isa AllocatingEvaluation
            ind(M, p) = (distance(M, p, project!!(M, p)) ≈ 0 ? 0 : Inf)
            return ConstrainedSetObjective{E,typeof(obj),typeof(project!!),typeof(ind)}(
                obj, project!!, ind
            )
        elseif evaluation isa InplaceEvaluation
            ind = function (M, p)
                q = rand(M)
                project!!(M, q, p)
                return distance(M, p, q) ≈ 0 ? 0 : Inf
            end
            return ConstrainedSetObjective{E,typeof(obj),typeof(project!!),typeof(ind)}(
                obj, project!!, ind
            )
        end
    end
    return ConstrainedSetObjective{E,typeof(obj),typeof(project!!),typeof(indicator)}(
        obj, project!!, indicator
    )
end

function get_cost(M::AbstractManifold, cso::ConstrainedSetObjective, p)
    return get_cost(M, cso.objective, p)
end
function get_cost_function(cso::ConstrainedSetObjective, recursive=false)
    return get_cost_function(cso.objective)
end
function get_gradient_function(cso::ConstrainedSetObjective, recursive=false)
    return get_gradient_function(cso.objective)
end
function get_gradient(M::AbstractManifold, cso::ConstrainedSetObjective, p)
    return get_gradient(M, cso.objective, p)
end
function get_gradient!(M::AbstractManifold, X, cso::ConstrainedSetObjective, p)
    return get_gradient!(M, X, cso.objective, p)
end

_doc_get_projected_point = """
    get_projected_point(amp::AbstractManoptProblem, p)
    get_projected_point!(amp::AbstractManoptProblem, q, p)
    get_projected_point(M::AbstractManifold, cso::ConstrainedSetObjective, p)
    get_projected_point!(M::AbstractManifold, q, cso::ConstrainedSetObjective, p)

Project `p` with the projection that is stored within the [`ConstrainedSetObjective`](@ref).
This can be done in-place of `q`.
"""

@doc "$(_doc_get_projected_point)"
function get_projected_point(amp::AbstractManoptProblem, p)
    return get_projected_point(get_manifold(amp), get_objective(amp), p)
end
@doc "$(_doc_get_projected_point)"
function get_projected_point!(amp::AbstractManoptProblem, q, p)
    return get_projected_point!(get_manifold(amp), q, get_objective(amp), p)
end

@doc "$(_doc_get_projected_point)"
get_projected_point(M::AbstractManifold, cso::ConstrainedSetObjective, p)
function get_projected_point(
    M::AbstractManifold, cso::ConstrainedSetObjective{AllocatingEvaluation}, p
)
    return cso.project!!(M, p)
end
function get_projected_point(
    M::AbstractManifold, cso::ConstrainedSetObjective{InplaceEvaluation}, p
)
    q = copy(M, p)
    cso.project!!(M, q, p)
    return q
end
@doc "$(_doc_get_projected_point)"
get_projected_point!(M::AbstractManifold, q, cso::ConstrainedSetObjective, p)
function get_projected_point!(
    M::AbstractManifold, q, cso::ConstrainedSetObjective{AllocatingEvaluation}, p
)
    copyto!(M, q, cso.project!!(M, p))
    return q
end
function get_projected_point!(
    M::AbstractManifold, q, cso::ConstrainedSetObjective{InplaceEvaluation}, p
)
    cso.project!!(M, q, p)
    return q
end
