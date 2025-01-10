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
* `project::PF` a projection function ``$(_tex(:proj))_{$(_tex(:Cal,"C"))}: $(_math(:M)) → $(_tex(:Cal,"C"))`` that projects onto the set ``$(_tex(:Cal,"C"))``.
* `indicator::IF` the indicator function ``ι_{$(_tex(:Cal,"C"))}(p) = $(_tex(:cases, "0 &"*_tex(:text, " for ")*"p∈"*_tex(:Cal,"C"), "∞ &"*_tex(:text, " else.")))

# Constructor

    ConstrainedSetObjective(f, grad_f, project; kwargs...)

Generate the constrained objective for a given function `f` its gradient `grad_f` and a `project`ion ``$(_tex(:proj))_{$(_tex(:Cal,"C"))}``.

## Keyword arguments

$(_var(:Keyword, :evaluation))
* `indicator=nothing`: the indicator function ``ι_{$(_tex(:Cal,"C"))}(p)``. If not provided a test, whether the projection yields the same point is performed.
  For the [`InplaceEvaluation`](@ref) this required one allocation.
"""
struct ConstrainedSetObjective{
    E<:AbstractEvaluationType,MO<:AbstractManifoldObjective,PF,IF
} <: AbstractManifoldObjective{E}
    objective::MO
    project::PF
    indicator::IF
end

function ConstrainedSetObjective(
    f, grad_f, project::PF; evaluation::E=AllocatingEvaluation(), indicator=nothing
) where {PF,E<:AbstractEvaluationType}
    obj = ManifoldGradientObjective(f, grad_f; evaluation=evaluation)
    if isnothing(indicator)
        if evaluation isa AllocatingEvaluation
            ind(M, p) = (distance(M, p, project(M, p)) ≈ 0 ? 0 : Inf)
            return ConstrainedSetObjective{E,typeof(obj),typeof(project),typeof(ind)}(
                obj, project, ind
            )
        elseif evaluation isa InplaceEvaluation
            ind = function (M, p)
                q = rand(M)
                project(M, q, p)
                return distance(M, p, q) ≈ 0 ? 0 : Inf
            end
            return ConstrainedSetObjective{E,typeof(obj),typeof(project),typeof(ind)}(
                obj, project, ind
            )
        end
    end
    return ConstrainedSetObjective{E,typeof(obj),typeof(project),typeof(indicator)}(
        obj, project, indicator
    )
end
