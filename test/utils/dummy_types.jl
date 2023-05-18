using Manopt
struct DummyDecoratedObjective{E,O<:AbstractManifoldObjective} <:
       Manopt.AbstractDecoratedManifoldObjective{E,O}
    objective::O
end
function DummyDecoratedObjective(
    o::O
) where {E<:AbstractEvaluationType,O<:AbstractManifoldObjective{E}}
    return DummyDecoratedObjective{E,O}(o)
end
