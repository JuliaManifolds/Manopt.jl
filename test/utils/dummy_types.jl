using Manopt
import Manopt: get_iterate, set_manopt_parameter!
struct DummyDecoratedObjective{E,O<:AbstractManifoldObjective} <:
       Manopt.AbstractDecoratedManifoldObjective{E,O}
    objective::O
end
function DummyDecoratedObjective(
    o::O
) where {E<:AbstractEvaluationType,O<:AbstractManifoldObjective{E}}
    return DummyDecoratedObjective{E,O}(o)
end

struct DummyStateProblem{M<:AbstractManifold} <: AbstractManoptProblem{M} end
mutable struct DummyState <: AbstractManoptSolverState
    storage::Vector{Float64}
end
DummyState() = DummyState([])
get_iterate(::DummyState) = NaN
set_manopt_parameter!(s::DummyState, ::Val, v) = s
