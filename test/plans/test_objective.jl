using Manopt, Test
import Manopt: dispatch_objective_decorator

struct DummyDecoObjective{O<:AbstractManifoldObjective} <:
       AbstractManifoldObjective{AllocatingEvaluation}
    objective::O
end
dispatch_objective_decorator(::DummyDecoObjective) = Val(true)

@testset "Test decorator" begin
    d = DummyDecoObjective(ManifoldCostObjective(x -> x))
    @test (get_objective(d) isa ManifoldCostObjective)
end
