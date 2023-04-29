using Manopt, Test
import Manopt: dispatch_objective_decorator

struct DummyDecoObjective{O<:AbstractManifoldObjective} <:
       AbstractManifoldObjective{AllocatingEvaluation}
    objective::O
end
dispatch_objective_decorator(::DummyDecoObjective) = Val(true)

@testset "Test decorator" begin
    o = ManifoldCostObjective(x -> x)
    d = DummyDecoObjective(o)
    @test (get_objective(d) isa ManifoldCostObjective)
    @test Manopt.is_objective_decorator(d)
    @test !Manopt.is_objective_decorator(o)
end
