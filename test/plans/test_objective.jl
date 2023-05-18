using Manopt, Test
import Manopt: dispatch_objective_decorator

include("../utils/dummy_types.jl")

@testset "Test decorator" begin
    o = ManifoldCostObjective(x -> x)
    d = DummyDecoratedObjective(o)
    @test (get_objective(d) isa ManifoldCostObjective)
    @test Manopt.is_objective_decorator(d)
    @test !Manopt.is_objective_decorator(o)
end
