using ManifoldsBase, Manopt, Test

include("../utils/dummy_types.jl")

@testset "Objective" begin
    @testset "Test decorator" begin
        o = ManifoldCostObjective(x -> x)
        d = DummyDecoratedObjective(o)
        @test (get_objective(d) isa ManifoldCostObjective)
        @test Manopt.is_objective_decorator(d)
        @test !Manopt.is_objective_decorator(o)
    end
    @testset "ReturnObjective" begin
        o = ManifoldCostObjective(x -> x)
        r = Manopt.ReturnManifoldObjective(o)
        @test repr(o) == "ManifoldCostObjective{AllocatingEvaluation}"
        @test repr(r) == "ManifoldCostObjective{AllocatingEvaluation}"
        @test Manopt.status_summary(o) == "" # both simplified to empty
        @test Manopt.status_summary(r) == ""
        @test repr((o, 1.0)) ==
            "To access the solver result, call `get_solver_result` on this variable."
        d = DummyDecoratedObjective(o)
        r2 = Manopt.ReturnManifoldObjective(d)
        @test repr(r) == "ManifoldCostObjective{AllocatingEvaluation}"
    end
    @testset "set_manopt_parameter!" begin
        o = ManifoldCostObjective(x -> x)
        mp = DefaultManoptProblem(ManifoldsBase.DefaultManifold(2), o)
        set_manopt_parameter!(mp, :Objective, :Dummy, 1)
    end
end
