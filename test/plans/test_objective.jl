s = joinpath(@__DIR__, "..", "ManoptTestSuite.jl")
!(s in LOAD_PATH) && (push!(LOAD_PATH, s))

using ManifoldsBase, Manopt, ManoptTestSuite, Test

@testset "Objective" begin
    @testset "Test decorator" begin
        o = ManifoldCostObjective(x -> x)
        d = ManoptTestSuite.DummyDecoratedObjective(o)
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
        d = ManoptTestSuite.DummyDecoratedObjective(o)
        r2 = Manopt.ReturnManifoldObjective(d)
        @test repr(r) == "ManifoldCostObjective{AllocatingEvaluation}"
    end
    @testset "set_parameter!" begin
        o = ManifoldCostObjective(x -> x)
        mp = DefaultManoptProblem(ManifoldsBase.DefaultManifold(2), o)
        Manopt.set_parameter!(mp, :Objective, :Dummy, 1)
    end
    @testset "functions" begin
        M = ManifoldsBase.DefaultManifold(2)
        p = [1.0, 2.0]
        X = [3.0, 4.0]
        oa = ManifoldHessianObjective((M, p) -> p[1], (M, p) -> p, (M, p, X) -> X)
        @test Manopt.get_cost_function(oa)(M, p) == p[1]
        @test Manopt.get_gradient_function(oa)(M, p) == p
        @test Manopt.get_hessian_function(oa)(M, p, X) == X
        oi = ManifoldHessianObjective(
            (M, p) -> p[1],
            (M, X, p) -> (X .= p),
            (M, Y, p, X) -> (Y .= X);
            evaluation=InplaceEvaluation(),
        )
        @test Manopt.get_cost_function(oi)(M, p) == p[1]
        Y = similar(X)
        @test Manopt.get_gradient_function(oi)(M, Y, p) == p
        @test Y == p
        @test Manopt.get_hessian_function(oi)(M, Y, p, X) == X
        @test Y == X
    end
end
