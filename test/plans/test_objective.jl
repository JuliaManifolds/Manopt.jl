using ManifoldsBase, Manopt, Test

@testset "Objective" begin
    @testset "Test decorator" begin
        o = ManifoldCostObjective(x -> x)
        d = Manopt.Test.DummyDecoratedObjective(o)
        @test (get_objective(d) isa ManifoldCostObjective)
        @test Manopt.is_objective_decorator(d)
        @test !Manopt.is_objective_decorator(o)
        io = IOBuffer()
        show(io, MIME"text/plain"(), o)
        @test startswith(String(take!(io)), "A cost function on a Riemannian manifold")
        d = Manopt.Test.DummyEmptyDecoratedObjective(o)
        # Check both default pass throughs
        Manopt.status_summary(io, d)
        @test startswith(String(take!(io)), "A cost function on a Riemannian manifold")
        @test startswith(Manopt.status_summary(d), "A cost function on a Riemannian manifold")
    end
    @testset "ReturnManifoldObjective" begin
        f(x) = x
        o = ManifoldCostObjective(f)
        r = Manopt.ReturnManifoldObjective(o)
        @test repr(o) == "ManifoldCostObjective($(f))"
        @test repr(r) == "ReturnManifoldObjective(ManifoldCostObjective($(f)))"
        @test Manopt.status_summary(o) == "A cost function on a Riemannian manifold `f = (M,p) -> ℝ`."
        @test Manopt.status_summary(r) == "A cost function on a Riemannian manifold `f = (M,p) -> ℝ`."
        d = Manopt.Test.DummyDecoratedObjective(o)
        r2 = Manopt.ReturnManifoldObjective(d)
        # Still acts transparent for one of them
        @test Manopt.status_summary(r2) == "A dummy decorator for A cost function on a Riemannian manifold `f = (M,p) -> ℝ`."
        # repr contains all is much longer
        @test repr(r2) == "ReturnManifoldObjective(DummyDecoratedObjective($(repr(o))))"
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
            (M, p) -> p[1], (M, X, p) -> (X .= p), (M, Y, p, X) -> (Y .= X);
            evaluation = InplaceEvaluation(),
        )
        @test Manopt.get_cost_function(oi)(M, p) == p[1]
        Y = similar(X)
        @test Manopt.get_gradient_function(oi)(M, Y, p) == p
        @test Y == p
        @test Manopt.get_hessian_function(oi)(M, Y, p, X) == X
        @test Y == X
        @test Manopt._to_kw(Manopt.ParentEvaluationType) == "evaluation = ParentEvaluationType()"
        @test Manopt._to_kw(Manopt.AllocatingInplaceEvaluation) == "evaluation = AllocatingInplaceEvaluation()"
    end
end
