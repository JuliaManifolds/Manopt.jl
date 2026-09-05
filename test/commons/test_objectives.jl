using Manifolds, Manopt, Test

@testset "Common Objectives" begin
    @testset "Immutable Variables wrapper for cost functions" begin
        f(::Circle, p) = p^2
        M = Circle()
        p0 = 1.0
        o = ManifoldCostObjective(f; p = p0)
        p0m = Manopt.maybe_wrap_variable(p0) # turns into a vector
        # Internally we now work on p0m, but we can verify against classical f above.
        @test get_cost(M, o, p0m) == f(M, p0)
    end
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
        mp = DefaultManoptProblem(Euclidean(2), o)
        Manopt.set_parameter!(mp, :Objective, :Dummy, 1)
    end
    @testset "functions" begin
        M = Euclidean(2)
        p = [1.0, 2.0]
        X = [3.0, 4.0]
        oa = ManifoldHessianObjective((M, p) -> p[1], (M, p) -> p, (M, p, X) -> X)
        @test Manopt.get_cost_function(oa)(M, p) == p[1]
        Y = zero_vector(M, p)
        @test Manopt.get_gradient_function(oa)(M, p) == p
        @test Manopt.get_hessian_function(oa)(M, p, X) == X
        oi = ManifoldHessianObjective(
            (M, p) -> p[1], (M, X, p) -> (X .= p), (M, Y, p, X) -> (Y .= X);
            evaluation = InplaceEvaluation(),
        )
        @test Manopt.get_cost_function(oi)(M, p) == p[1]
        @test Manopt.get_gradient_function(oi)(M, p) == p
        @test Manopt.get_hessian_function(oi)(M, p, X) == X
    end
end

@testset "First order objective aliases" begin
    M = ManifoldsBase.DefaultManifold(2)
    f(M, p) = sum(p .^ 2)
    grad_f(M, p) = 2 .* p
    costgrad_f(M, p) = (f(M, p), grad_f(M, p))
    df(M, p, X) = 2 * sum(p .* X)
    # every constructor combination must be matched by the alias it is named after
    @test ManifoldGradientObjective(f, grad_f) isa ManifoldGradientObjective
    @test ManifoldGradientObjective(f, grad_f; differential = df) isa ManifoldGradientObjective
    @test ManifoldCostGradientObjective(costgrad_f) isa ManifoldCostGradientObjective
    @test ManifoldCostGradientObjective(costgrad_f; differential = df) isa
        ManifoldCostGradientObjective
    # and by no other one
    @test !(ManifoldGradientObjective(f, grad_f) isa ManifoldCostGradientObjective)
    @test !(ManifoldCostGradientObjective(costgrad_f) isa ManifoldGradientObjective)
end
