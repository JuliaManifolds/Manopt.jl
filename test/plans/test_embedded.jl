using Manifolds, Manopt, Test, LinearAlgebra, Random

@testset "Test Embedding accessors and conversion" begin
    @testset "Basics" begin
        n = 5
        k = 2
        E = ℝ^(5, 2)
        M = Grassmann(n, k)
        A = diagm([1.0, 2.0, 3.0, 4.0, 5.0])
        f(M, p) = 1 / 2 * tr(p' * A * p)
        ∇f(M, p) = A * p
        ∇²f(M, p, X) = A * X
        grad_f(M, p) = A * p - p * (p' * A * p)
        Hess_f(M, p, X) = A * X - p * p' * A * X - X * p' * A * p
        o = ManifoldHessianObjective(f, ∇f, ∇²f)
        p = [1.0 0.0; 0.0 1.0; 0.0 0.0; 0.0 0.0; 0.0 0.0]
        X = [0.0 0.0; 0.0 0.0; 1.0 0.0; 0.0 1.0; 0.0 0.0]

        # With interim caches for p and X
        eo1 = Manopt.decorate_objective!(
            M, o; objective_type = :Euclidean, embedded_p = copy(p), embedded_X = copy(X)
        )
        eo2 = EmbeddedManifoldObjective(o, missing, copy(X))
        eo3 = EmbeddedManifoldObjective(o, copy(p), missing)
        eo4 = EmbeddedManifoldObjective(o)

        for eo in [eo1, eo2, eo3, eo4]
            @testset "$(split(repr(eo), " ")[1])" begin
                @test get_cost(M, eo, p) == f(E, p)
                @test get_gradient(E, o, p) == ∇f(E, p)
                @test get_gradient(M, eo, p) == grad_f(M, p)
                Y = zero_vector(M, p)
                get_gradient!(M, Y, eo, p)
                @test Y == grad_f(M, p)
                @test get_hessian(M, o, p, X) == ∇²f(M, p, X)
                @test get_hessian(M, eo, p, X) == Hess_f(M, p, X)
                get_hessian!(M, Y, eo, p, X)
                @test Y == Hess_f(M, p, X)
            end
        end
        # Without interim caches for p and X
        @test repr(eo4) ==
            "EmbeddedManifoldObjective(ManifoldHessianObjective(f, ∇f, ∇²f, #483; evaluation = AllocatingEvaluation()), missing, missing)"

        # Constraints, though this is not the most practical constraint
        o2 = ConstrainedManifoldObjective(f, ∇f, [f], [∇f], [f], [∇f])
        eco1 = EmbeddedManifoldObjective(M, o2)
        eco2 = EmbeddedManifoldObjective(o2, missing, copy(X))
        eco3 = EmbeddedManifoldObjective(o2, copy(p), missing)
        eco4 = EmbeddedManifoldObjective(o2)
        for eco in [eco1, eco2, eco3, eco4]
            @testset "$(split(repr(eco), " ")[1])" begin
                @test get_constraints(M, eco, p) == [[f(E, p)], [f(E, p)]]
                @test get_equality_constraint(M, eco, p, :) == [f(E, p)]
                @test get_equality_constraint(M, eco, p, 1) == f(E, p)
                @test get_inequality_constraint(M, eco, p, :) == [f(E, p)]
                @test get_inequality_constraint(M, eco, p, 1) == f(E, p)
                @test get_grad_equality_constraint(M, eco, p, :) == [grad_f(M, p)]
                Z = [zero_vector(M, p)]
                get_grad_equality_constraint!(M, Z, eco, p, :)
                @test Z == [grad_f(M, p)]
                @test get_grad_equality_constraint(M, eco, p, 1) == grad_f(M, p)
                Y = zero_vector(M, p)
                get_grad_equality_constraint!(M, Y, eco, p, 1)
                @test Y == grad_f(M, p)
                @test get_grad_inequality_constraint(M, eco, p, :) == [grad_f(M, p)]
                Z = [zero_vector(M, p)]
                get_grad_inequality_constraint!(M, Z, eco, p, :)
                @test Z == [grad_f(M, p)]
                @test get_grad_inequality_constraint(M, eco, p, 1) == grad_f(M, p)
                get_grad_inequality_constraint!(M, Y, eco, p, 1)
                @test Y == grad_f(M, p)
            end
        end
        # just verify that this also works for double decorated ones.
        o3 = EmbeddedManifoldObjective(ManifoldCountObjective(M, o, [:Cost]), p, X)
    end
    @testset "Function passthrough" begin
        Random.seed!(42)
        n = 4
        A = Symmetric(randn(n, n))
        M = Sphere(n - 1)
        p = [1.0, zeros(n - 1)...]
        X = [0.0, 1.0, zeros(n - 2)...]
        f(M, p) = 0.5 * p' * A * p
        ∇f(M, p) = A * p
        ∇²f(M, p, X) = A * X
        grad_f(M, p) = A * p - (p' * A * p) * p
        Hess_f(M, p, X) = A * X - (p' * A * X) .* p - (p' * A * p) .* X
        obj = ManifoldHessianObjective(f, ∇f, ∇²f)
        e_obj = EmbeddedManifoldObjective(obj)
        # undecorated / recursive cost -> exactly f
        @test Manopt.get_cost_function(obj) === Manopt.get_cost_function(e_obj, true)
        # otherwise different
        f1 = Manopt.get_cost_function(e_obj)
        @test f1 != f
        @test f1(M, p) == f(M, p)
        # The same for gradient
        @test Manopt.get_gradient_function(obj) ===
            Manopt.get_gradient_function(e_obj, true)
        grad_f1 = Manopt.get_gradient_function(e_obj)
        @test grad_f1 != grad_f
        @test grad_f1(M, p) == grad_f(M, p)
        # And Hessian
        @test Manopt.get_hessian_function(obj) === Manopt.get_hessian_function(e_obj, true)
        Hess_f1 = Manopt.get_hessian_function(e_obj)
        @test Hess_f1 != Hess_f
        @test Hess_f1(M, p, X) == Hess_f(M, p, X)
        #
        # And all three for mutating again
        ∇f!(M, X, p) = (X .= A * p)
        ∇²f!(M, Y, p, X) = (Y .= A * X)
        grad_f!(M, X, p) = (X .= A * p - (p' * A * p) * p)
        Hess_f!(M, Y, p, X) = (Y .= A * X - (p' * A * X) .* p - (p' * A * p) .* X)
        obj_i = ManifoldHessianObjective(f, ∇f!, ∇²f!; evaluation = InplaceEvaluation())
        e_obj_i = EmbeddedManifoldObjective(obj_i)
        @test Manopt.get_cost_function(obj_i) === Manopt.get_cost_function(e_obj_i, true)
        f2 = Manopt.get_cost_function(e_obj_i)
        @test f2 != f
        @test f2(M, p) == f(M, p)
        # The same for gradient
        @test Manopt.get_gradient_function(obj_i) ===
            Manopt.get_gradient_function(e_obj_i, true)
        grad_f1! = Manopt.get_gradient_function(e_obj_i)
        @test grad_f1! != grad_f!
        Y = similar(X)
        Z = similar(X)
        @test grad_f1!(M, Y, p) == grad_f!(M, Z, p)
        # And Hessian
        @test Manopt.get_hessian_function(obj_i) ===
            Manopt.get_hessian_function(e_obj_i, true)
        Hess_f1! = Manopt.get_hessian_function(e_obj_i)
        @test Hess_f1 != Hess_f
        @test Hess_f1!(M, Y, p, X) == Hess_f!(M, Z, p, X)
    end
end
