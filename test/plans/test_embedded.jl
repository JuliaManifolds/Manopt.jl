using Manifolds, Manopt, Test, LinearAlgebra

@testset "Test Embedding accessors and conversion" begin
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

    # With interims caches for p and X
    eo1 = Manopt.decorate_objective!(
        M, o; objective_type=:Euclidean, embedded_p=copy(p), embedded_X=copy(X)
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
            get_hessian(M, eo, p, X) == Hess_f(M, p, X)
            get_hessian!(M, Y, eo, p, X)
            @test Y == Hess_f(M, p, X)
        end
    end
    # Without interims caches for p and X
    @test repr(eo4) ==
        "EmbeddedManifoldObjective{Missing,Missing} of an $(repr(eo4.objective))"

    # Constraints, though this is not the most practical constraint
    o2 = ConstrainedManifoldObjective(f, ∇f, [f], [∇f], [f], [∇f])
    eco1 = EmbeddedManifoldObjective(M, o2)
    eco2 = EmbeddedManifoldObjective(o2, missing, copy(X))
    eco3 = EmbeddedManifoldObjective(o2, copy(p), missing)
    eco4 = EmbeddedManifoldObjective(o2)

    for eco in [eco1, eco2, eco3, eco4]
        @testset "$(split(repr(eco), " ")[1])" begin
            @test get_constraints(M, eco, p) == [[f(E, p)], [f(E, p)]]
            @test get_equality_constraints(M, eco, p) == [f(E, p)]
            @test get_equality_constraint(M, eco, p, 1) == f(E, p)
            @test get_inequality_constraints(M, eco, p) == [f(E, p)]
            @test get_inequality_constraint(M, eco, p, 1) == f(E, p)
            @test get_grad_equality_constraints(M, eco, p) == [grad_f(M, p)]
            @test get_grad_equality_constraint(M, eco, p, 1) == grad_f(M, p)
            Y = zero_vector(M, p)
            get_grad_equality_constraint!(M, Y, eco, p, 1)
            @test Y == grad_f(M, p)
            @test get_grad_inequality_constraints(M, eco, p) == [grad_f(M, p)]
            @test get_grad_inequality_constraint(M, eco, p, 1) == grad_f(M, p)
            get_grad_inequality_constraint!(M, Y, eco, p, 1)
            @test Y == grad_f(M, p)
        end
    end
    o3 = EmbeddedManifoldObjective(ManifoldCountObjective(M, o, [:Cost]), p, X)
end
