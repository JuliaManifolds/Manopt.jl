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
        "EmbeddedManifoldObjective{Missing,Missing} of an $(repr(eo.objective))"
end
