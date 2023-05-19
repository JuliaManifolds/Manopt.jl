using Manopt, Manifolds, Test
include("../utils/dummy_types.jl")

@testset "Hessian access functions" begin
    M = Euclidean(2)
    f(M, p) = 1
    grad_f(M, p) = zeros(2)
    grad_f!(M, X, p) = copyto!(M, X, p, zeros(2))
    Hess_f(M, p, X) = 0.5 * X
    Hess_f!(M, Y, p, X) = copyto!(M, Y, p, 0.5 * X)
    precon(M, p, X) = X
    precon!(M, Y, p, X) = copyto!(M, Y, p, X)

    mho1 = ManifoldHessianObjective(f, grad_f, Hess_f)
    mho2 = ManifoldHessianObjective(f, grad_f, Hess_f, precon)
    mho3 = ManifoldHessianObjective(f, grad_f!, Hess_f!; evaluation=InplaceEvaluation())
    mho4 = ManifoldHessianObjective(f, grad_f, Hess_f)

    p = zeros(2)
    X = ones(2)

    for mho in [mho1, mho2, mho3, mho4]
        mp = DefaultManoptProblem(M, mho)
        Y = similar(X)
        # Gradient
        @test get_gradient(mp, p) == zeros(2)
        get_gradient!(mp, Y, p)
        @test Y == zeros(2)
        # Hessian
        @test get_hessian(mp, p, X) == 0.5 * X
        get_hessian!(mp, Y, p, X)
        @test Y == 0.5 * X
        # precon
        @test get_preconditioner(mp, p, X) == X
        get_preconditioner!(mp, Y, p, X)
        @test Y == X
    end
    @testset "Objetive Decorator passthrough" begin
        Y1 = zero_vector(M, p)
        Y2 = zero_vector(M, p)
        for obj in [mho1, mho2, mho3, mho4]
            ddo = DummyDecoratedObjective(obj)
            @test get_hessian(M, obj, p, X) == get_hessian(M, ddo, p, X)
            get_hessian!(M, Y1, obj, p, X)
            get_hessian!(M, Y2, ddo, p, X)
            @test Y1 == Y2
            @test get_preconditioner(M, obj, p, X) == get_preconditioner(M, ddo, p, X)
            get_preconditioner!(M, Y1, obj, p, X)
            get_preconditioner!(M, Y2, ddo, p, X)
            @test Y1 == Y2
        end
    end
    @testset "Counting Objective" begin
        Y1 = zero_vector(M, p)
        Y2 = zero_vector(M, p)
        for obj in [mho1, mho2, mho3, mho4]
            cobj = Manopt.objective_count_factory(M, obj, [:Hessian, :Preconditioner])
            ddo = DummyDecoratedObjective(obj)
            @test get_hessian(M, obj, p, X) == get_hessian(M, cobj, p, X)
            get_hessian!(M, Y1, obj, p, X)
            get_hessian!(M, Y2, cobj, p, X)
            @test Y1 == Y2
            @test get_preconditioner(M, obj, p, X) == get_preconditioner(M, cobj, p, X)
            get_preconditioner!(M, Y1, obj, p, X)
            get_preconditioner!(M, Y2, cobj, p, X)
            @test Y1 == Y2
            @test get_count(cobj, :Hessian) == 2
            @test get_count(cobj, :Preconditioner) == 2
        end
    end
end
