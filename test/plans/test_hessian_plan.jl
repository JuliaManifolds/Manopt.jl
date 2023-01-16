using Manopt, Manifolds, Test

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
end
