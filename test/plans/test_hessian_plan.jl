using Manopt, Manifolds, Test, Random

@testset "Hessian access functions" begin
    M = Euclidean(2)
    F(M, p) = 1
    gradF(M, p) = zero(2)
    gradF!(M, X, p) = copyto!(M, X, p, zeros(2))
    HessF(M, p, X) = X
    HessF!(M, Y, p, X) = copyto!(M, Y, p, X)
    precon = (M, p, X) -> X
    P1 = HessianProblem(M, F, gradF, HessF, precon)
    P2 = HessianProblem(M, F, gradF!, HessF!, precon; evaluation=MutatingEvaluation())

    p = zeros(2)
    X = ones(2)
    @test get_hessian(P1, p, X) == get_hessian(P2, p, X)
    Y1 = similar(X)
    Y2 = similar(X)
    get_hessian!(P1, Y1, p, X)
    get_hessian!(P2, Y2, p, X)
    @test Y1 == Y2
end
