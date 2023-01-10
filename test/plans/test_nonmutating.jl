#
# Test nonmutating specials
#

using Manifolds, Manopt, Test

@testset "Nonmutating special implementations" begin
    mho = ManifoldHessianObjective(p -> 1, (M, x) -> p, (M, p, X) -> p + X, (M, p, X) -> X)
    mp = DefaultManoptProblem(Euclidean(), mho)
    Y = -2.0
    Y = get_hessian!(mp, Y, 1.0, 2.0)
    @test Y == 3.0
end
