#
# Test nonmutating specials
#
@testset "Nonmutating special implementations" begin
    p = HessianProblem(Euclidean(), x -> x, (M, x) -> x, (M, x, X) -> x + X, x -> x)
    Y = -2.0
    Y = get_hessian!(p, Y, 1.0, 2.0)
    @test Y == 3.0
end
