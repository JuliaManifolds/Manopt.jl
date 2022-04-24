using Manopt, ManifoldsBase, Test

@testset "Problem Test" begin
    @testset "Update functions" begin
        M = ManifoldsBase.DefaultManifold(2)
        p = zeros(2)
        X = zeros(2)

        f(M, p) = 1
        f2(M, p) = 2

        P1 = CostProblem(M, f)
        @test get_cost(P1, p) == 1
        update_cost!(P1, f2)
        @test get_cost(P1, p) == 2

        g(M, p) = 3 * ones(p)
        g!(M, X, p) = (X .= 3)

        g2(M, p) = 4 * ones(p)
        g2!(M, X, p) = (X .= 4)

        P2a = GradientProblem(M, f, g)
        @test get_gradient(P2a, p) == 3 .* ones(p)
        update_gradient!(P2a, g2)
        @test get_gradient(P2a, p) == 4 .* ones(p)

        P2b = GradientProblem(M, f, g!; evaluation=MutatingEvaluation())
        @test get_gradient(P2b, p) == 3 .* ones(p)
        update_gradient!(P2a, g2!)
        @test get_gradient(P2b, p) == 4 .* ones(p)

        h(M, p, X) = 5 .* ones(p)
        h!(M, Y, p, X) = (Y .= 5)

        h2(M, p) = 6 .* ones(p)
        h2!(M, X, p) = (Y .= 6)

        P3a = HessianProblem(M, f, g, h)
        @test get_hessian(P3a, p, X) == 5 .* ones(p)
        update_hessian!(P3a, h2)
        @test get_hessian(P3a, p, X) == 6 .* ones(p)

        P3b = HessianProblem(M, f, g!, h!; evaluation=MutatingEvaluation())
        @test get_hessian(P3a, p, X) == 5 .* ones(p)
        update_hessian!(P3a, h2!)
        @test get_hessian(P3a, p, X) == 6 .* ones(p)
    end
end
