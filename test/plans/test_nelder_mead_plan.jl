using Manifolds, Manopt, Test

@testset "Nelder Mead" begin
    @testset "Nelder Mead State" begin
        M = Euclidean(2)
        o = NelderMeadState(M)
        o2 = NelderMeadState(M; population = o.population)
        @test o.p == o2.p
        @test o.population == o2.population
        @test get_state(o) == o
        p = [1.0, 1.0]
        set_iterate!(o, M, p)
        @test get_iterate(o) == p

        @testset "StopWhenPopulationConcentrated" begin
            f(M, p) = norm(p)
            obj = ManifoldCostObjective(f)
            mp = DefaultManoptProblem(M, obj)
            s = StopWhenPopulationConcentrated(0.1, 0.1)
            # tweak an iteration
            o.costs = [0.0, 0.1]
            @test !s(mp, o, 1)
            @test get_reason(s) == ""
            s.value_f = 0.05
            s.value_p = 0.05
            s.at_iteration = 2
            @test length(get_reason(s)) > 0
        end
    end
end
