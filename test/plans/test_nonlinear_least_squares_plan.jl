using Manifolds, Manopt, Test

@testset "Nonlinear lest squares plane" begin
    @testset "Test cost/residual/jacobian cases with smoothing" begin end
    @testset "Smootthing factory" begin
        s1 = Manopt.smoothing_factory(:Identity)
        @test s1 isa ManifoldHessianObjective

        s2 = Manopt.smoothing_factory((:Identity, 2))
        @test s2 isa VectorHessianFunction
        @test length(s2) == 2

        s3 = Manopt.smoothing_factory((:Identity, 3.0))
        @test s3 isa ManifoldHessianObjective

        for s in [:Arctan, :Cauchy, :Huber, :SoftL1, :Tukey]
            s4 = Manopt.smoothing_factory(s)
            @test s4 isa ManifoldHessianObjective
        end

        s5 = Manopt.smoothing_factory(((:Identity, 2), (:Huber, 3)))
        @test s5 isa VectorHessianFunction
        @test length(s5) == 5
    end
end
