using Manopt, Manifolds, LinearAlgebra, Test, Random
Random.seed!(42)

@testset "Riemannian quasi-Newton Methods" begin
    A = [18.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
    B = [0.0 0.0 0.0 0.009; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
    C = [0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; -5.0 0.0 0.0 0.0]
    ABC = [A, B, C]
    x_solution = mean(ABC)
    F(x) = 0.5 * norm(A - x)^2 + 0.5 * norm(B - x)^2 + 0.5 * norm(C - x)^2
    ∇F(x) = -A - B - C + 3 * x
    M = Euclidean(4, 4)
    x = [0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
    x_lrbfgs = quasi_Newton(
        M,
        F,
        ∇F,
        x;
        stopping_criterion=StopWhenGradientNormLess(10^(-6))
        )
    x_clrbfgs = quasi_Newton(
        M,
        F,
        ∇F,
        x;
        cautious_update = true,
        stopping_criterion=StopWhenGradientNormLess(10^(-6))
        )
    x_rbfgs = quasi_Newton(
        M,
        F,
        ∇F,
        x;
        memory_size=-1,
        stopping_criterion=StopWhenGradientNormLess(10^(-6))
        )
    x_crbfgs = quasi_Newton(
        M,
        F,
        ∇F,
        x;
        memory_size=-1,
        cautious_update = true,
        stopping_criterion=StopWhenGradientNormLess(10^(-6))
        )

    @test norm(x_lrbfgs - x_solution) ≈ 0 atol = 10.0^(-14)
    @test norm(x_clrbfgs - x_solution) ≈ 0 atol = 10.0^(-14)
    @test norm(x_rbfgs - x_solution) ≈ 0 atol = 10.0^(-14)
    @test norm(x_crbfgs - x_solution) ≈ 0 atol = 10.0^(-14)
    

    
end
