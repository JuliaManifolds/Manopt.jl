using Manopt, Manifolds, LinearAlgebra, Test, Random
Random.seed!(42)

@testset "Riemannian quasi-Newton Methods" begin
    # Mean of 3 matrices
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
        M, F, ∇F, x; stopping_criterion=StopWhenGradientNormLess(10^(-6))
    )

    x = [0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]

    x_clrbfgs = quasi_Newton(
        M, F, ∇F, x; cautious_update = true, stopping_criterion=StopWhenGradientNormLess(10^(-6))
    )

    @test norm(x_lrbfgs - x_solution) ≈ 0 atol = 10.0^(-14)
    @test norm(x_clrbfgs - x_solution) ≈ 0 atol = 10.0^(-14)

    for T in [InverseBFGS(), BFGS(), InverseDFP(), DFP(), InverseSR1(), SR1()], c in [true, false]
        x = [0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
        x_direction = quasi_Newton(
            M,
            F,
            ∇F,
            x;
            direction_update=T,
            cautious_update = c,
            memory_size=-1,
            stopping_criterion=StopWhenGradientNormLess(10^(-12)),
        )
        @test norm(x_direction - x_solution) ≈ 0 atol = 10.0^(-14)
    end
   

    

    # Rayleigh Quotient minimization
    n = 30
    A_Ray = randn(n, n)
    A_Ray = (A_Ray + A_Ray') / 2
    F_Ray(X::Array{Float64,1}) = X' * A_Ray * X
    ∇F_Ray(X::Array{Float64,1}) = 2 * (A_Ray * X - X * X' * A_Ray * X)
    M_Ray = Sphere(n - 1)
    x_solution_Ray = abs.(eigvecs(A_Ray)[:, 1])

    x_Ray = random_point(M_Ray)

    x_lrbfgs_Ray = quasi_Newton(
        M_Ray, F_Ray, ∇F_Ray, x_Ray; stopping_criterion=StopWhenGradientNormLess(10^(-12))
    )

    x_Ray = random_point(M_Ray)

    x_clrbfgs_Ray = quasi_Newton(
        M_Ray,
        F_Ray,
        ∇F_Ray,
        x_Ray;
        cautious_update=true,
        stopping_criterion=StopWhenGradientNormLess(10^(-12)),
    )


    @test norm(abs.(x_lrbfgs_Ray) - x_solution_Ray) ≈ 0 atol = 2e-13

    @test norm(abs.(x_clrbfgs_Ray) - x_solution_Ray) ≈ 0 atol = 2e-13
 
    for T in [InverseBFGS(), BFGS()], c in [true, false]
        x_Ray = random_point(M_Ray)
        x_direction_Ray = quasi_Newton(
            M_Ray,
            F_Ray,
            ∇F_Ray,
            x_Ray;
            direction_update = T,
            cautious_update = c,
            memory_size=-1,
            stopping_criterion=StopWhenGradientNormLess(10^(-12)),
        )
        @test norm(abs.(x_direction_Ray) - x_solution_Ray) ≈ 0 atol = 2e-13
    end

    x_Ray = random_point(M_Ray)

    x_inverseDFP_Ray = quasi_Newton(
        M_Ray,
        F_Ray,
        ∇F_Ray,
        x_Ray;
        direction_update = InverseDFP(),
        memory_size=-1,
        stopping_criterion=StopWhenGradientNormLess(10^(-12)),
    )

    x_Ray = random_point(M_Ray)

    x_directDFP_Ray = quasi_Newton(
        M_Ray,
        F_Ray,
        ∇F_Ray,
        x_Ray;
        direction_update = DFP(),
        memory_size=-1,
        stopping_criterion=StopWhenGradientNormLess(10^(-12)),
    )

    x_Ray = random_point(M_Ray)

    x_inverseSR1_Ray = quasi_Newton(
        M_Ray,
        F_Ray,
        ∇F_Ray,
        x_Ray;
        direction_update = InverseSR1(),
        memory_size=-1,
        stopping_criterion=StopWhenGradientNormLess(10^(-12)),
    )

    x_Ray = random_point(M_Ray)

    x_directSR1_Ray = quasi_Newton(
        M_Ray,
        F_Ray,
        ∇F_Ray,
        x_Ray;
        direction_update = SR1(),
        memory_size=-1,
        stopping_criterion=StopWhenGradientNormLess(10^(-12)),
    )

    @test norm(abs.(x_inverseDFP_Ray) - x_solution_Ray) ≈ 0 atol = 2e-13
    @test norm(abs.(x_directDFP_Ray) - x_solution_Ray) ≈ 0 atol = 2e-13
    @test norm(abs.(x_inverseSR1_Ray) - x_solution_Ray) ≈ 0 atol = 2e-13
    @test norm(abs.(x_directSR1_Ray) - x_solution_Ray) ≈ 0 atol = 2e-13
end
