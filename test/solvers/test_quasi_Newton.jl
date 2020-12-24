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
        M,
        F,
        ∇F,
        x
        )
    x_clrbfgs = quasi_Newton(
        M,
        F,
        ∇F,
        x;
        cautious_update = true,
        )
    x_rbfgs = quasi_Newton(
        M,
        F,
        ∇F,
        x;
        memory_size=-1,
        )
    x_crbfgs = quasi_Newton(
        M,
        F,
        ∇F,
        x;
        memory_size=-1,
        cautious_update = true,
        )

    @test norm(x_lrbfgs - x_solution) ≈ 0 atol = 10.0^(-14)
    @test norm(x_clrbfgs - x_solution) ≈ 0 atol = 10.0^(-14)
    @test norm(x_rbfgs - x_solution) ≈ 0 atol = 10.0^(-14)
    @test norm(x_crbfgs - x_solution) ≈ 0 atol = 10.0^(-14)
    
    # Rayleigh Quotient minimization
    A_Ray = randn(300,300)
    A_Ray = (A_Ray + A_Ray') / 2
    F_Ray(X::Array{Float64,1}) = X' * A_Ray * X
    ∇F_Ray(X::Array{Float64,1}) = 2 * (A_Ray * X - X * X' * A_Ray * X)
    M_Ray = Sphere(299)
    x_Ray = random_point(M_Ray)
    x_solution_Ray = abs.(eigvecs(A_Ray)[:,1])

    x_lrbfgs_Ray = quasi_Newton(
        M_Ray,
        F_Ray,
        ∇F_Ray,
        x_Ray
        )
    x_clrbfgs_Ray = quasi_Newton(
        M_Ray,
        F_Ray,
        ∇F_Ray,
        x_Ray;
        cautious_update = true,
        )
    x_rbfgs_Ray = quasi_Newton(
        M_Ray,
        F_Ray,
        ∇F_Ray,
        x_Ray;
        memory_size=-1,
        )
    x_crbfgs_Ray = quasi_Newton(
       M_Ray,
        F_Ray,
        ∇F_Ray,
        x_Ray;
        memory_size=-1,
        cautious_update = true,
        )

    @test norm(abs.(x_lrbfgs_Ray) - x_solution_Ray) ≈ 0 atol = 10.0^(-14)
    @test norm(abs.(x_clrbfgs_Ray) - x_solution_Ray) ≈ 0 atol = 10.0^(-14)
    @test norm(abs.(x_rbfgs_Ray) - x_solution_Ray) ≈ 0 atol = 10.0^(-14)
    @test norm(abs.(x_crbfgs_Ray) - x_solution_Ray) ≈ 0 atol = 10.0^(-14)
end
