using Manopt, Manifolds, LinearAlgebra, Test, Random
Random.seed!(42)

@testset "Riemannian quasi-Newton Methods" begin
    @testset "Mean of 3 Matrices" begin
        # Mean of 3 matrices
        A = [18.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
        B = [0.0 0.0 0.0 0.009; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
        C = [0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; -5.0 0.0 0.0 0.0]
        ABC = [A, B, C]
        x_solution = mean(ABC)
        F(x) = 0.5 * norm(A - x)^2 + 0.5 * norm(B - x)^2 + 0.5 * norm(C - x)^2
        ∇F(x) = -A - B - C + 3 * x
        M = Euclidean(4, 4)
        x = zeros(Float64, 4, 4)
        x_lrbfgs = quasi_Newton(
            M, F, ∇F, x; stopping_criterion=StopWhenGradientNormLess(10^(-6))
        )
        @test norm(x_lrbfgs - x_solution) ≈ 0 atol = 10.0^(-14)
        # with Options
        lrbfgs_o = quasi_Newton(
            M,
            F,
            ∇F,
            x;
            stopping_criterion=StopWhenGradientNormLess(10^(-6)),
            return_options=true,
        )
        @test lrbfgs_o.x == x_lrbfgs
        # with Cached Basis
        x = zeros(Float64, 4, 4)
        x_lrbfgs_cached = quasi_Newton(
            M,
            F,
            ∇F,
            x;
            stopping_criterion=StopWhenGradientNormLess(10^(-6)),
            basis=get_basis(M, x, DefaultOrthonormalBasis()),
        )
        @test x_lrbfgs_cached == x_lrbfgs

        x = zeros(Float64, 4, 4)
        x_clrbfgs = quasi_Newton(
            M,
            F,
            ∇F,
            x;
            cautious_update=true,
            stopping_criterion=StopWhenGradientNormLess(10^(-6)),
        )
        @test norm(x_clrbfgs - x_solution) ≈ 0 atol = 10.0^(-14)

        for T in [InverseBFGS(), BFGS(), InverseDFP(), DFP(), InverseSR1(), SR1()]
            for c in [true, false]
                x = zeros(Float64, 4, 4)
                x_direction = quasi_Newton(
                    M,
                    F,
                    ∇F,
                    x;
                    direction_update=T,
                    cautious_update=c,
                    memory_size=-1,
                    stopping_criterion=StopWhenGradientNormLess(10^(-12)),
                )
                @test norm(x_direction - x_solution) ≈ 0 atol = 10.0^(-14)
            end
        end
    end
    @testset "Rayleigh Quotient Minimzation" begin
        n = 30
        A_Ray = randn(n, n)
        A_Ray = (A_Ray + A_Ray') / 2
        F_Ray(X::Array{Float64,1}) = X' * A_Ray * X
        ∇F_Ray(X::Array{Float64,1}) = 2 * (A_Ray * X - X * X' * A_Ray * X)
        M_Ray = Sphere(n - 1)
        x_solution_Ray = abs.(eigvecs(A_Ray)[:, 1])

        x_Ray = random_point(M_Ray)

        x_lrbfgs_Ray = quasi_Newton(
            M_Ray,
            F_Ray,
            ∇F_Ray,
            x_Ray;
            stopping_criterion=StopWhenGradientNormLess(10^(-12)),
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
                direction_update=T,
                cautious_update=c,
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
            direction_update=InverseDFP(),
            memory_size=-1,
            stopping_criterion=StopWhenGradientNormLess(10^(-12)),
        )

        x_Ray = random_point(M_Ray)

        x_directDFP_Ray = quasi_Newton(
            M_Ray,
            F_Ray,
            ∇F_Ray,
            x_Ray;
            direction_update=DFP(),
            memory_size=-1,
            stopping_criterion=StopWhenGradientNormLess(10^(-12)),
        )

        x_Ray = random_point(M_Ray)

        x_inverseSR1_Ray = quasi_Newton(
            M_Ray,
            F_Ray,
            ∇F_Ray,
            x_Ray;
            direction_update=InverseSR1(),
            memory_size=-1,
            stopping_criterion=StopWhenGradientNormLess(10^(-12)),
        )

        x_Ray = random_point(M_Ray)

        x_directSR1_Ray = quasi_Newton(
            M_Ray,
            F_Ray,
            ∇F_Ray,
            x_Ray;
            direction_update=SR1(),
            memory_size=-1,
            stopping_criterion=StopWhenGradientNormLess(10^(-12)),
        )

        x_Ray = random_point(M_Ray)

        x_directBroydenConstant_Ray = quasi_Newton(
            M_Ray,
            F_Ray,
            ∇F_Ray,
            x_Ray;
            direction_update=Broyden(0.5),
            memory_size=-1,
            stopping_criterion=StopWhenGradientNormLess(10^(-12)),
        )

        x_Ray = random_point(M_Ray)

        x_inverseBroydenConstant_Ray = quasi_Newton(
            M_Ray,
            F_Ray,
            ∇F_Ray,
            x_Ray;
            direction_update=InverseBroyden(0.5),
            memory_size=-1,
            stopping_criterion=StopWhenGradientNormLess(10^(-12)),
        )

        @test norm(abs.(x_inverseDFP_Ray) - x_solution_Ray) ≈ 0 atol = 2e-13
        @test norm(abs.(x_directDFP_Ray) - x_solution_Ray) ≈ 0 atol = 2e-13
        @test norm(abs.(x_inverseSR1_Ray) - x_solution_Ray) ≈ 0 atol = 2e-13
        @test norm(abs.(x_directSR1_Ray) - x_solution_Ray) ≈ 0 atol = 2e-13
        @test norm(abs.(x_directBroydenConstant_Ray) - x_solution_Ray) ≈ 0 atol = 2e-13
        @test norm(abs.(x_inverseBroydenConstant_Ray) - x_solution_Ray) ≈ 0 atol = 2e-13
    end
    @tesset "Brocket" begin
        struct GradF
            A::Matrix{Float64}
            N::Diagonal{Float64,Vector{Float64}}
        end
        function (∇F::GradF)(X::Array{Float64,2})
            AX = ∇F.A * X
            XpAX = X' * AX
            return 2 .* AX * ∇F.N .- X * XpAX * ∇F.N .- X * ∇F.N * XpAX
        end

        n = 64
        k = 8
        M_brockett = Stiefel(n, k)
        A_brockett = randn(n, n)
        A_brockett = (A_brockett + A_brockett') / 2
        F_brockett(X) = tr((X' * A_brockett * X) * Diagonal(k:-1:1))
        ∇F_brockett = GradF(A_brockett, Diagonal(Float64.(collect(k:-1:1))))
        x_brockett = random_point(M_brockett)

        x_inverseBFGSCautious_brockett = quasi_Newton(
            M_brockett,
            F_brockett,
            ∇F_brockett,
            x_brockett;
            memory_size=2,
            vector_transport_method=ProjectionTransport(),
            retraction_method=QRRetraction(),
            cautious_update=true,
            stopping_criterion=StopWhenGradientNormLess(10^(-6)),
        )
    end
end
