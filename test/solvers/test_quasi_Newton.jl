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
        x_lrbfgs_cached = quasi_Newton(
            M,
            F,
            ∇F,
            x;
            stopping_criterion=StopWhenGradientNormLess(10^(-6)),
            basis=get_basis(M, x, DefaultOrthonormalBasis()),
        )
        @test x_lrbfgs_cached == x_lrbfgs

        x_lrbfgs_cached_2 = quasi_Newton(
            M,
            F,
            ∇F,
            x;
            stopping_criterion=StopWhenGradientNormLess(10^(-6)),
            basis=get_basis(M, x, DefaultOrthonormalBasis()),
            memory_size=-1,
        )
        @test isapprox(M, x_lrbfgs_cached_2, x_lrbfgs; atol=1e-5)

        x_clrbfgs = quasi_Newton(
            M,
            F,
            ∇F,
            x;
            cautious_update=true,
            stopping_criterion=StopWhenGradientNormLess(10^(-6)),
        )
        @test norm(x_clrbfgs - x_solution) ≈ 0 atol = 10.0^(-14)

        x_rbfgs_Huang = quasi_Newton(
            M,
            F,
            ∇F,
            x;
            memory_size=-1,
            step_size=WolfePowellBinaryLinesearch(
                ExponentialRetraction(), ParallelTransport()
            ),
            stopping_criterion=StopWhenGradientNormLess(10^(-6)),
        )
        @test norm(x_rbfgs_Huang - x_solution) ≈ 0 atol = 10.0^(-14)

        for T in [InverseBFGS(), BFGS(), InverseDFP(), DFP(), InverseSR1(), SR1()]
            for c in [true, false]
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
        n = 4
        rayleigh_atol = 1e-12
        A = randn(n, n)
        A = (A + A') / 2
        F(X) = X' * A * X
        ∇F(X) = 2 * (A * X - X * (X' * A * X))
        M = Sphere(n - 1)
        x_solution = abs.(eigvecs(A)[:, 1])

        x = Matrix{Float64}(I, n, n)[n, :]
        x_lrbfgs = quasi_Newton(
            M,
            F,
            ∇F,
            x;
            basis=get_basis(M, x, DefaultOrthonormalBasis()),
            memory_size=-1,
            stopping_criterion=StopWhenGradientNormLess(10^(-12)),
        )
        @test norm(abs.(x_lrbfgs) - x_solution) ≈ 0 atol = rayleigh_atol

        x_clrbfgs = quasi_Newton(
            M,
            F,
            ∇F,
            x;
            cautious_update=true,
            stopping_criterion=StopWhenGradientNormLess(10^(-12)),
        )

        x_cached_lrbfgs = quasi_Newton(
            M,
            F,
            ∇F,
            x;
            basis=get_basis(M, x, DefaultOrthonormalBasis()),
            memory_size=-1,
            stopping_criterion=StopWhenGradientNormLess(10^(-12)),
        )
        @test norm(abs.(x_cached_lrbfgs) - x_solution) ≈ 0 atol = rayleigh_atol

        for T in [InverseBFGS(), BFGS()], c in [true, false]
            x = Matrix{Float64}(I, n, n)[n, :]
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
            @test norm(abs.(x_direction) - x_solution) ≈ 0 atol = rayleigh_atol
        end

        for T in [
            InverseDFP(),
            DFP(),
            Broyden(0.5),
            InverseBroyden(0.5),
            Broyden(0.5, :Davidon),
            Broyden(0.5, :InverseDavidon),
        ]
            x_direction = quasi_Newton(
                M,
                F,
                ∇F,
                x;
                direction_update=T,
                memory_size=-1,
                stopping_criterion=StopWhenGradientNormLess(10^(-12)),
            )
            @test norm(abs.(x_direction) - x_solution) ≈ 0 atol = rayleigh_atol
        end

        for T in [SR1(), InverseSR1(), SR1(1e-9), InverseSR1(1e-9)]
            x_direction = quasi_Newton(
                M,
                F,
                ∇F,
                x;
                direction_update=T,
                memory_size=-1,
                stopping_criterion=StopWhenGradientNormLess(1e-9),
            )
            @test norm(abs.(x_direction) - x_solution) ≈ 0 atol = 1e-9
        end
    end
    @testset "Brockett" begin
        struct GradF
            A::Matrix{Float64}
            N::Diagonal{Float64,Vector{Float64}}
        end
        function (∇F::GradF)(X::Array{Float64,2})
            AX = ∇F.A * X
            XpAX = X' * AX
            return 2 .* AX * ∇F.N .- X * XpAX * ∇F.N .- X * ∇F.N * XpAX
        end

        n = 1000
        k = 5
        M = Stiefel(n, k)
        A = randn(n, n)
        A = (A + A') / 2
        F(X) = tr((X' * A * X) * Diagonal(k:-1:1))
        ∇F = GradF(A, Diagonal(Float64.(collect(k:-1:1))))

        x = Matrix{Float64}(I, n, n)[:, 2:(k + 1)]
        x_inverseBFGSCautious = quasi_Newton(
            M,
            F,
            ∇F,
            x;
            memory_size=8,
            vector_transport_method=ProjectionTransport(),
            retraction_method=QRRetraction(),
            cautious_update=true,
            stopping_criterion=StopWhenGradientNormLess(10^(-6)),
        )

        x_inverseBFGSHuang = quasi_Newton(
            M,
            F,
            ∇F,
            x;
            memory_size=8,
            step_size=WolfePowellBinaryLinesearch(QRRetraction(), ProjectionTransport()),
            vector_transport_method=ProjectionTransport(),
            retraction_method=QRRetraction(),
            cautious_update=true,
            stopping_criterion=StopWhenGradientNormLess(10^(-6)),
        )
        @test isapprox(M, x_inverseBFGSCautious, x_inverseBFGSHuang; atol=2e-5)
    end
end
