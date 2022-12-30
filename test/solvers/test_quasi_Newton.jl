using Manopt, Manifolds, Test
using LinearAlgebra: I, eigvecs, tr, Diagonal

@testset "Riemannian quasi-Newton Methods" begin
    @testset "Mean of 3 Matrices" begin
        # Mean of 3 matrices
        A = [18.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
        B = [0.0 0.0 0.0 0.009; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
        C = [0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; -5.0 0.0 0.0 0.0]
        ABC = [A, B, C]
        x_solution = mean(ABC)
        f(::Euclidean, x) = 0.5 * norm(A - x)^2 + 0.5 * norm(B - x)^2 + 0.5 * norm(C - x)^2
        grad_f(::Euclidean, x) = -A - B - C + 3 * x
        M = Euclidean(4, 4)
        x = zeros(Float64, 4, 4)
        x_lrbfgs = quasi_Newton(
            M, f, grad_f, x; stopping_criterion=StopWhenGradientNormLess(10^(-6))
        )
        @test norm(x_lrbfgs - x_solution) ≈ 0 atol = 10.0^(-14)
        # with State
        lrbfgs_o = quasi_Newton(
            M,
            f,
            grad_f,
            x;
            stopping_criterion=StopWhenGradientNormLess(10^(-6)),
            return_state=true,
        )
        dmp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
        @test get_last_stepsize(dmp, lrbfgs_o, lrbfgs_o.stepsize) > 0
        @test lrbfgs_o.x == x_lrbfgs
        # with Cached Basis
        x_lrbfgs_cached = quasi_Newton(
            M,
            f,
            grad_f,
            x;
            stopping_criterion=StopWhenGradientNormLess(10^(-6)),
            basis=get_basis(M, x, DefaultOrthonormalBasis()),
        )
        @test x_lrbfgs_cached == x_lrbfgs

        x_lrbfgs_cached_2 = quasi_Newton(
            M,
            f,
            grad_f,
            x;
            stopping_criterion=StopWhenGradientNormLess(10^(-6)),
            basis=get_basis(M, x, DefaultOrthonormalBasis()),
            memory_size=-1,
        )
        @test isapprox(M, x_lrbfgs_cached_2, x_lrbfgs; atol=1e-5)

        x_clrbfgs = quasi_Newton(
            M,
            f,
            grad_f,
            x;
            cautious_update=true,
            stopping_criterion=StopWhenGradientNormLess(10^(-6)),
        )
        @test norm(x_clrbfgs - x_solution) ≈ 0 atol = 10.0^(-14)

        x_rbfgs_Huang = quasi_Newton(
            M,
            f,
            grad_f,
            x;
            memory_size=-1,
            stepsize=WolfePowellBinaryLinesearch(
                M;
                retraction_method=ExponentialRetraction(),
                vector_transport_method=ParallelTransport(),
            ),
            stopping_criterion=StopWhenGradientNormLess(10^(-6)),
        )
        @test norm(x_rbfgs_Huang - x_solution) ≈ 0 atol = 10.0^(-14)

        for T in [InverseBFGS(), BFGS(), InverseDFP(), DFP(), InverseSR1(), SR1()]
            for c in [true, false]
                x_direction = quasi_Newton(
                    M,
                    f,
                    grad_f,
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
        rayleigh_atol = 1e-8
        A = [2.0 1.0 0.0 3.0; 1.0 3.0 4.0 5.0; 0.0 4.0 3.0 2.0; 3.0 5.0 2.0 6.0]
        A = (A + A') / 2
        M = Sphere(n - 1)
        f(::Sphere, X) = X' * A * X
        grad_f(::Sphere, X) = 2 * (A * X - X * (X' * A * X))
        x_solution = abs.(eigvecs(A)[:, 1])

        x = Matrix{Float64}(I, n, n)[n, :]
        x_lrbfgs = quasi_Newton(
            M,
            f,
            grad_f,
            x;
            basis=get_basis(M, x, DefaultOrthonormalBasis()),
            memory_size=-1,
            stopping_criterion=StopWhenGradientNormLess(1e-9),
        )
        @test norm(abs.(x_lrbfgs) - x_solution) ≈ 0 atol = rayleigh_atol

        x_clrbfgs = quasi_Newton(
            M,
            f,
            grad_f,
            x;
            cautious_update=true,
            stopping_criterion=StopWhenGradientNormLess(1e-9),
        )

        x_cached_lrbfgs = quasi_Newton(
            M,
            f,
            grad_f,
            x;
            basis=get_basis(M, x, DefaultOrthonormalBasis()),
            memory_size=-1,
            stopping_criterion=StopWhenGradientNormLess(1e-9),
        )
        @test norm(abs.(x_cached_lrbfgs) - x_solution) ≈ 0 atol = rayleigh_atol

        for T in [
                InverseDFP(),
                DFP(),
                Broyden(0.5),
                InverseBroyden(0.5),
                Broyden(0.5, :Davidon),
                Broyden(0.5, :InverseDavidon),
                InverseBFGS(),
                BFGS(),
            ],
            c in [true, false]

            x_direction = quasi_Newton(
                M,
                f,
                grad_f,
                x;
                direction_update=T,
                cautious_update=c,
                memory_size=-1,
                stopping_criterion=StopWhenGradientNormLess(5 * 1e-8),
            )
            @test norm(abs.(x_direction) - x_solution) ≈ 0 atol = rayleigh_atol
        end
    end
    @testset "Brockett" begin
        struct GradF
            A::Matrix{Float64}
            N::Diagonal{Float64,Vector{Float64}}
        end
        function (gradF::GradF)(::Stiefel, X::Array{Float64,2})
            AX = gradF.A * X
            XpAX = X' * AX
            return 2 .* AX * gradF.N .- X * XpAX * gradF.N .- X * gradF.N * XpAX
        end

        n = 4
        k = 2
        M = Stiefel(n, k)
        A = [2.0 1.0 0.0 3.0; 1.0 3.0 4.0 5.0; 0.0 4.0 3.0 2.0; 3.0 5.0 2.0 6.0]
        f(::Stiefel, X) = tr((X' * A * X) * Diagonal(k:-1:1))
        grad_f = GradF(A, Diagonal(Float64.(collect(k:-1:1))))

        x = Matrix{Float64}(I, n, n)[:, 2:(k + 1)]
        x_inverseBFGSCautious = quasi_Newton(
            M,
            f,
            grad_f,
            x;
            memory_size=8,
            vector_transport_method=ProjectionTransport(),
            retraction_method=QRRetraction(),
            cautious_update=true,
            stopping_criterion=StopWhenGradientNormLess(1e-6),
        )

        x_inverseBFGSHuang = quasi_Newton(
            M,
            f,
            grad_f,
            x;
            memory_size=8,
            stepsize=WolfePowellBinaryLinesearch(
                M;
                retraction_method=QRRetraction(),
                vector_transport_method=ProjectionTransport(),
            ),
            vector_transport_method=ProjectionTransport(),
            retraction_method=QRRetraction(),
            cautious_update=true,
            stopping_criterion=StopWhenGradientNormLess(1e-6),
        )
        @test isapprox(M, x_inverseBFGSCautious, x_inverseBFGSHuang; atol=2e-4)
    end

    @testset "Wolfe Powell linesearch" begin
        n = 4
        rayleigh_atol = 1e-8
        A = [2.0 1.0 0.0 3.0; 1.0 3.0 4.0 5.0; 0.0 4.0 3.0 2.0; 3.0 5.0 2.0 6.0]
        A = (A + A') / 2
        M = Sphere(n - 1)
        F(::Sphere, X) = X' * A * X
        grad_f(::Sphere, X) = 2 * (A * X - X * (X' * A * X))
        x_solution = abs.(eigvecs(A)[:, 1])

        x = [
            0.7011245948687502
            -0.1726003159556036
            0.38798265967671103
            -0.5728026616491424
        ]
        x_lrbfgs = quasi_Newton(
            M,
            F,
            grad_f,
            x;
            basis=get_basis(M, x, DefaultOrthonormalBasis()),
            memory_size=-1,
            stopping_criterion=StopWhenGradientNormLess(1e-9),
        )
        @test norm(abs.(x_lrbfgs) - x_solution) ≈ 0 atol = rayleigh_atol
    end

    @testset "update rules" begin
        n = 4
        A = [2.0 1.0 0.0 3.0; 1.0 3.0 4.0 5.0; 0.0 4.0 3.0 2.0; 3.0 5.0 2.0 6.0]
        A = (A + A') / 2
        M = Sphere(n - 1)
        F(::Sphere, X) = X' * A * X
        grad_f(::Sphere, X) = 2 * (A * X - X * (X' * A * X))
        grad_f!(::Sphere, X, p) = (X .= 2 * (A * X - X * (X' * A * X)))

        p_1 = [1.0; 0.0; 0.0; 0.0]
        p_2 = [0.0; 0.0; 1.0; 0.0]

        SR1_allocating = ApproxHessianSymmetricRankOne(
            M, p_1, grad_f; evaluation=AllocatingEvaluation()
        )

        SR1_mutating = ApproxHessianSymmetricRankOne(
            M, p_1, grad_f!; evaluation=InplaceEvaluation()
        )

        BFGS_allocating = ApproxHessianBFGS(
            M, p_1, grad_f; evaluation=AllocatingEvaluation()
        )

        BFGS_mutating = ApproxHessianBFGS(M, p_1, grad_f!; evaluation=InplaceEvaluation())

        Y = [0.0; 1.0; 0.0; 0.0]
        X_1 = SR1_allocating(M, p_1, Y)
        SR1_allocating.p_tmp = p_2
        X_2 = SR1_allocating(M, p_1, Y)
        @test isapprox(M, p_1, X_1, X_2; atol=1e-10)

        X_3 = zero_vector(M, p_1)
        X_4 = zero_vector(M, p_1)
        SR1_mutating(M, X_3, p_1, Y)
        SR1_mutating.p_tmp = p_2
        SR1_mutating(M, X_4, p_1, Y)
        @test isapprox(M, p_1, X_3, X_4; atol=1e-10)

        X_5 = BFGS_allocating(M, p_1, Y)
        X_6 = BFGS_allocating(M, p_2, Y)
        @test isapprox(M, p_1, X_5, X_6; atol=1e-10)

        X_7 = zero_vector(M, p_1)
        X_8 = zero_vector(M, p_1)
        BFGS_mutating(M, X_7, p_1, Y)
        BFGS_mutating(M, X_8, p_2, Y)

        @test isapprox(M, p_1, X_3, X_4; atol=1e-10)

        BFGS_allocating.grad_tmp = ones(4)
        BFGS_allocating.matrix = one(zeros(3, 3))
        Manopt.update_hessian!(M, BFGS_allocating, p_1, p_2, Y)
        test_m = [1.0 -1.0 5.0; -1.0 2.0 -5.0; 5.0 -5.0 26.0]
        @test isapprox(test_m, BFGS_allocating.matrix)

        update_hessian_basis!(M, BFGS_allocating, p_1)
        @test isapprox(M, p_1, BFGS_allocating.grad_tmp, [0.0, 2.0, 0.0, 6.0])
    end
end
