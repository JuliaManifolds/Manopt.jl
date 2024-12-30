using Manifolds, Manopt, Test

@testset "Nonlinear lest squares plan" begin
    @testset "Test cost/residual/jacobian cases with smoothing" begin
        # a simple nlso objetive on R2
        M = Euclidean(2)
        d1 = [1, 0]
        d2 = [0, 1]
        f1(M, x) = norm(x - d1)
        f2(M, x) = norm(x - d2)
        f(M, x) = [f1(M, x), f2(M, x)]
        # Components
        f!(M, V, x) = (V .= [f1(M, x), f2(M, x)])
        j1(M, x) = (x - d1) / norm(x - d1)
        j1!(M, X, x) = (X .= (x - d1) / norm(x - d1))
        j2(M, x) = (x - d2) / norm(x - d2)
        j2!(M, X, x) = (X .= (x - d2) / norm(x - d2))
        # Function
        JF(M, x) = [j1(M, x), j2(M, x)]
        JF!(M, JF, x) = (JF .= [j1(M, x), j2(M, x)])
        # Jacobi matrix
        J(M, x) = cat(j1(M, x), j2(M, x); dims=2)
        J!(M, J, x) = (J .= cat(j1(M, x), j2(M, x); dims=2))
        # Smoothing types
        s1 = Manopt.smoothing_factory(:Identity)
        s2 = Manopt.smoothing_factory((:Identity, 2))

        # Test all (new) possible combinations of vectorial cost and Jacobian
        # (1) [F]unction (Gradient), [C]omponent (Gradients), [J] Coordinate (Jacobian in Basis)
        # (2) [a]llocating [i] inplace
        # (3) [s] single smoothing [v] vector smoothing
        nlsoFas = NonlinearLeastSquaresObjective(
            f, JF, 2; jacobian_type=FunctionVectorialType(), smoothing=s1
        )
        nlsoFav = NonlinearLeastSquaresObjective(
            f, JF, 2; jacobian_type=FunctionVectorialType(), smoothing=s2
        )
        nlsoFis = NonlinearLeastSquaresObjective(
            f!,
            JF!,
            2;
            evaluation=InplaceEvaluation(),
            jacobian_type=FunctionVectorialType(),
            smoothing=s1,
        )
        nlsoFiv = NonlinearLeastSquaresObjective(
            f!,
            JF!,
            2;
            evaluation=InplaceEvaluation(),
            jacobian_type=FunctionVectorialType(),
            smoothing=s2,
        )

        nlsoCas = NonlinearLeastSquaresObjective(
            [f1, f2],
            [j1, j2],
            2;
            function_type=ComponentVectorialType(),
            jacobian_type=ComponentVectorialType(),
            smoothing=s1,
        )
        nlsoCav = NonlinearLeastSquaresObjective(
            [f1, f2],
            [j1, j2],
            2;
            function_type=ComponentVectorialType(),
            jacobian_type=ComponentVectorialType(),
            smoothing=s2,
        )
        nlsoCis = NonlinearLeastSquaresObjective(
            [f1, f2],
            [j1!, j2!],
            2;
            function_type=ComponentVectorialType(),
            jacobian_type=ComponentVectorialType(),
            evaluation=InplaceEvaluation(),
            smoothing=s1,
        )
        nlsoCiv = NonlinearLeastSquaresObjective(
            [f1, f2],
            [j1!, j2!],
            2;
            function_type=ComponentVectorialType(),
            jacobian_type=ComponentVectorialType(),
            evaluation=InplaceEvaluation(),
            smoothing=s2,
        )

        nlsoJas = NonlinearLeastSquaresObjective(f, J, 2; smoothing=s1)
        nlsoJav = NonlinearLeastSquaresObjective(f, J, 2; smoothing=s2)
        nlsoJis = NonlinearLeastSquaresObjective(
            f!, J!, 2; evaluation=InplaceEvaluation(), smoothing=s1
        )
        nlsoJiv = NonlinearLeastSquaresObjective(
            f!, J!, 2; evaluation=InplaceEvaluation(), smoothing=s2
        )

        p = [0.5, 0.5]
        V = [0.0, 0.0]
        Vt = [1 / sqrt(2), 1 / sqrt(2)]
        G = zeros(2, 2)
        Gt = 1 / sqrt(2) .* [-1.0 1.0; 1.0 -1.0]
        for nlso in [
            nlsoFas,
            nlsoFav,
            nlsoFis,
            nlsoFiv,
            nlsoCas,
            nlsoCav,
            nlsoCis,
            nlsoCiv,
            nlsoJas,
            nlsoJav,
            nlsoJis,
            nlsoJiv,
        ]
            c = get_cost(M, nlso, p)
            @test c ≈ 0.5
            fill!(V, 0.0)
            get_residuals!(M, V, nlso, p)
            @test V == get_residuals(M, nlso, p)
            @test V ≈ Vt
            @test 0.5 * sum(abs.(V) .^ 2) ≈ c
            fill!(G, 0.0)
            get_jacobian!(M, G, nlso, p)
            @test G == get_jacobian(M, nlso, p)
            @test G == Gt
        end
    end
    @testset "Smootthing factory" begin
        s1 = Manopt.smoothing_factory()
        @test s1 isa ManifoldHessianObjective
        s1s = Manopt.smoothing_factory((s1, 2.0))
        @test s1s isa ManifoldHessianObjective
        s1v = Manopt.smoothing_factory((s1, 3))
        @test s1v isa VectorHessianFunction
        @test length(s1v) == 3

        @test Manopt.smoothing_factory(s1) === s1 # Passthrough for mhos
        s2 = Manopt.smoothing_factory((:Identity, 2))
        @test s2 isa VectorHessianFunction
        @test length(s2) == 2
        @test Manopt.smoothing_factory(s2) === s2 # Passthrough for vhfs

        s3 = Manopt.smoothing_factory((:Identity, 3.0))
        @test s3 isa ManifoldHessianObjective

        for s in [:Arctan, :Cauchy, :Huber, :SoftL1, :Tukey]
            s4 = Manopt.smoothing_factory(s)
            @test s4 isa ManifoldHessianObjective
        end

        # Combine all different types
        s5 = Manopt.smoothing_factory((:Identity, 2), (:Huber, 3), s1, :Tukey, s2)
        @test s5 isa VectorHessianFunction
        @test length(s5) == 9
    end
end
