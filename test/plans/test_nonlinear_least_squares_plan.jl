using Manifolds, Manopt, Test

@testset "Nonlinear lest squares plan" begin
    @testset "Test cost/residual/jacobian cases" begin
        # a simple nlso objective on R2
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
        J(M, x) = cat(j1(M, x), j2(M, x); dims = 2)
        J!(M, J, x) = (J .= cat(j1(M, x), j2(M, x); dims = 2))
        # Smoothing types

        # Test all (new) possible combinations of vectorial cost and Jacobian
        # (1) Function (F, Gradient), Component (C, Gradients), [J] Coefficient (Jacobian in Basis)
        # (2) [a] allocating [i] in place
        nlsoFa = NonlinearLeastSquaresObjective(
            f, JF, 2; jacobian_type = FunctionVectorialType()
        )
        nlsoFi = NonlinearLeastSquaresObjective(
            f!,
            JF!,
            2;
            evaluation = InplaceEvaluation(),
            jacobian_type = FunctionVectorialType(),
        )
        nlsoCa = NonlinearLeastSquaresObjective(
            [f1, f2],
            [j1, j2],
            2;
            function_type = ComponentVectorialType(),
            jacobian_type = ComponentVectorialType(),
        )
        nlsoCi = NonlinearLeastSquaresObjective(
            [f1, f2],
            [j1!, j2!],
            2;
            function_type = ComponentVectorialType(),
            jacobian_type = ComponentVectorialType(),
            evaluation = InplaceEvaluation(),
        )
        nlsoJa = NonlinearLeastSquaresObjective(
            f, J, 2; jacobian_type = CoefficientVectorialType()
        )
        nlsoJi = NonlinearLeastSquaresObjective(f!, J!, 2; evaluation = InplaceEvaluation())

        p = [0.5, 0.5]
        V = [0.0, 0.0]
        Vt = [1 / sqrt(2), 1 / sqrt(2)]
        G = zeros(2, 2)
        Gt = 1 / sqrt(2) .* [-1.0 1.0; 1.0 -1.0]
        for nlso in [nlsoFa, nlsoFi, nlsoCa, nlsoCi, nlsoJa, nlsoJi]
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
            # since s1/s2 are the identity we can also always check against the allocating
            # jacobian of the objective
            G2 = get_jacobian(M, nlso.objective, p)
            @test G2 == Gt
        end
    end
    @testset "Test Change of basis" begin
        J = ones(2, 2)
        Jt = ones(2, 2)
        M = Euclidean(2)
        p = [0.5, 0.5]
        B1 = DefaultBasis()
        B2 = DefaultOrthonormalBasis()
        Manopt._change_basis!(M, J, p, B1, B2)
        # In practice both are the same basis in coordinates, so Jtt stays as iss
        @test J == Jt
    end
end
