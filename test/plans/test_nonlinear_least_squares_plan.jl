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
        # (1) Function (F, Gradient), Component (C, Gradients), [J] Coordinate (Jacobian in Basis)
        # (2) [a] allocating [i] in place
        nlsoFa = ManifoldNonlinearLeastSquaresObjective(
            f, JF, 2; jacobian_type = FunctionVectorialType()
        )
        nlsoFi = ManifoldNonlinearLeastSquaresObjective(
            f!, JF!, 2;
            evaluation = InplaceEvaluation(), jacobian_type = FunctionVectorialType(),
        )
        nlsoCa = ManifoldNonlinearLeastSquaresObjective(
            [f1, f2], [j1, j2], 2;
            function_type = ComponentVectorialType(), jacobian_type = ComponentVectorialType(),
        )
        vgf1 = VectorGradientFunction(f, JF, 2; jacobian_type = FunctionVectorialType())
        nlsoRobust = ManifoldNonlinearLeastSquaresObjective(
            [vgf1, vgf1], [HuberRobustifier(), IdentityRobustifier()]
        )
        nlsoCi = ManifoldNonlinearLeastSquaresObjective(
            [f1, f2], [j1!, j2!], 2; evaluation = InplaceEvaluation(),
            function_type = ComponentVectorialType(), jacobian_type = ComponentVectorialType(),
        )
        nlsoJa = ManifoldNonlinearLeastSquaresObjective(
            f, J, 2; jacobian_type = CoefficientVectorialType()
        )
        nlsoJi = ManifoldNonlinearLeastSquaresObjective(f!, J!, 2; evaluation = InplaceEvaluation())

        p = [0.5, 0.5]
        X = [0.25, 0.25]
        Y = [0.25, -0.25]
        V = [0.0, 0.0]
        Vt = [1 / sqrt(2), 1 / sqrt(2)]
        G = zeros(2, 2)
        Gt = 1 / sqrt(2) .* [-1.0 1.0; 1.0 -1.0]
        for nlso in [nlsoFa, nlsoFi, nlsoCa, nlsoCi, nlsoJa, nlsoJi]
            @testset "$(nlso) and its internal VGF" begin
                vgf = nlso.objective[1] # the vector of VGFs is length 1 here for all cases.
                c = get_cost(M, nlso, p)
                @test c ≈ 0.5
                fill!(V, 0.0)
                get_residuals!(M, V, nlso, p)
                @test V == get_residuals(M, nlso, p)
                @test V ≈ Vt
                @test 0.5 * sum(abs.(V) .^ 2) ≈ c
                fill!(G, 0.0)
                get_jacobian!(M, G, vgf, p)
                @test G == get_jacobian(M, vgf, p)
                @test G == Gt
            end
            c = get_cost(M, nlso, p)
            @test c ≈ 0.5
            fill!(V, 0.0)
            get_residuals!(M, V, nlso, p)
            @test V == get_residuals(M, nlso, p)
            @test V ≈ Vt
            @test 0.5 * sum(abs.(V) .^ 2) ≈ c
            @test startswith(repr(nlso), "ManifoldNonlinearLeastSquaresObjective(")
            @test startswith(Manopt.status_summary(nlso), "A nonlinear least squares objective")
        end
        @testset "Linear Surrogate accessors" begin
            X = [0.25, 0.25]
            Y = [0.25, -0.25]
            for nlso in [nlsoFa, nlsoFi, nlsoCa, nlsoRobust, nlsoCi, nlsoJa, nlsoJi]
                lmlso = LevenbergMarquardtLinearSurrogateObjective(nlso)
                sG = get_gradient(M, lmlso, p, X)
                sG! = zero_vector(M, p)
                sG = get_gradient!(M, sG!, lmlso, p, X)
                @test isapprox(M, p, sG, sG!)
                sH = get_hessian(M, lmlso, p, X, Y)
                sH! = zero_vector(M, p)
                sH = get_hessian!(M, sH!, lmlso, p, X, Y)
                @test isapprox(M, p, sH, sH!)
                nvf = Manopt.get_normal_vector_field(M, lmlso, p, DefaultOrthogonalBasis())
                nvf! = zeros(2)
                Manopt.get_normal_vector_field!(M, nvf!, lmlso, p, DefaultOrthogonalBasis())
                @test isapprox(nvf, nvf!)
                @test norm(nvf) ≈ 0 atol=1e-14
                nlo = Manopt.get_normal_linear_operator(M, lmlso, p, X)
                nlo! = zeros(2)
                Manopt.get_normal_linear_operator!(M, nlo!, lmlso, p, X)
                @test isapprox(nlo, nlo!)
                nloB = Manopt.get_normal_linear_operator(M, lmlso, p, [1.0, 2.0], DefaultOrthogonalBasis())
                nloB! = zeros(2)
                Manopt.get_normal_linear_operator!(M, nloB!, lmlso, p, [1.0, 2.0], DefaultOrthogonalBasis())
                @test isapprox(nloB, nloB!)
                nloBA = Manopt.get_normal_linear_operator(M, lmlso, p, DefaultOrthogonalBasis())
                nloBA! = zeros(2,2)
                Manopt.get_normal_linear_operator!(M, nloBA!, lmlso, p, DefaultOrthogonalBasis())
                @test isapprox(nloBA, nloBA!)
                # the normal ones are mapped to _ ones for the NormalEq and the vector gets a minus
                neo = Manopt.NormalEquationsObjective(lmlso)
                # its linear operator and vector field (in a basis)
                # TODO Adapt calls similar to the previous two once the normal lin op works

            end
        end
    end
    @testset "Dummy decorator pass through" begin

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
    @testset "show/repr and status_summary" begin
        M = Euclidean(3)
        f(M, p) = p
        J_f(M, p) = one(p)
        mnlso = ManifoldNonlinearLeastSquaresObjective(f, J_f, 3)
    end
    @testset "Inner consistency checks" begin
        s = zeros(2)
        Manopt.default_lm_lin_solve!(s, [0.2 0.4; 0.4 0.2], [0.6, 0.0])
        @test s ≈ [-1.0, 2.0]
    end
end
