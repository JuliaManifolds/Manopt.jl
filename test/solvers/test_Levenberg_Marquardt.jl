using Manifolds, Manopt, Test, ManifoldsBase

@testset "The (robust) Riemannian Levenberg Marquardt Algorithm" begin
    @testset "Linear Regression" begin
        # Linear regression with bounds
        q1(x; a, b) = a * x + b
        X1 = [1.0, 2.0, 3.0]
        # the first two are outliers, the third fits q(x) = 0.5x + 2
        Y1 = [2.6, 2.9, 3.5]
        # the vectorial function hence is q(x) - y
        res1(a, b; X, Y) = [q1(x; a = a, b = b) - y for (x, y) in zip(X, Y)]
        # its differential is x for a and 1 for b
        Dres1(a, b; X, Y) = [[x, one(x)] for (x, y) in zip(X, Y)]
        M1 = Euclidean(2)
        F1(M::AbstractManifold, p) = res1(p[1], p[2]; X = X1, Y = Y1)
        JF1(M::AbstractManifold, p) = Dres1(p[1], p[2]; X = X1, Y = Y1)
        vgf1 = VectorGradientFunction(
            F1, JF1, length(X1);
            evaluation = AllocatingEvaluation(),
            function_type = FunctionVectorialType(), jacobian_type = FunctionVectorialType(),
        )
        p1 = [0.0, 0.0]
        r1 = LevenbergMarquardt(M1, vgf1, p1)
        # the error is less than the deviation from above
        @test norm(F1(M1, r1)) < 0.2
        M1b = Hyperrectangle([-1.0, -1.0], [1.0, 1.0])
        # Then b is out of bounds and we get something where b is on the boundary, namely 1
        # and a is chosen accordingly
        # We have to use the normal coordinates subsolver here then.
        r1b = LevenbergMarquardt(M1b, vgf1, p1; sub_state = CoordinatesNormalSystemState(M1b))
        @test is_point(M1b, r1b)
    end
    @testset "Geodesic Regression on the Sphere" begin
        # TODO
    end
    # TODO: Allocating vs in-place F and JacF
    @testset "errors" begin
        sub_fake_f = (args...) -> 0
        sub_state = AllocatingEvaluation()
        x0 = [4.0, 2.0]
        M = Euclidean(2)
        i_res = similar(x0, 3) # similar to regression where we have vectors of length 3
        i_JF = similar(x0, 3, 2) # and on R2 the Jac is 3x2
        # η too large (≥ 1)
        @test_throws ArgumentError LevenbergMarquardtState(
            M, sub_fake_f, sub_state, i_res, i_JF; p = x0, candidate_acceptance_threshold = 2,
        )
        # η too small (≤ 0)
        @test_throws ArgumentError LevenbergMarquardtState(
            M, sub_fake_f, sub_state, i_res, i_JF; p = x0, candidate_acceptance_threshold = -1,
        )
        # damping term negative
        @test_throws ArgumentError LevenbergMarquardtState(
            M, sub_fake_f, sub_state, i_res, i_JF; p = x0, damping_term_min = -1,
        )
        # damping_increase_factor too small (≤ 1)
        @test_throws ArgumentError LevenbergMarquardtState(
            M, sub_fake_f, sub_state, i_res, i_JF; p = x0, damping_increase_factor = 0.5,
        )
        # For the evaluating case num_components can not be derived in code, hence this errors
        @test_throws ArgumentError LevenbergMarquardt(
            M, (M, v, p) -> v, (M, X, p) -> X, x0; evaluation = InplaceEvaluation(),
        )
    end
end
