using ManifoldDiff, Manifolds, Manopt, Test, RecursiveArrayTools

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
        # F1 is in default form, but for JF1 we have to declare it; we can also start without p1
        r1_2 = LevenbergMarquardt(M1, F1, JF1; jacobian_type = FunctionVectorialType())
        @test isapprox(M1, r1, r1_2; atol = 1.0e-7)
        # the error is less than the deviation from above
        @test norm(F1(M1, r1)) < 0.2
        M1b = Hyperrectangle([-1.0, -1.0], [1.0, 1.0])
        # Then b is out of bounds and we get something where b is on the boundary, namely 1
        # and a is chosen accordingly
        # We have to use the normal coordinates subsolver here then.
        r1b = LevenbergMarquardt(M1b, vgf1, p1; sub_state = CoordinatesNormalSystemState(M1b))
        @test is_point(M1b, r1b)
    end
    @testset "Robust Geodesic Regression on the Sphere" begin
        # Testing the case of one vector function and a single robustifier, so that it is applied componentwise
        M2 = Manifolds.Sphere(2); p1 = [0.0, 0.0, 1.0]; p2 = [0.0, 1.0, 0.0]
        ts = [0.0, 1 / 3, 2 / 3, 1.0]
        qs = shortest_geodesic(M2, p1, p2, ts)
        # Move the last two “east”, the other two “west”
        ps = [exp(M2, p, [i == 2 ? 0.1 : (i == 3 ? -0.1 : 0.0), 0.0, 0.0]) for (i, p) in enumerate(qs)]
        TM2 = TangentBundle(M2)
        function F2(TM::TangentBundle, P; time, data)
            M = base_manifold(TM); p = P[TM, :point]; X = P[TM, :vector]
            return [distance(M, geodesic(M, p, X, ti), di) for (ti, di) in zip(time, data)]
        end
        function f2(TM::TangentBundle, P; time, data)
            M = base_manifold(TM); p = P[TM, :point]; X = P[TM, :vector]
            return 1 / 2 * sum(distance(M, exp(M, p, ti * X), di)^2 for (ti, di) in zip(time, data))
        end
        function f2_robust(TM::TangentBundle, P; time, data)
            M = base_manifold(TM); p = P[TM, :point]; X = P[TM, :vector]
            return sum(distance(M, exp(M, p, ti * X), di) for (ti, di) in zip(time, data))
        end
        function f2_comp_grad(TM::TangentBundle, P; t, d)
            M = base_manifold(TM); p = P[TM, :point]; X = P[TM, :vector]
            g = ManifoldDiff.grad_distance(M, d, exp(M, p, t * X), 1)
            return ArrayPartition(
                ManifoldDiff.adjoint_differential_exp_basepoint(M, p, t * X, g), # w.r.t. base point p
                t * ManifoldDiff.adjoint_differential_exp_argument(M, p, t * X, g), # w.r.t. argument X
            )
        end
        JF2(TM, P; time, data) = [f2_comp_grad(TM, P; t = ti, d = di) for (ti, di) in zip(time, data)]
        vgf2 = VectorGradientFunction(
            (TM, P) -> F2(TM, P; time = ts, data = ps), (TM, P) -> JF2(TM, P; time = ts, data = ps), length(ps);
            evaluation = AllocatingEvaluation(), function_type = FunctionVectorialType(), jacobian_type = FunctionVectorialType(),
        )
        p0 = 1 / sqrt(2) .* [0.0, 1.0, 1.0]; X0 = [1.0, 0.0, 0.0]
        P0 = ArrayPartition(p0, X0)
        # LSQ
        P2a = LevenbergMarquardt(
            TM2, vgf2, P0;
            retraction_method = StabilizedRetraction(default_retraction_method(TM2)),
            # debug = [:Iteration, (:Cost, "f(x): %8.8e "), :damping_term, "\n", :Stop],
        )
        @test is_point(TM2, P2a)
        # robust – and test both decorators for state and objective
        (o2, s2) = LevenbergMarquardt(
            TM2, vgf2, P0;
            robustifier = 1.0e-4 ∘ HuberRobustifier(), return_objective = true, return_state = true,
            retraction_method = StabilizedRetraction(default_retraction_method(TM2)),
            # debug = [:Iteration, (:Cost, "f(x): %8.8e "), :damping_term, "\n", :Stop],\
        );
        P2b = get_solver_result(s2)
        @test is_point(TM2, P2b)
        @test f2(TM2, P2a; time = ts, data = ps) < f2(TM2, P2b; time = ts, data = ps)
        @test f2_robust(TM2, P2b; time = ts, data = ps) < f2_robust(TM2, P2a; time = ts, data = ps)
        p2a = P2a[TM2, :point]; X2a = P2a[TM2, :vector]; p2b = P2b[TM2, :point]; X2b = P2b[TM2, :vector]
        geoa = geodesic(M2, p2a, X2a, range(0.0, 1.0, 100)); geob = geodesic(M2, p2b, X2b, range(0.0, 1.0, 100))
        # for the robust case: end points are closer to data than for lsq
        @test distance(M2, geob[1], ps[1]) < distance(M2, geoa[1], ps[1])
        @test distance(M2, geob[end], ps[end]) < distance(M2, geoa[end], ps[end])
        # You can easily plot this as
        # using ManifoldMakie
        # scatter(M2, ps); geodesics!(M2, geob); geodesics!(M2, geoa);
        # the first curve (same color as points) should hit the end points, the second is “skewed”
        @testset "show/repr on the LevenbergMarquardt state on NL objective" begin
            @test startswith(repr(o2), "ManifoldNonlinearLeastSquaresObjective(")
            @test Manopt.status_summary(o2) == "A nonlinear least squares objective 1 vectorial block"
            @test startswith(repr(s2), "LevenbergMarquardtState(")
            @test startswith(Manopt.status_summary(s2), "# Solver state for `Manopt.jl`s Levenberg Marquardt Algorithm")
        end
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
