using ManifoldDiff, Manifolds, Manopt, Test, RecursiveArrayTools

@testset "The (robust) Riemannian Levenberg Marquardt Algorithm" begin
    @testset "Linear Regression" begin
        # Linear regression with bounds
        q1(x; a, b) = a * x + b
        X1 = [1.0, 2.0, 3.0]
        m = length(X1)
        Y1 = [2.6, 2.9, 3.5]
        # the vectorial function hence is q(x) - y
        res1(a, b; X, Y) = [q1(x; a = a, b = b) - y for (x, y) in zip(X, Y)]
        function res1!(v, a, b; X, Y)
            for (i, (x, y)) in enumerate(zip(X, Y))
                v[i] = q1(x; a = a, b = b) - y
            end
            return v
        end
        # its differential is x for a and 1 for b
        Dres1(a, b; X, Y) = [[x, one(x)] for (x, y) in zip(X, Y)]
        function Dres1!(D, a, b; X, Y)
            for (i, (x, y)) in enumerate(zip(X, Y))
                D[i] .= [x, one(x)]
            end
            return D
        end
        M1 = Euclidean(2)
        F1(M::AbstractManifold, p) = res1(p[1], p[2]; X = X1, Y = Y1)
        F1!(M::AbstractManifold, v, p) = res1!(v, p[1], p[2]; X = X1, Y = Y1)
        JF1(M::AbstractManifold, p) = Dres1(p[1], p[2]; X = X1, Y = Y1)
        JF1!(M::AbstractManifold, J, p) = Dres1!(J, p[1], p[2]; X = X1, Y = Y1)
        JF1mat(M::AbstractManifold, p) = hcat(Dres1(p[1], p[2]; X = X1, Y = Y1)...)'
        vgf1 = VectorGradientFunction(
            F1, JF1, m; evaluation = AllocatingEvaluation(),
            function_type = FunctionVectorialType(), jacobian_type = FunctionVectorialType(),
        )
        vgf1! = VectorGradientFunction(
            F1!, JF1!, m; evaluation = InplaceEvaluation(),
            function_type = FunctionVectorialType(), jacobian_type = FunctionVectorialType(),
        )
        p1 = [0.0, 0.0]
        # Interface I: Functions
        r1a1 = LevenbergMarquardt(
            M1, F1, JF1, p1, m;
            function_type = FunctionVectorialType(), jacobian_type = FunctionVectorialType()
        )
        @test norm(F1(M1, r1a1)) < 0.2
        # We can even leave out m
        r1a2 = LevenbergMarquardt(
            M1, F1, JF1, p1;
            function_type = FunctionVectorialType(), jacobian_type = FunctionVectorialType()
        )
        @test isapprox(M1, r1a1, r1a2; atol = 1.0e-7)
        # We can do the same for in-place
        r1a3 = copy(M1, p1)
        LevenbergMarquardt!(
            M1, F1, JF1, r1a3;
            function_type = FunctionVectorialType(), jacobian_type = FunctionVectorialType()
        )
        @test isapprox(M1, r1a1, r1a3; atol = 1.0e-7)
        # We can even leave out both p1 _and_ m
        r1a4 = LevenbergMarquardt(
            M1, F1, JF1;
            function_type = FunctionVectorialType(), jacobian_type = FunctionVectorialType()
        )
        @test isapprox(M1, r1a1, r1a4; atol = 1.0e-7)
        # Interface II: vgf
        r1a5 = LevenbergMarquardt(M1, vgf1, p1)
        # two identical runs agree exactly (deterministic sub solver start)
        @test r1a5 == LevenbergMarquardt(M1, vgf1, p1)
        @test isapprox(M1, r1a1, r1a5; atol = 1.0e-7)
        # also in place vgf
        r1a6 = copy(M1, p1)
        LevenbergMarquardt!(M1, vgf1, r1a6)
        @test isapprox(M1, r1a1, r1a6; atol = 1.0e-7)
        # Interface III: Functions, in-place
        r1i1 = LevenbergMarquardt(
            M1, F1!, JF1!, p1, m; evaluation = InplaceEvaluation(),
            function_type = FunctionVectorialType(), jacobian_type = FunctionVectorialType()
        )
        @test isapprox(M1, r1a1, r1i1; atol = 1.0e-7)
        # Interface IV: vgf in-place
        r1i2 = LevenbergMarquardt(M1, vgf1!, p1)
        @test isapprox(M1, r1i1, r1i2; atol = 1.0e-7)
        # try one with accepting early
        r1i3 = LevenbergMarquardt(M1, vgf1!, p1; damping_reduction_threshold = 0.1)
        @test isapprox(M1, r1i1, r1i3; atol = 1.0e-7)

        @testset "Callbacks" begin
            @test Manopt.provided_callbacks(Manopt.LevenbergMarquardtState) == [:Any, :BeforeInit, :BeforeStep, :BeforeStop, :Init, :Step, :Stop, :Stepsize, :DampingIncreaseStepTooLong, :DampingIncreaseModelInadequate, :DampingDecreaseImprovementTooGood, :DampingIncreaseImprovementTooPoor, :CandidateAccept, :CandidateReject]

            sk_record = Tuple{Symbol, Int}[]
            cb(symbol, problem, state, k) = append!(sk_record, [(symbol, k)])
            s1cb = LevenbergMarquardt(
                M1, F1, JF1, p1, m;
                function_type = FunctionVectorialType(), jacobian_type = FunctionVectorialType(),
                callbacks = cb, return_state = true,
            )
            @test s1cb isa Manopt.LevenbergMarquardtState
            @test sk_record[1:6] == [(:BeforeInit, 0), (:Init, 0), (:BeforeStop, 0), (:BeforeStep, 1), (:Stepsize, 1), (:CandidateAccept, 1)]
            @test first.(sk_record[(end - 1):end]) == [:BeforeStop, :Stop]
        end

        # the error is less than the deviation from above
        M1b = Hyperrectangle([-1.0, -1.0], [1.0, 1.0])
        # Then b is out of bounds and we get something where b is on the boundary, namely 1
        # and a is chosen accordingly
        # We have to use the normal coordinates subsolver here then.
        r1c1 = LevenbergMarquardt(M1b, vgf1, p1; sub_state = CoordinatesNormalSystemState(M1b))
        cnss1 = CoordinatesNormalSystemState(M1b)
        @test startswith(repr(cnss1), "CoordinatesNormalSystemState(; ")
        @test startswith(Manopt.status_summary(cnss1), "# Solver state to solve the normal system")
        @test Manopt.status_summary(cnss1; context = :inline) == repr(cnss1)
        r1c0 = LevenbergMarquardt(M1b, vgf1, p1)  # defaults must work on a Hyperrectangle
        @test distance(M1b, r1c0, r1c1) < 1.0e-8
        @test is_point(M1b, r1c1)
        vgf1c = VectorGradientFunction(
            F1, JF1mat, m; evaluation = AllocatingEvaluation(),
            jacobian_type = CoefficientVectorialType(),
        )
        r1c2 = LevenbergMarquardt(
            M1b, vgf1c, p1;
            sub_state = CoordinatesNormalSystemState(M1b), use_unified_basis = true,
        )
        @test isapprox(M1, r1c1, r1c2)
        # the default sub state has to work with `use_unified_basis` as well
        r1c3 = LevenbergMarquardt(M1, vgf1c, p1; use_unified_basis = true)
        @test isapprox(M1, r1a1, r1c3; atol = 1.0e-7)
        # at a corner where the step points outwards in every coordinate the box
        # subsolver finds no admissible stepsize, returns a zero step, and the solver stops
        M1c = Hyperrectangle([-1.0, -1.0], [0.0, 0.0])
        s1c = LevenbergMarquardt(M1c, vgf1, [0.0, 0.0]; return_state = true)
        @test get_solver_result(s1c) == [0.0, 0.0]
        @test s1c.sub_state.last_gcd_result === :not_found
        @test get_count(s1c, :Iterations) == 1
        @test startswith(Manopt.get_reason(s1c), "The algorithm computed a step size (0.0) less than")

        @testset "coordinate surrogate agrees with operator surrogate" begin
            B1 = DefaultOrthonormalBasis(); n1 = length(X1)
            nlso = ManifoldNonlinearLeastSquaresObjective(
                F1, JF1, n1;
                evaluation = AllocatingEvaluation(),
                function_type = FunctionVectorialType(), jacobian_type = FunctionVectorialType(),
            )
            lmso = LevenbergMarquardtLinearSurrogateObjective(nlso; penalty = 1.0e-5)
            @test startswith(Manopt.status_summary(lmso), "A linear surrogate objective for")
            @test startswith(repr(lmso), "LevenbergMarquardtLinearSurrogateObjective(")
            lmcso = Manopt.LevenbergMarquardtLinearSurrogateCoordinatesObjective(
                nlso; penalty = 1.0e-5, basis = B1, jacobian_cache = [zeros(n1, 2) for _ in eachindex(nlso.objective)],
                residuals = zeros(length(X1))
            )
            @test startswith(Manopt.status_summary(lmcso), "A linear surrogate objective in coordinates for")
            @test startswith(repr(lmcso), "LevenbergMarquardtLinearSurrogateCoordinatesObjective(")
            # Coordinate surrogate requires explicit caches, which are normally updated in LM steps.
            get_residuals!(M1, lmcso.value_cache, nlso, p1)
            for (i, o) in enumerate(nlso.objective)
                lmcso.jacobian_cache[i] = get_jacobian(M1, o, p1; basis = B1)
            end
            slso = Manopt.NormalEquationsObjective(lmso)
            slco = Manopt.NormalEquationsObjective(lmcso)
            # Test accessors
            Manopt.set_parameter!(slso, :Penalty, 1.0e-3)
            Manopt.set_parameter!(slco, :Penalty, 1.0e-3)
            @test get_objective(slso) === lmso
            @test get_objective(slco) === lmcso
            @test lmso.penalty == 1.0e-3
            @test lmcso.penalty == 1.0e-3
            d = number_of_coordinates(M1, B1)
            A_lmso = zeros(d, d); A_lmcso = zeros(d, d)
            Manopt.get_linear_operator!(M1, A_lmso, slso, p1, B1)
            Manopt.get_linear_operator!(M1, A_lmcso, slco, p1, B1)
            @test isapprox(A_lmso, A_lmcso; atol = 1.0e-12, rtol = 1.0e-12)
            nvf_lmso = zeros(d)
            nvf_lmcso = zeros(d)
            Manopt.get_normal_vector_field!(M1, nvf_lmso, lmso, p1, B1)
            Manopt.get_normal_vector_field_coord!(M1, nvf_lmcso, lmcso, p1)
            @test isapprox(nvf_lmso, nvf_lmcso; atol = 1.0e-12, rtol = 1.0e-12)
            # Directly test add_normal_vector_field_coord! (no-basis overload that uses mul!).
            len_o = length(nlso.objective[1])
            val_cache = view(lmcso.value_cache, 1:len_o)
            jc = lmcso.jacobian_cache[1]
            nvf_direct = zeros(d)
            Manopt.add_normal_vector_field_coord!(
                M1, nvf_direct, nlso.objective[1], nlso.robustifier[1], p1;
                value_cache = val_cache, jacobian_cache = jc,
                threshold = lmcso.threshold, mode = lmcso.mode,
            )
            @test isapprox(nvf_direct, nvf_lmcso; atol = 1.0e-12, rtol = 1.0e-12)
            # Verify accumulation semantics from mul!(..., true, true).
            seed = fill(0.7, d)
            nvf_acc = copy(seed)
            Manopt.add_normal_vector_field_coord!(
                M1, nvf_acc, nlso.objective[1], nlso.robustifier[1], p1;
                value_cache = val_cache, jacobian_cache = jc,
                threshold = lmcso.threshold, mode = lmcso.mode,
            )
            @test isapprox(nvf_acc, seed .+ nvf_direct; atol = 1.0e-12, rtol = 1.0e-12)

            # Cross-check with the basis overload of add_normal_vector_field_coord!.
            nvf_direct_B = zeros(d)
            Manopt.add_normal_vector_field_coord!(
                M1, nvf_direct_B, nlso.objective[1], nlso.robustifier[1], p1;
                value_cache = val_cache, jacobian_cache = jc,
                threshold = lmcso.threshold, mode = lmcso.mode,
            )
            @test isapprox(nvf_direct_B, nvf_direct; atol = 1.0e-12, rtol = 1.0e-12)
            n_res = Manopt.residuals_count(nlso)
            vf_lmso = zeros(n_res)
            vf_lmcso = zeros(n_res)
            Manopt.get_vector_field!(M1, vf_lmso, lmso, p1)
            Manopt.get_vector_field!(M1, vf_lmcso, lmcso, p1)
            @test isapprox(vf_lmso, vf_lmcso; atol = 1.0e-12, rtol = 1.0e-12)
            TpM1 = TangentSpace(M1, p1)
            X0 = Manopt.ZeroVector()
            cX = [0.3, -0.5]
            X = get_vector(M1, p1, cX, B1)
            @test isapprox(get_cost(TpM1, slso, X0), get_cost(TpM1, slco, X0); atol = 1.0e-12, rtol = 1.0e-12)
            @test isapprox(get_cost(TpM1, slso, X), get_cost(TpM1, slco, X); atol = 1.0e-12, rtol = 1.0e-12)

            # Coordinate normal operator action should match the assembled normal matrix.
            c_lmso = A_lmso * cX
            c_lmcso = zeros(d)
            Manopt.add_normal_linear_operator_coord!(M1, c_lmcso, lmcso, p1, cX)
            @test isapprox(c_lmso, c_lmcso; atol = 1.0e-12, rtol = 1.0e-12)

            # Coordinate residual-space operator action should match operator-form action.
            y_lmso = zeros(n_res)
            Manopt.get_linear_operator!(M1, y_lmso, lmso, p1, X)
            y_lmcso = zeros(n_res)
            Manopt.add_linear_operator_coord!(M1, y_lmcso, lmcso, p1, cX)
            @test isapprox(y_lmso, y_lmcso; atol = 1.0e-12, rtol = 1.0e-12)

            # Symmetric system coordinate RHS is minus the coordinate normal vector field.
            rhs_slco = zeros(d)
            Manopt.get_vector_field!(M1, rhs_slco, slco, p1, B1)
            @test isapprox(rhs_slco, -nvf_lmcso; atol = 1.0e-12, rtol = 1.0e-12)

            # Coordinate linear-system solution coefficients map back to the right tangent vector.
            dmp = DefaultManoptProblem(TpM1, slco)
            cnss = Manopt.solve!(dmp, CoordinatesNormalSystemState(M1, p1; basis = B1))
            X_sub = get_vector(M1, p1, cnss.c, B1)
            @test isapprox(M1, p1, get_solver_result(dmp, cnss), X_sub; atol = 1.0e-12, rtol = 1.0e-12)
        end

        @testset "coordinate surrogate robustified high-damping regression" begin
            B2 = DefaultOrthonormalBasis(); n = length(X1)
            c2X = [0.3, -0.5]
            X2 = get_vector(M1, p1, c2X, B2)
            penalty = 1.0e3

            for r in (
                    CauchyRobustifier(), SoftL1Robustifier(),
                    ComponentwiseRobustifierFunction(CauchyRobustifier()),
                    ComponentwiseRobustifierFunction(SoftL1Robustifier()),
                )
                vgf = VectorGradientFunction(
                    F1, JF1mat, n; function_type = FunctionVectorialType(),
                    jacobian_type = CoefficientVectorialType(B2),
                )
                # Build as a single block with one robustifier (not componentwise wrapping).
                nlso = ManifoldNonlinearLeastSquaresObjective([vgf], [r])
                lmso = LevenbergMarquardtLinearSurrogateObjective(nlso; penalty = penalty)
                lmso_normal = LevenbergMarquardtLinearSurrogateObjective(nlso; penalty = penalty, mode = :Normal)
                lmcso = Manopt.LevenbergMarquardtLinearSurrogateCoordinatesObjective(
                    nlso;
                    penalty = penalty, basis = B2,
                    jacobian_cache = [zeros(n, 2) for _ in eachindex(nlso.objective)],
                    residuals = zeros(n),
                )
                # Coordinate surrogate requires explicit caches, which are normally updated in LM steps.
                get_residuals!(M1, lmcso.value_cache, nlso, p1)
                for (i, o) in enumerate(nlso.objective)
                    lmcso.jacobian_cache[i] = get_jacobian(M1, o, p1; basis = B2)
                end

                slso = Manopt.NormalEquationsObjective(lmso)
                slso_normal = Manopt.NormalEquationsObjective(lmso_normal)
                slco = Manopt.NormalEquationsObjective(lmcso)

                d = number_of_coordinates(M1, B2)
                n_res = Manopt.residuals_count(nlso)
                A_lmso = zeros(d, d)
                A_lmcso = zeros(d, d)
                Manopt.get_linear_operator!(M1, A_lmso, slso, p1, B2)
                Manopt.get_linear_operator!(M1, A_lmcso, slco, p1, B2)
                @test isapprox(A_lmso, A_lmcso; atol = 1.0e-12, rtol = 1.0e-12)

                nvf_lmso = zeros(d)
                nvf_lmcso = zeros(d)
                Manopt.get_normal_vector_field!(M1, nvf_lmso, lmso, p1, B2)
                Manopt.get_normal_vector_field_coord!(M1, nvf_lmcso, lmcso, p1)
                @test isapprox(nvf_lmso, nvf_lmcso; atol = 1.0e-12, rtol = 1.0e-12)
                vf_lmso = zeros(n_res)
                vf_lmcso = zeros(n_res)
                Manopt.get_vector_field!(M1, vf_lmso, lmso, p1)
                Manopt.get_vector_field!(M1, vf_lmcso, lmcso, p1)
                @test isapprox(vf_lmso, vf_lmcso; atol = 1.0e-12, rtol = 1.0e-12)

                TpM1 = TangentSpace(M1, p1)
                X0 = Manopt.ZeroVector()
                @test isapprox(get_cost(TpM1, slso, X0), get_cost(TpM1, slco, X0); atol = 1.0e-12, rtol = 1.0e-12)
                # The LM-relevant regression: both surrogate systems should produce the same step.
                dmp_so = DefaultManoptProblem(TpM1, slso)
                dmp_so_normal = DefaultManoptProblem(TpM1, slso_normal)
                dmp_co = DefaultManoptProblem(TpM1, slco)
                cnss_so = Manopt.solve!(dmp_so, CoordinatesNormalSystemState(M1, p1; basis = B2))
                cnss_so_normal = Manopt.solve!(dmp_so_normal, CoordinatesNormalSystemState(M1, p1; basis = B2))
                cnss_co = Manopt.solve!(dmp_co, CoordinatesNormalSystemState(M1, p1; basis = B2))
                @test isapprox(cnss_so.c, cnss_co.c; atol = 1.0e-12, rtol = 1.0e-12)
                @test !isapprox(cnss_so_normal.c, cnss_co.c; atol = 1.0e-12, rtol = 1.0e-12)
                @test isapprox(
                    M1, p1, get_solver_result(dmp_so, cnss_so), get_solver_result(dmp_co, cnss_co);
                    atol = 1.0e-12, rtol = 1.0e-12,
                )
            end
        end
    end
    @testset "Robust Geodesic Regression on the Sphere" begin
        # Testing the case of one vector function and a single robustifier, so that it is applied componentwise
        M2 = Manifolds.Sphere(2); p1 = [0.0, 0.0, 1.0]; p2 = [0.0, 1.0, 0.0]
        ts = [0.0, 1 / 3, 2 / 3, 1.0]
        qs = shortest_geodesic(M2, p1, p2, ts)
        # Move the middle two “east”, the other two “west”
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
        )
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

        @testset "mode selection" begin
            (o2_ns, s2_ns) = LevenbergMarquardt(
                TM2, vgf2, P0;
                robustifier = 1.0e-4 ∘ HuberRobustifier(), return_objective = true, return_state = true,
                retraction_method = StabilizedRetraction(default_retraction_method(TM2)),
                scaling_mode = :Normal,
                sub_state = CoordinatesNormalSystemState(TM2),
            )

            (o2_ns_ub, s2_ns_ub) = LevenbergMarquardt(
                TM2, vgf2, P0;
                robustifier = 1.0e-4 ∘ HuberRobustifier(), return_objective = true, return_state = true,
                retraction_method = StabilizedRetraction(default_retraction_method(TM2)),
                scaling_mode = :Normal,
                sub_state = CoordinatesNormalSystemState(TM2),
                use_unified_basis = true,
            )

            @test isapprox(TM2, s2.p, s2_ns.p; atol = 1.0e-2)
            @test isapprox(TM2, s2_ns.p, s2_ns_ub.p; atol = 1.0e-2)
            # due to different scaling they should be a bit different
            @test !isapprox(TM2, s2.p, s2_ns.p; atol = 1.0e-8)
            @test !isapprox(TM2, s2.p, s2_ns_ub.p; atol = 1.0e-8)
        end

        @testset "Block Robust Geodesic Regression on the Sphere" begin
            # We group the 4 points from before into start/end (nonrobust) and middle (robust)
            b1 = [1, 4]; m1 = length(b1)
            b2 = [2, 3]; m2 = length(b2)
            vgf2b1 = VectorGradientFunction(
                (TM, P) -> F2(TM, P; time = ts[b1], data = ps[b1]), (TM, P) -> JF2(TM, P; time = ts[b1], data = ps[b1]), m1;
                evaluation = AllocatingEvaluation(), function_type = FunctionVectorialType(), jacobian_type = FunctionVectorialType(),
            )
            vgf2b2 = VectorGradientFunction(
                (TM, P) -> F2(TM, P; time = ts[b2], data = ps[b2]), (TM, P) -> JF2(TM, P; time = ts[b2], data = ps[b2]), m2;
                evaluation = AllocatingEvaluation(), function_type = FunctionVectorialType(), jacobian_type = FunctionVectorialType(),
            )
            P2c = LevenbergMarquardt(TM2, [vgf2b1, vgf2b2], P0; robustifier = [IdentityRobustifier(), 1.0e-4 ∘ HuberRobustifier()])
            P2d = copy(TM2, P0)
            LevenbergMarquardt!(TM2, [vgf2b1, vgf2b2], P2d; robustifier = [IdentityRobustifier(), 1.0e-4 ∘ HuberRobustifier()])
            @test isapprox(M2, P2c[TM2, :point], P2d[TM2, :point]; atol = 1.0e-5)
            @test norm(P2c[TM2, :vector] - P2d[TM2, :vector]) < 1.0e-4
        end
        @testset "show/repr on the LevenbergMarquardt state on NL objective" begin
            @test startswith(repr(o2), "ManifoldNonlinearLeastSquaresObjective(")
            @test Manopt.status_summary(o2) == "A nonlinear least squares objective with 1 vectorial block"
            @test startswith(repr(s2), "LevenbergMarquardtState(")
            @test startswith(Manopt.status_summary(s2), "# Solver state for `Manopt.jl`s Levenberg Marquardt Algorithm")
        end
        @testset "jacobian_tangent_basis is honoured by both variants" begin
            Ml = Euclidean(2)
            xs = [0.0, 1.0, 2.0, 3.0]
            ys = [0.5, 2.4, 4.8, 6.9]
            fl(M, p) = [p[1] + p[2] * x - y for (x, y) in zip(xs, ys)]
            # the Jacobian is given in a basis with the two coordinates swapped
            Bl = CachedBasis(DefaultOrthonormalBasis(), [[0.0, 1.0], [1.0, 0.0]])
            jac_l(M, p) = hcat([x for x in xs], ones(length(xs)))
            pl0 = [1.0, 1.0]
            pa = LevenbergMarquardt(Ml, fl, jac_l, pl0, length(xs); jacobian_tangent_basis = Bl)
            pi_ = LevenbergMarquardt!(Ml, fl, jac_l, copy(pl0), length(xs); jacobian_tangent_basis = Bl)
            @test isapprox(pa, pi_; atol = 1.0e-8)
        end
    end
    @testset "Jacobian cache shapes" begin
        M = Euclidean(2)
        X1 = [1.0, 2.0, 3.0]; Y1 = [2.6, 2.9, 3.5]; m = 3
        F1(M, p) = [p[1] * x + p[2] - y for (x, y) in zip(X1, Y1)]
        JF1(M, p) = [[x, one(x)] for x in X1]
        vgf = VectorGradientFunction(
            F1, JF1, m; evaluation = AllocatingEvaluation(),
            function_type = FunctionVectorialType(), jacobian_type = FunctionVectorialType(),
        )
        nlso = ManifoldNonlinearLeastSquaresObjective(vgf)
        # the documented default `initial_jacobian_matrices = nothing` yields a runnable state
        lms = LevenbergMarquardtState(
            M, (a...) -> 0, Manopt.ClosedFormSubSolverState(), zeros(m); p = [0.0, 0.0],
        )
        @test isnothing(lms.jacobian_matrices)
        Manopt.initialize_solver!(DefaultManoptProblem(M, nlso), lms)
        @test lms.X == [-18.9, -9.0]
        # a single bare matrix is normalized to one entry per block
        lms2 = LevenbergMarquardtState(
            M, (a...) -> 0, Manopt.ClosedFormSubSolverState(), zeros(m), zeros(m, 2); p = [0.0, 0.0],
        )
        @test lms2.jacobian_matrices isa Vector
        @test length(lms2.jacobian_matrices) == 1
    end
    @testset "the too-long-step rejection clamps to damping_term_max" begin
        M2 = Manifolds.Sphere(2)
        # a tiny Jacobian with a large residual makes the Gauss-Newton step exceed max_stepsize
        Fl(M, p) = [10.0 * p[1], 10.0 * p[2]]
        JFl(M, p) = 1.0e-2 .* [1.0 0.0; 0.0 1.0]
        hits = Int[]
        cb(sym, prob, st, k) = (sym === :DampingIncreaseStepTooLong && push!(hits, k))
        # max is below initial * factor, so the first rejection has to clamp
        s = LevenbergMarquardt(
            M2, Fl, JFl, [1.0, 0.0, 0.0], 2;
            initial_damping_term = 1.0e-6, damping_term_max = 5.0e-6, damping_increase_factor = 10.0,
            stopping_criterion = StopAfterIteration(3), callbacks = cb, return_state = true,
        )
        @test !isempty(hits) # the branch under test was actually taken
        @test s.damping_term == 5.0e-6 # without the clamp this would be 1.0e-5
    end
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
        # damping_reduction_factor too large (≥ 1) – used to raise an `UndefVarError`
        @test_throws ArgumentError LevenbergMarquardtState(
            M, sub_fake_f, sub_state, i_res, i_JF; p = x0, damping_reduction_factor = 1.5,
        )
        # For the evaluating case num_components can not be derived in code, hence this errors
        @test_throws ArgumentError LevenbergMarquardt(
            M, (M, v, p) -> v, (M, X, p) -> X, x0; evaluation = InplaceEvaluation(),
        )
        @test_throws ArgumentError LevenbergMarquardt!(
            M, (M, v, p) -> v, (M, X, p) -> X, x0; evaluation = InplaceEvaluation(),
        )
    end
end
