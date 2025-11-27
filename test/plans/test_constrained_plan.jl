using LRUCache, Manifolds, ManifoldsBase, Manopt, Test, RecursiveArrayTools

@testset "Constrained Plan" begin
    M = ManifoldsBase.DefaultManifold(3)
    # Cost
    f(::ManifoldsBase.DefaultManifold, p) = norm(p)^2
    grad_f(M, p) = 2 * p
    grad_f!(M, X, p) = (X .= 2 * p)
    hess_f(M, p, X) = [2.0, 2.0, 2.0]
    hess_f!(M, Y, p, X) = (Y .= [2.0, 2.0, 2.0])
    # Inequality constraints
    g(M, p) = [p[1] - 1, -p[2] - 1]
    g!(M, V, p) = (V .= [p[1] - 1, -p[2] - 1])
    # # Function
    grad_g(M, p) = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    grad_gA(M, p) = [1.0 0.0; 0.0 -1.0; 0.0 0.0]
    hess_g(M, p, X) = [copy(X), -copy(X)]
    hess_g!(M, Y, p, X) = (Y .= [copy(X), -copy(X)])
    function grad_g!(M, X, p)
        X[1] .= [1.0, 0.0, 0.0]
        X[2] .= [0.0, -1.0, 0.0]
        return X
    end
    # vectorial
    g1(M, p) = p[1] - 1
    grad_g1(M, p) = [1.0, 0.0, 0.0]
    grad_g1!(M, X, p) = (X .= [1.0, 0.0, 0.0])
    hess_g1(M, p, X) = copy(X)
    hess_g1!(M, Y, p, X) = copyto!(Y, X)
    g2(M, p) = -p[2] - 1
    grad_g2(M, p) = [0.0, -1.0, 0.0]
    grad_g2!(M, X, p) = (X .= [0.0, -1.0, 0.0])
    hess_g2(M, p, X) = copy(-X)
    hess_g2!(M, Y, p, X) = copyto!(Y, -X)
    @test Manopt._number_of_constraints(
        nothing, [grad_g1, grad_g2]; jacobian_type = ComponentVectorialType()
    ) == 2
    @test Manopt._number_of_constraints(
        [g1, g2], nothing; jacobian_type = ComponentVectorialType()
    ) == 2
    # Equality Constraints
    h(M, p) = [2 * p[3] - 1]
    h!(M, V, p) = (V .= [2 * p[3] - 1])
    h1(M, p) = 2 * p[3] - 1
    grad_h(M, p) = [[0.0, 0.0, 2.0]]
    grad_hA(M, p) = [[0.0, 0.0, 2.0];;]
    function grad_h!(M, X, p)
        X[1] .= [0.0, 0.0, 2.0]
        return X
    end
    hess_h(M, p, X) = [[0.0, 0.0, 0.0]]
    hess_h!(M, Y, p, X) = (Y .= [[0.0, 0.0, 0.0]])
    grad_h1(M, p) = [0.0, 0.0, 2.0]
    grad_h1!(M, X, p) = (X .= [0.0, 0.0, 2.0])
    hess_h1(M, p, X) = [0.0, 0.0, 0.0]
    hess_h1!(M, Y, p, X) = (Y .= [0.0, 0.0, 0.0])

    #A set of values for an example point and tangent
    p = [1.0, 2.0, 3.0]
    c = [[0.0, -3.0], [5.0]]
    fp = 14.0
    gg = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    gh = [[0.0, 0.0, 2.0]]
    gf = 2 * p
    X = [1.0, 0.0, 0.0]
    hf = [2.0, 2.0, 2.0]
    hg = [X, -X]
    hh = [[0.0, 0.0, 0.0]]

    cofa = ConstrainedManifoldObjective(
        f, grad_f, g, grad_g, h, grad_h; inequality_constraints = 2, equality_constraints = 1
    )
    cofaA = ConstrainedManifoldObjective( # Array representation tangent vector
        f,
        grad_f,
        g,
        grad_gA,
        h,
        grad_hA;
        inequality_constraints = 2,
        equality_constraints = 1,
    )
    cofm = ConstrainedManifoldObjective(
        f,
        grad_f!,
        g!,
        grad_g!,
        h!,
        grad_h!;
        evaluation = InplaceEvaluation(),
        inequality_constraints = 2,
        equality_constraints = 1,
    )
    cova = ConstrainedManifoldObjective(
        f,
        grad_f,
        [g1, g2],
        [grad_g1, grad_g2],
        [h1],
        [grad_h1];
        inequality_constraints = 2,
        equality_constraints = 1,
    )
    covm = ConstrainedManifoldObjective(
        f,
        grad_f!,
        [g1, g2],
        [grad_g1!, grad_g2!],
        [h1],
        [grad_h1!];
        evaluation = InplaceEvaluation(),
        inequality_constraints = 2,
        equality_constraints = 1,
    )
    @test repr(cofa) === "ConstrainedManifoldObjective{AllocatingEvaluation}"
    @test repr(cofm) === "ConstrainedManifoldObjective{InplaceEvaluation}"
    @test repr(cova) === "ConstrainedManifoldObjective{AllocatingEvaluation}"
    @test repr(covm) === "ConstrainedManifoldObjective{InplaceEvaluation}"
    # Test cost/grad pass through
    @test Manopt.get_cost_function(cofa)(M, p) == f(M, p)
    @test Manopt.get_gradient_function(cofa)(M, p) == grad_f(M, p)
    @testset "lengths" begin
        @test equality_constraints_length(cofa) == 1
        @test inequality_constraints_length(cofa) == 2
        cofE = ConstrainedManifoldObjective(
            f, grad_f, nothing, nothing, h, grad_h; equality_constraints = 1
        )

        cofI = ConstrainedManifoldObjective(
            f, grad_f, g, grad_g, nothing, nothing; inequality_constraints = 2
        )
        @test equality_constraints_length(cofI) == 0
        @test inequality_constraints_length(cofE) == 0
    end

    @test Manopt.get_unconstrained_objective(cofa) isa ManifoldFirstOrderObjective
    cofha = ConstrainedManifoldObjective(
        f,
        grad_f,
        g,
        grad_g,
        h,
        grad_h;
        hess_f = hess_f,
        hess_g = hess_g,
        hess_h = hess_h,
        inequality_constraints = 2,
        equality_constraints = 1,
    )
    cofhm = ConstrainedManifoldObjective(
        f,
        grad_f!,
        g!,
        grad_g!,
        h!,
        grad_h!;
        hess_f = (hess_f!),
        hess_g = (hess_g!),
        hess_h = (hess_h!),
        evaluation = InplaceEvaluation(),
        inequality_constraints = 2,
        equality_constraints = 1,
    )
    covha = ConstrainedManifoldObjective(
        f,
        grad_f,
        [g1, g2],
        [grad_g1, grad_g2],
        [h1],
        [grad_h1];
        hess_f = hess_f,
        hess_g = [hess_g1, hess_g2],
        hess_h = [hess_h1],
        inequality_constraints = 2,
        equality_constraints = 1,
    )
    covhm = ConstrainedManifoldObjective(
        f,
        grad_f!,
        [g1, g2],
        [grad_g1!, grad_g2!],
        [h1],
        [grad_h1!];
        hess_f = (hess_f!),
        hess_g = [hess_g1!, hess_g2!],
        hess_h = [hess_h1!],
        evaluation = InplaceEvaluation(),
        inequality_constraints = 2,
        equality_constraints = 1,
    )

    mp = DefaultManoptProblem(M, cofha)
    cop = ConstrainedManoptProblem(M, cofha)
    cop2 = ConstrainedManoptProblem(
        M,
        cofaA;
        gradient_equality_range = ArrayPowerRepresentation(),
        gradient_inequality_range = ArrayPowerRepresentation(),
    )

    @testset "ConstrainedManoptProblem special cases" begin
        Y = zero_vector(M, p)
        for mcp in [mp, cop]
            @test get_equality_constraint(mcp, p, :) == c[2]
            @test get_inequality_constraint(mcp, p, :) == c[1]
            @test get_grad_equality_constraint(mcp, p, :) == gh
            @test get_grad_inequality_constraint(mcp, p, :) == gg
            get_grad_equality_constraint!(mcp, Y, p, 1)
            @test Y == gh[1]
            get_grad_inequality_constraint!(mcp, Y, p, 1)
            @test Y == gg[1]
            #
            @test get_hess_equality_constraint(mcp, p, X, :) == hh
            @test get_hess_inequality_constraint(mcp, p, X, :) == hg
            get_hess_equality_constraint!(mcp, Y, p, X, 1)
            @test Y == hh[1]
            get_hess_inequality_constraint!(mcp, Y, p, X, 1)
            @test Y == hg[1]
        end
        #
        @test get_equality_constraint(cop2, p, :) == c[2]
        @test get_inequality_constraint(cop2, p, :) == c[1]
        @test get_grad_equality_constraint(cop2, p, :) == cat(gh...; dims = 2)
        @test get_grad_inequality_constraint(cop2, p, :) == cat(gg...; dims = 2)
        get_grad_equality_constraint!(cop2, Y, p, 1)
        @test Y == gh[1]
        get_grad_inequality_constraint!(cop2, Y, p, 1)
        @test Y == gg[1]
    end
    @testset "ConstrainedObjective with Hessian" begin
        # Function access
        @test Manopt.get_hessian_function(cofha) == hess_f
        @test Manopt.get_hessian_function(cofhm) == hess_f!
        @test Manopt.get_hessian_function(covha) == hess_f
        @test Manopt.get_hessian_function(covhm) == hess_f!
        for coh in [cofha, cofhm, covha, covhm]
            @testset "Hessian access for $coh" begin
                @test get_hessian(M, coh, p, X) == hf
                Y = zero_vector(M, p)
                get_hessian!(M, Y, coh, p, X) == hf
                @test Y == hf
                #
                @test get_hess_equality_constraint(M, coh, p, X) == hh
                @test get_hess_equality_constraint(M, coh, p, X, :) == hh
                @test get_hess_equality_constraint(M, coh, p, X, 1:1) == hh
                @test get_hess_equality_constraint(M, coh, p, X, 1) == hh[1]
                Ye = [zero_vector(M, p)]
                get_hess_equality_constraint!(M, Ye, coh, p, X)
                @test Ye == hh
                get_hess_equality_constraint!(M, Ye, coh, p, X, :)
                @test Ye == hh
                get_hess_equality_constraint!(M, Ye, coh, p, X, 1:1)
                @test Ye == hh
                get_hess_equality_constraint!(M, Y, coh, p, X, 1)
                @test Y == hh[1]
                #
                @test get_hess_inequality_constraint(M, coh, p, X) == hg
                @test get_hess_inequality_constraint(M, coh, p, X, :) == hg
                @test get_hess_inequality_constraint(M, coh, p, X, 1:2) == hg
                @test get_hess_inequality_constraint(M, coh, p, X, 1) == hg[1]
                @test get_hess_inequality_constraint(M, coh, p, X, 2) == hg[2]
                Yi = [zero_vector(M, p), zero_vector(M, p)]
                get_hess_inequality_constraint!(M, Yi, coh, p, X)
                @test Yi == hg
                get_hess_inequality_constraint!(M, Yi, coh, p, X, :)
                @test Yi == hg
                get_hess_inequality_constraint!(M, Yi, coh, p, X, 1:2)
                @test Yi == hg
                get_hess_inequality_constraint!(M, Y, coh, p, X, 1)
                @test Y == hg[1]
                get_hess_inequality_constraint!(M, Y, coh, p, X, 2)
                @test Y == hg[2]
            end
        end
    end
    @testset "Partial Constructors" begin
        # At least one constraint necessary
        @test_throws ErrorException ConstrainedManifoldObjective(f, grad_f)
        @test_throws ErrorException ConstrainedManifoldObjective(
            f, grad_f!; evaluation = InplaceEvaluation()
        )
        co1f = ConstrainedManifoldObjective(
            f, grad_f!; g = g, grad_g = grad_g, hess_g = hess_g, M = M
        )
        @test get_equality_constraint(M, co1f, p, :) == []
        @test get_inequality_constraint(M, co1f, p, :) == c[1]
        @test get_grad_equality_constraint(M, co1f, p, :) == []
        @test get_grad_inequality_constraint(M, co1f, p, :) == gg
        @test get_hess_equality_constraint(M, co1f, p, X, :) == []
        @test get_hess_inequality_constraint(M, co1f, p, X, :) == hg

        co1v = ConstrainedManifoldObjective(
            f, grad_f!; g = [g1, g2], grad_g = [grad_g1, grad_g2], hess_g = [hess_g1, hess_g2]
        )
        @test get_equality_constraint(M, co1v, p, :) == []
        @test get_inequality_constraint(M, co1v, p, :) == c[1]
        @test get_grad_equality_constraint(M, co1v, p, :) == []
        @test get_grad_inequality_constraint(M, co1v, p, :) == gg
        @test get_hess_equality_constraint(M, co1f, p, X, :) == []
        @test get_hess_inequality_constraint(M, co1f, p, X, :) == hg

        co2f = ConstrainedManifoldObjective(
            f, grad_f!; h = h, grad_h = grad_h, hess_h = hess_h, M = M
        )
        @test get_equality_constraint(M, co2f, p, :) == c[2]
        @test get_inequality_constraint(M, co2f, p, :) == []
        @test get_grad_equality_constraint(M, co2f, p, :) == gh
        @test get_grad_inequality_constraint(M, co2f, p, :) == []
        @test get_hess_equality_constraint(M, co2f, p, X, :) == hh
        @test get_hess_inequality_constraint(M, co2f, p, X, :) == []

        co2v = ConstrainedManifoldObjective(
            f, grad_f!; h = h, grad_h = grad_h, hess_h = hess_h, M = M
        )
        @test get_equality_constraint(M, co2v, p, :) == c[2]
        @test get_inequality_constraint(M, co2v, p, :) == []
        @test get_grad_equality_constraint(M, co2v, p, :) == gh
        @test get_grad_inequality_constraint(M, co2v, p, :) == []
        @test get_hess_equality_constraint(M, co2v, p, X, :) == hh
        @test get_hess_inequality_constraint(M, co2v, p, X, :) == []
    end
    @testset "Gradient access" begin
        for co in [cofa, cofm, cova, covm, cofha, cofhm, covha, covhm]
            @testset "Gradients for $co" begin
                dmp = DefaultManoptProblem(M, co)
                @test get_equality_constraint(dmp, p, :) == c[2]
                @test get_equality_constraint(dmp, p, 1) == c[2][1]
                @test get_inequality_constraint(dmp, p, :) == c[1]
                @test get_inequality_constraint(dmp, p, 1) == c[1][1]
                @test get_inequality_constraint(dmp, p, 2) == c[1][2]

                @test get_grad_equality_constraint(dmp, p, :) == gh
                Xh = [zeros(3)]
                @test get_grad_equality_constraint!(dmp, Xh, p, :) == gh
                @test Xh == gh
                X = zeros(3)
                @test get_grad_equality_constraint(dmp, p, 1) == gh[1]
                @test get_grad_equality_constraint!(dmp, X, p, 1) == gh[1]
                @test X == gh[1]

                @test get_grad_inequality_constraint(dmp, p, :) == gg
                Xg = [zeros(3), zeros(3)]
                @test get_grad_inequality_constraint!(dmp, Xg, p, :) == gg
                @test Xg == gg
                @test get_grad_inequality_constraint(dmp, p, 1) == gg[1]
                @test get_grad_inequality_constraint!(dmp, X, p, 1) == gg[1]
                @test X == gg[1]
                @test get_grad_inequality_constraint(dmp, p, 2) == gg[2]
                @test get_grad_inequality_constraint!(dmp, X, p, 2) == gg[2]
                @test X == gg[2]

                @test get_gradient(dmp, p) == gf
                @test get_gradient!(dmp, X, p) == gf
                @test X == gf
            end
        end
    end
    @testset "is_feasible & DebugFeasibility" begin
        coh = ConstrainedManifoldObjective(
            f,
            grad_f;
            hess_f = hess_f,
            g = g,
            grad_g = grad_g,
            hess_g = hess_g,
            h = h,
            grad_h = grad_h,
            hess_h = hess_h,
            M = M,
        )
        @test is_feasible(M, coh, [-2.0, 3.0, 0.5]; error = :info)
        @test_throws ErrorException is_feasible(M, coh, p; error = :error)
        @test_logs (:info,) !is_feasible(M, coh, p; error = :info)
        @test_logs (:warn,) !is_feasible(M, coh, p; error = :warn)
        st = Manopt.StepsizeState(p, X)
        mp = DefaultManoptProblem(M, coh)
        io = IOBuffer()
        df = DebugFeasibility(; io = io)
        @test repr(df) === "DebugFeasibility([\"feasible: \", :Feasible])"
        # short form:
        @test Manopt.status_summary(df) === "(:Feasibility, [\"feasible: \", :Feasible])"
        df(mp, st, 1)
        @test String(take!(io)) == "feasible: No"
    end
    @testset "Lagrangians" begin
        μ = [1.0, 1.0]
        λ = [1.0]
        β = 7.0
        s = [1.0, 2.0]
        N = M × ℝ^2 × ℝ^1 × ℝ^2
        q = rand(N)
        q[N, 1] = p
        q[N, 2] = μ
        q[N, 3] = λ
        q[N, 4] = s
        coh = ConstrainedManifoldObjective(
            f,
            grad_f;
            hess_f = hess_f,
            g = g,
            grad_g = grad_g,
            hess_g = hess_g,
            h = h,
            grad_h = grad_h,
            hess_h = hess_h,
            M = M,
        )
        @testset "Lagrangian Cost, Grad and Hessian" begin
            Lc = LagrangianCost(coh, μ, λ)
            @test startswith(repr(Lc), "LagrangianCost")
            Lg = LagrangianGradient(coh, μ, λ)
            @test startswith(repr(Lg), "LagrangianGradient")
            Lh = LagrangianHessian(coh, μ, λ)
            @test startswith(repr(Lh), "LagrangianHessian")
            @test Lc(M, p) == f(M, p) + g(M, p)'μ + h(M, p)'λ
            @test Lg(M, p) == gf + sum(gg .* μ) + sum(gh .* λ)
            LX = zero_vector(M, p)
            Lg(M, LX, p)
            @test LX == Lg(M, p)
            @test Lh(M, p, X) == hf + sum(hg .* μ) + sum(hh .* λ)
            Lh(M, LX, p, X)
            @test LX == Lh(M, p, X)
            # Get & Set
            @test Manopt.set_parameter!(Lc, :μ, [2.0, 2.0]) == Lc
            @test Manopt.get_parameter(Lc, :μ) == [2.0, 2.0]
            @test Manopt.set_parameter!(Lc, :λ, [2.0]) == Lc
            @test Manopt.get_parameter(Lc, :λ) == [2.0]

            @test Manopt.set_parameter!(Lg, :μ, [2.0, 2.0]) == Lg
            @test Manopt.get_parameter(Lg, :μ) == [2.0, 2.0]
            @test Manopt.set_parameter!(Lg, :λ, [2.0]) == Lg
            @test Manopt.get_parameter(Lg, :λ) == [2.0]

            @test Manopt.set_parameter!(Lh, :μ, [2.0, 2.0]) == Lh
            @test Manopt.get_parameter(Lh, :μ) == [2.0, 2.0]
            @test Manopt.set_parameter!(Lh, :λ, [2.0]) == Lh
            @test Manopt.get_parameter(Lh, :λ) == [2.0]
        end
        @testset "Full KKT and its norm" begin
            # Full KKT Vector field
            KKTvf = KKTVectorField(coh)
            @test startswith(repr(KKTvf), "KKTVectorField\n")
            Xp = LagrangianGradient(coh, μ, λ)(M, p) #Xμ = g + s; Xλ = h, Xs = μ .* s
            Y = KKTvf(N, q)
            @test Y[N, 1] == Xp
            @test Y[N, 2] == c[1] .+ s
            @test Y[N, 3] == c[2]
            @test Y[N, 4] == μ .* s
            KKTvfJ = KKTVectorFieldJacobian(coh)
            @test startswith(repr(KKTvfJ), "KKTVectorFieldJacobian\n")
            #
            Xp =
                LagrangianHessian(coh, μ, λ)(M, p, Y[N, 1]) +
                sum(Y[N, 2] .* gg) +
                sum(Y[N, 3] .* gh)
            Xμ = [inner(M, p, gg[i], Y[N, 1]) + Y[N, 4][i] for i in 1:length(gg)]
            Xλ = [inner(M, p, gh[j], Y[N, 1]) for j in 1:length(gh)]
            Z = KKTvfJ(N, q, Y)
            @test Z[N, 1] == Xp
            @test Z[N, 2] == Xμ
            @test Z[N, 3] == Xλ
            @test Z[N, 4] == μ .* Y[N, 4] + s .* Y[N, 2]

            KKTvfAdJ = KKTVectorFieldAdjointJacobian(coh)
            @test startswith(repr(KKTvfAdJ), "KKTVectorFieldAdjointJacobian\n")
            Xp2 =
                LagrangianHessian(coh, μ, λ)(M, p, Y[N, 1]) +
                sum(Y[N, 2] .* gg) +
                sum(Y[N, 3] .* gh)
            Xμ2 = [inner(M, p, gg[i], Y[N, 1]) + s[i] * Y[N, 4][i] for i in 1:length(gg)]
            Xλ2 = [inner(M, p, gh[j], Y[N, 1]) for j in 1:length(gh)]
            Z2 = KKTvfAdJ(N, q, Y)
            @test Z2[N, 1] == Xp2
            @test Z2[N, 2] == Xμ2
            @test Z2[N, 3] == Xλ2
            @test Z2[N, 4] == μ .* Y[N, 4] + Y[N, 2]

            # Full KKT Vector field norm – the Merit function
            KKTvfN = KKTVectorFieldNormSq(coh)
            @test startswith(repr(KKTvfN), "KKTVectorFieldNormSq\n")
            vfn = KKTvfN(N, q)
            @test vfn == norm(N, q, Y)^2
            KKTvfNG = KKTVectorFieldNormSqGradient(coh)
            @test startswith(repr(KKTvfNG), "KKTVectorFieldNormSqGradient\n")
            Zg1 = KKTvf(N, q)
            Zg2 = 2.0 * KKTvfAdJ(N, q, Zg1)
            W = KKTvfNG(N, q)
            @test W == Zg2
        end
        @testset "Condensed KKT, Jacobian" begin
            CKKTvf = CondensedKKTVectorField(coh, μ, s, β)
            @test startswith(repr(CKKTvf), "CondensedKKTVectorField\n")
            b1 =
                gf +
                sum(λ .* gh) +
                sum(μ .* gg) +
                sum(((μ ./ s) .* (μ .* (c[1] .+ s) .+ β .- μ .* s)) .* gg)
            b2 = c[2]
            Nc = M × ℝ^1 # (p,λ)
            qc = rand(Nc)
            qc[Nc, 1] = p
            qc[Nc, 2] = λ
            V = CKKTvf(Nc, qc)
            @test V[Nc, 1] == b1
            @test V[Nc, 2] == b2
            V2 = copy(Nc, qc, V)
            CKKTvf(Nc, V2, qc)
            @test V2 == V
            CKKTVfJ = CondensedKKTVectorFieldJacobian(coh, μ, s, β)
            @test startswith(repr(CKKTVfJ), "CondensedKKTVectorFieldJacobian\n")
            Yc = zero_vector(Nc, qc)
            Yc[Nc, 1] = [1.0, 3.0, 5.0]
            Yc[Nc, 2] = [7.0]
            # Compute by hand – somehow the formula is still missing a Y
            Wc = zero_vector(Nc, qc)
            # (1) Hess L + The g sum + the grad g sum
            Wc[N, 1] = hf + sum(hg .* μ) + sum(hh .* Yc[2])
            # (2) grad g terms
            Wc[N, 1] += sum(
                (μ ./ s) .*
                    [inner(Nc[1], qc[N, 1], gg[i], Yc[Nc, 1]) for i in 1:length(gg)] .* gg,
            )
            # (3) grad h terms (note the Y_2 component)
            Wc[N, 1] += sum(Yc[N, 2] .* gh)
            # Second component, just h terms
            Wc[N, 2] = [inner(Nc[1], qc[N, 1], gh[j], Yc[Nc, 1]) for j in 1:length(gh)]
            W = CKKTVfJ(Nc, qc, Yc)
            W2 = copy(Nc, qc, Yc)
            CKKTVfJ(Nc, W2, qc, Yc)
            @test W2 == W
            @test Wc == W
            # get & set
            for ck in [CKKTvf, CKKTVfJ]
                @test Manopt.set_parameter!(ck, :μ, [2.0, 2.0]) == ck
                @test Manopt.get_parameter(ck, :μ) == [2.0, 2.0]
                @test Manopt.set_parameter!(ck, :s, [2.0, 2.0]) == ck
                @test Manopt.get_parameter(ck, :s) == [2.0, 2.0]
                @test Manopt.set_parameter!(ck, :β, 2.0) == ck
                @test Manopt.get_parameter(ck, :β) == 2.0
            end
        end
    end
    @testset "Augmented Lagrangian Cost & Grad" begin
        μ = [1.0, 1.0]
        λ = [1.0]
        ρ = 0.1
        cg = sum(max.([0.0, 0.0], c[1] .+ μ ./ ρ) .^ 2)
        ch = sum((c[2] .+ λ ./ ρ) .^ 2)
        ac = f(M, p) + ρ / 2 * (cg + ch)
        agg = sum((c[1] .* ρ .+ μ) .* gg .* (c[1] .+ μ ./ ρ .> 0))
        agh = sum((c[2] .* ρ .+ λ) .* gh)
        ag = gf + agg + agh
        X = zero_vector(M, p)
        for P in [cofa, cofm, cova, covm]
            @testset "$P" begin
                ALC = AugmentedLagrangianCost(P, ρ, μ, λ)
                @test ALC(M, p) ≈ ac
                gALC = AugmentedLagrangianGrad(P, ρ, μ, λ)
                @test gALC(M, p) ≈ ag
                gALC(M, X, p)
                @test gALC(M, X, p) ≈ ag
                @test Manopt.set_parameter!(ALC, :ρ, 2 * ρ) == ALC
                @test Manopt.get_parameter(ALC, :ρ) == 2 * ρ
                @test Manopt.set_parameter!(gALC, :ρ, 2 * ρ) == gALC
                @test Manopt.get_parameter(gALC, :ρ) == 2 * ρ
            end
        end
    end
    @testset "Exact Penalties Cost & Grad" begin
        u = 1.0
        ρ = 0.1
        for P in [cofa, cofm, cova, covm]
            @testset "$P" begin
                EPCe = ExactPenaltyCost(P, ρ, u; smoothing = LogarithmicSumOfExponentials())
                EPGe = ExactPenaltyGrad(P, ρ, u; smoothing = LogarithmicSumOfExponentials())
                # LogExp Cost
                v1 = sum(u .* log.(1 .+ exp.(c[1] ./ u))) # cost g
                v2 = sum(u .* log.(exp.(c[2] ./ u) .+ exp.(-c[2] ./ u))) # cost h
                @test EPCe(M, p) ≈ f(M, p) + ρ * (v1 + v2)
                # Log exp grad
                vg1 = sum(gg .* (ρ .* exp.(c[1] ./ u) ./ (1 .+ exp.(c[1] ./ u))))
                vg2f =
                    ρ .* (exp.(c[2] ./ u) .- exp.(-c[2] ./ u)) ./
                    (exp.(c[2] ./ u) .+ exp.(-c[2] ./ u))
                vg2 = sum(vg2f .* gh)
                @test EPGe(M, p) == gf + vg1 + vg2
                # Huber Cost
                EPCh = ExactPenaltyCost(P, ρ, u; smoothing = LinearQuadraticHuber())
                EPGh = ExactPenaltyGrad(P, ρ, u; smoothing = LinearQuadraticHuber())
                w1 = sum((c[1] .- u / 2) .* (c[1] .> u)) # g > u
                w2 = sum((c[1] .^ 2 ./ (2 * u)) .* ((c[1] .> 0) .& (c[1] .<= u))) #
                w3 = sum(sqrt.(c[2] .^ 2 .+ u^2))
                @test EPCh(M, p) ≈ f(M, p) + ρ * (w1 + w2 + w3)
                wg1 = sum(gg .* (c[1] .>= u) .* ρ)
                wg2 = sum(gg .* (c[1] ./ u .* (0 .<= c[1] .< u)) .* ρ)
                wg3 = sum(gh .* (c[2] ./ sqrt.(c[2] .^ 2 .+ u^2)) .* ρ)
                @test EPGh(M, p) ≈ gf + wg1 .+ wg2 .+ wg3
            end
        end
    end
    @testset "Objective Decorator passthrough" begin
        for obj in [cofa, cofm, cova, covm, cofha, cofhm, covha, covhm]
            ddo = Manopt.Test.DummyDecoratedObjective(obj)
            @test get_equality_constraint(M, ddo, p, :) ==
                get_equality_constraint(M, obj, p, :)
            @test get_inequality_constraint(M, ddo, p, :) ==
                get_inequality_constraint(M, obj, p, :)
            Xe = get_grad_equality_constraint(M, ddo, p, :)
            Ye = get_grad_equality_constraint(M, obj, p, :)
            @test Ye == Xe
            for i in 1:1 #number of equality constr
                @test get_equality_constraint(M, ddo, p, i) ==
                    get_equality_constraint(M, obj, p, i)
                X = get_grad_equality_constraint(M, ddo, p, i)
                Y = get_grad_equality_constraint(M, obj, p, i)
                @test X == Y
                X = get_grad_equality_constraint!(M, X, ddo, p, i)
                Y = get_grad_equality_constraint!(M, Y, obj, p, i)
                @test X == Y
            end
            for j in 1:2 # for every equality constraint
                @test get_inequality_constraint(M, ddo, p, j) ==
                    get_inequality_constraint(M, obj, p, j)
                X = get_grad_inequality_constraint(M, ddo, p, j)
                Y = get_grad_inequality_constraint(M, obj, p, j)
                @test X == Y
                X = get_grad_inequality_constraint!(M, X, ddo, p, j)
                Y = get_grad_inequality_constraint!(M, Y, obj, p, j)
                @test X == Y
            end
            Xe = get_grad_inequality_constraint(M, ddo, p, :)
            Ye = get_grad_inequality_constraint(M, obj, p, :)
            @test Ye == Xe
            get_grad_inequality_constraint!(M, Xe, ddo, p, :)
            get_grad_inequality_constraint!(M, Ye, obj, p, :)
            @test Ye == Xe

            get_grad_inequality_constraint!(M, Xe, ddo, p, 1:2)
            get_grad_inequality_constraint!(M, Ye, obj, p, 1:2)
            @test Ye == Xe
        end
        for obj in [cofha, cofhm, covha, covhm]
            ddo = Manopt.Test.DummyDecoratedObjective(obj)
            Xe = get_hess_equality_constraint(M, ddo, p, X, :)
            Ye = get_hess_equality_constraint(M, obj, p, X, :)
            @test Ye == Xe
            for i in 1:1 #number of equality constr
                X = get_hess_equality_constraint(M, ddo, p, X, i)
                Y = get_hess_equality_constraint(M, obj, p, X, i)
                @test X == Y
                X = get_hess_equality_constraint!(M, X, ddo, p, X, i)
                Y = get_hess_equality_constraint!(M, Y, obj, p, X, i)
                @test X == Y
            end
            for j in 1:2 # for every equality constraint
                X = get_hess_inequality_constraint(M, ddo, p, X, j)
                Y = get_hess_inequality_constraint(M, obj, p, X, j)
                @test X == Y
                X = get_hess_inequality_constraint!(M, X, ddo, p, X, j)
                Y = get_hess_inequality_constraint!(M, Y, obj, p, X, j)
                @test X == Y
            end
            Xe = get_hess_inequality_constraint(M, ddo, p, X, :)
            Ye = get_hess_inequality_constraint(M, obj, p, X, :)
            @test Ye == Xe
            get_hess_inequality_constraint!(M, Xe, ddo, p, X, :)
            get_hess_inequality_constraint!(M, Ye, obj, p, X, :)
            @test Ye == Xe
            get_hess_inequality_constraint!(M, Xe, ddo, p, X, 1:2)
            get_hess_inequality_constraint!(M, Ye, obj, p, X, 1:2)
            @test Ye == Xe
        end
    end
    @testset "Count Objective" begin
        ccofa = Manopt.objective_count_factory(
            M,
            cofa,
            [
                :InequalityConstraints,
                :InequalityConstraint,
                :EqualityConstraints,
                :EqualityConstraint,
                :GradInequalityConstraints,
                :GradInequalityConstraint,
                :GradEqualityConstraints,
                :GradEqualityConstraint,
            ],
        )
        @test equality_constraints_length(ccofa) == 1
        @test inequality_constraints_length(ccofa) == 2
        @test get_equality_constraint(M, ccofa, p, :) ==
            get_equality_constraint(M, cofa, p, :)
        @test get_count(ccofa, :EqualityConstraints) == 1
        @test get_equality_constraint(M, ccofa, p, 1) ==
            get_equality_constraint(M, cofa, p, 1)
        @test get_count(ccofa, :EqualityConstraint) == 1
        @test get_count(ccofa, :EqualityConstraint, 1) == 1
        @test get_inequality_constraint(M, ccofa, p, :) ==
            get_inequality_constraint(M, cofa, p, :)
        @test get_count(ccofa, :InequalityConstraints) == 1
        @test get_inequality_constraint(M, ccofa, p, 1) ==
            get_inequality_constraint(M, cofa, p, 1)
        @test get_inequality_constraint(M, ccofa, p, 2) ==
            get_inequality_constraint(M, cofa, p, 2)
        @test get_count(ccofa, :InequalityConstraint) == [1, 1]
        @test get_count(ccofa, :InequalityConstraint, 1) == 1
        @test get_count(ccofa, :InequalityConstraint, 2) == 1
        @test get_count(ccofa, :InequalityConstraint, [1, 2, 3]) == -1

        Xe = get_grad_equality_constraint(M, cofa, p, :)
        @test get_grad_equality_constraint(M, ccofa, p, :) == Xe
        Ye = copy.(Ref(M), Ref(p), Xe)
        get_grad_equality_constraint!(M, Ye, ccofa, p, :)
        @test Ye == Xe
        @test get_count(ccofa, :GradEqualityConstraints) == 2
        X = get_grad_equality_constraint(M, cofa, p, 1)
        @test get_grad_equality_constraint(M, ccofa, p, 1) == X
        Y = copy(M, p, X)
        get_grad_equality_constraint!(M, Y, ccofa, p, 1) == X
        @test Y == X
        @test get_count(ccofa, :GradEqualityConstraint) == 2
        @test get_count(ccofa, :GradEqualityConstraint, 1) == 2
        Xi = get_grad_inequality_constraint(M, cofa, p, :)
        @test get_grad_inequality_constraint(M, ccofa, p, :) == Xi
        Yi = copy.(Ref(M), Ref(p), Xi)
        @test get_grad_inequality_constraint!(M, Yi, ccofa, p, :) == Xi
        @test get_count(ccofa, :GradInequalityConstraints) == 2
        X1 = get_grad_inequality_constraint(M, cofa, p, 1)
        @test get_grad_inequality_constraint(M, ccofa, p, 1) == X1
        @test get_grad_inequality_constraint!(M, Y, ccofa, p, 1) == X1
        X2 = get_grad_inequality_constraint(M, cofa, p, 2)
        @test get_grad_inequality_constraint(M, ccofa, p, 2) == X2
        @test get_grad_inequality_constraint!(M, Y, ccofa, p, 2) == X2
        @test get_count(ccofa, :GradInequalityConstraint) == [2, 2]
        @test get_count(ccofa, :GradInequalityConstraint, 1) == 2
        @test get_count(ccofa, :GradInequalityConstraint, 2) == 2
        @test get_count(ccofa, :GradInequalityConstraint, [1, 2, 3]) == -1
        # test vectorial reset
        reset_counters!(ccofa)
        @test get_count(ccofa, :GradInequalityConstraint) == [0, 0]
    end
    @testset "Cache Objective" begin
        cache_and_count = [
            :Constraints,
            :InequalityConstraints,
            :InequalityConstraint,
            :EqualityConstraints,
            :EqualityConstraint,
            :GradEqualityConstraint,
            :GradEqualityConstraints,
            :GradInequalityConstraint,
            :GradInequalityConstraints,
        ]
        ce = get_equality_constraint(M, cofa, p, :)
        ci = get_inequality_constraint(M, cofa, p, :)
        Xe = get_grad_equality_constraint(M, cofa, p, :)
        Xe2 = get_grad_equality_constraint(M, cofa, -p, :)
        Xi = get_grad_inequality_constraint(M, cofa, p, :) #
        Xi2 = get_grad_inequality_constraint(M, cofa, -p, :) #
        Ye = copy.(Ref(M), Ref(p), Xe)
        Yi = copy.(Ref(M), Ref(p), Xi)
        Y = copy(M, p, Xe[1])

        ccofa = Manopt.objective_count_factory(M, cofa, cache_and_count)
        cccofa = Manopt.objective_cache_factory(M, ccofa, (:LRU, cache_and_count))
        # to always trigger fallbacks: a cache that does not cache
        nccofa = Manopt.objective_cache_factory(M, ccofa, (:LRU, Vector{Symbol}()))

        @test get_equality_constraint(M, cccofa, p, :) == ce # counts
        @test get_equality_constraint(M, cccofa, p, :) == ce # cached
        @test get_equality_constraint(M, cccofa, p, [1]) == ce # cached, too

        @test get_count(cccofa, :EqualityConstraints) == 1
        @test get_equality_constraint(M, nccofa, p, [1]) == ce # fallback, too
        @test get_count(cccofa, :EqualityConstraint) == 1

        @test get_equality_constraint(M, cccofa, p, :) == ce # cached
        for i in 1:1
            ce_i = get_equality_constraint(M, cofa, p, i)
            @test get_equality_constraint(M, cccofa, p, i) == ce_i # counts
            @test get_equality_constraint(M, cccofa, p, i) == ce_i # cached
            @test get_count(cccofa, :EqualityConstraint, i) == 2
        end

        # Reset Counter & Cache
        ccofa = Manopt.objective_count_factory(M, cofa, cache_and_count)
        cccofa = Manopt.objective_cache_factory(M, ccofa, (:LRU, cache_and_count))
        # to always trigger fallbacks: a cache that does not cache
        nccofa = Manopt.objective_cache_factory(M, ccofa, (:LRU, Vector{Symbol}()))

        @test get_equality_constraint(M, cccofa, p, 1:1) == ce # counts
        @test get_equality_constraint(M, cccofa, p, 1:1) == ce # cached
        @test get_count(cccofa, :EqualityConstraint) == 1

        # Fill single entry with range
        @test get_inequality_constraint(M, cccofa, p, 1:2) == ci # counts single
        @test get_inequality_constraint(M, cccofa, p, 1:2) == ci # cached single
        @test get_count(cccofa, :InequalityConstraint, 1) == 1
        @test get_count(cccofa, :InequalityConstraint, 1) == 1

        @test get_inequality_constraint(M, cccofa, p, :) == ci # counts
        @test get_inequality_constraint(M, cccofa, p, :) == ci #cached
        @test get_inequality_constraint(M, cccofa, p, 1:2) == ci # cached, too
        @test get_count(cccofa, :InequalityConstraints) == 1
        @test get_inequality_constraint(M, nccofa, p, 1:2) == ci # fallback, counts
        @test get_count(nccofa, :InequalityConstraint, 1) == 2
        @test get_count(nccofa, :InequalityConstraint, 2) == 2
        for j in 1:2
            ci_j = get_inequality_constraint(M, cofa, p, j)
            @test get_inequality_constraint(M, cccofa, p, j) == ci_j # cached
            @test get_count(cccofa, :InequalityConstraint, j) == 2
        end

        get_grad_equality_constraint!(M, Ye, cccofa, p, 1:1) # cache miss on single integer
        @test Ye == Xe
        get_grad_inequality_constraint!(M, Yi, cccofa, p, 1:2) # cache miss on single integer
        @test Yi == Xi

        # Reset Counter & Cache (yet again)
        ccofa = Manopt.objective_count_factory(M, cofa, cache_and_count)
        cccofa = Manopt.objective_cache_factory(M, ccofa, (:LRU, cache_and_count))
        # to always trigger fallbacks: a cache that does not cache
        nccofa = Manopt.objective_cache_factory(M, ccofa, (:LRU, Vector{Symbol}()))
        # Trigger single integer cache misses
        for i in 1:1
            ce_i = get_equality_constraint(M, cofa, p, i)
            @test get_equality_constraint(M, cccofa, p, i) == ce_i # counts
            @test get_equality_constraint(M, cccofa, p, i) == ce_i # cached
            @test get_count(cccofa, :EqualityConstraint, i) == 1
        end
        for j in 1:2
            ci_j = get_inequality_constraint(M, cofa, p, j)
            @test get_inequality_constraint(M, cccofa, p, j) == ci_j # cached
            @test get_count(cccofa, :InequalityConstraint, j) == 1
        end

        @test get_grad_equality_constraint(M, cccofa, p, 1:1) == Xe # counts single
        @test get_grad_equality_constraint(M, cccofa, p, 1:1) == Xe # cached single
        for i in 1:1
            @test get_grad_equality_constraint(M, cccofa, p, i) == Xe[i] #cached
            get_grad_equality_constraint!(M, Y, cccofa, p, i) == Xe[i] # cached
            @test Y == Xe[i]
            @test get_count(cccofa, :GradEqualityConstraint, i) == 1
            get_grad_equality_constraint!(M, Y, cccofa, -p, i) == Xe2[i] # counts
            @test Y == Xe2[i]
            get_grad_equality_constraint!(M, Y, cccofa, -p, i) == Xe2[i] # cached
            @test Y == Xe2[i]
            @test get_grad_equality_constraint(M, cccofa, -p, i) == Xe2[i] #cached
            @test get_count(cccofa, :GradEqualityConstraint, i) == 2
        end
        @test get_grad_equality_constraint(M, cccofa, p, :) == Xe # counts
        @test get_grad_equality_constraint(M, cccofa, p, :) == Xe # cached
        @test get_grad_equality_constraint(M, cccofa, p, 1:1) == Xe # cached, too
        get_grad_equality_constraint!(M, Ye, cccofa, p, 1:1) # cached, too
        @test Ye == Xe
        @test get_grad_equality_constraint(M, nccofa, p, 1:1) == Xe # fallback, counts

        get_grad_equality_constraint!(M, Ye, cccofa, p, :) # cached
        @test Ye == Xe
        @test get_count(ccofa, :GradEqualityConstraints) == 1
        # New point to trigger caches again
        get_grad_equality_constraint!(M, Ye, cccofa, -p, 1:1) # counts, but here single
        @test Ye == Xe2
        get_grad_equality_constraint!(M, Ye, cccofa, -p, 1:1) # cached from single
        @test Ye == Xe2
        @test get_count(cccofa, :GradEqualityConstraint, 1) == 3
        get_grad_equality_constraint!(M, Ye, cccofa, -p, :) # cached
        @test Ye == Xe2
        @test get_grad_equality_constraint(M, cccofa, -p, :) == Xe2 # cached
        @test get_count(cccofa, :GradEqualityConstraint, 1) == 3
        get_grad_equality_constraint!(M, Ye, cccofa, -p, :) # cached
        @test Ye == Xe2
        @test get_count(cccofa, :GradEqualityConstraint, 1) == 3
        get_grad_equality_constraint!(M, Ye, nccofa, -p, 1:1) # fallback, counts
        @test Ye == Xe2
        @test get_count(cccofa, :GradEqualityConstraint, 1) == 4

        @test get_grad_inequality_constraint(M, cccofa, p, 1:2) == Xi # counts single
        @test get_grad_inequality_constraint(M, cccofa, p, 1:2) == Xi # cached single
        get_grad_inequality_constraint!(M, Yi, cccofa, p, 1:2) # cached single
        @test Yi == Xi
        @test get_grad_inequality_constraint(M, nccofa, p, 1:2) == Xi # fallback, counts
        @test get_count(cccofa, :GradInequalityConstraint, 1) == 2
        @test get_count(cccofa, :GradInequalityConstraint, 2) == 2
        for j in 1:2
            @test get_grad_inequality_constraint(M, cccofa, p, j) == Xi[j] # cached
            @test get_count(ccofa, :GradInequalityConstraint, j) == 2
            @test get_grad_inequality_constraint!(M, Y, cccofa, p, j) == Xi[j] # cached
            @test get_count(ccofa, :GradInequalityConstraint, j) == 2
            @test get_grad_inequality_constraint!(M, Y, cccofa, -p, j) == Xi2[j] # counts
            @test get_grad_inequality_constraint(M, cccofa, p, j) == Xi2[j] # cached
            @test get_count(ccofa, :GradInequalityConstraint, j) == 3
        end
        @test get_grad_inequality_constraint(M, cccofa, p, :) == Xi # counts
        @test get_grad_inequality_constraint(M, cccofa, p, 1:2) == Xi # cached from full
        @test get_grad_inequality_constraint(M, cccofa, p, :) == Xi # cached
        @test get_grad_inequality_constraint!(M, Yi, cccofa, p, :) == Xi # cached
        @test Yi == Xi
        @test get_count(cccofa, :GradInequalityConstraints) == 1
        get_grad_inequality_constraint!(M, Yi, cccofa, -p, 1:2) # cached from single
        @test Yi == Xi2
        @test get_count(ccofa, :GradInequalityConstraint, 1) == 3
        @test get_count(ccofa, :GradInequalityConstraint, 2) == 3
        @test get_grad_inequality_constraint!(M, Yi, cccofa, -p, :) == Xi # counts for full
        @test Yi == Xi2
        @test get_grad_inequality_constraint!(M, Yi, cccofa, -p, :) == Xi # cached
        @test Yi == Xi2
        get_grad_inequality_constraint!(M, Yi, cccofa, p, 1:2) # cached from full
        @test Yi == Xi
        @test get_grad_inequality_constraint(M, cccofa, -p, :) == Xi2 # cached
        @test get_count(cccofa, :GradInequalityConstraints) == 2
        get_grad_inequality_constraint!(M, Yi, nccofa, -p, 1:2) # fallback, counts
        @test Yi == Xi2
        @test get_count(ccofa, :GradInequalityConstraint, 1) == 4
        @test get_count(ccofa, :GradInequalityConstraint, 2) == 4

        # Reset Counter & Cache (yet again)
        ccofa = Manopt.objective_count_factory(M, cofa, cache_and_count)
        cccofa = Manopt.objective_cache_factory(M, ccofa, (:LRU, cache_and_count))
        # Trigger single integer cache misses
        for i in 1:1
            @test get_equality_constraint(M, cccofa, p, i) == ce[i] # counts
            @test get_equality_constraint(M, cccofa, p, i) == ce[i] # cached
            @test get_count(cccofa, :EqualityConstraint, i) == 1
        end
        for j in 1:2
            @test get_inequality_constraint(M, cccofa, p, j) == ci[j] # cached
            @test get_count(cccofa, :InequalityConstraint, j) == 1
        end
        for i in 1:1
            @test get_grad_equality_constraint(M, cccofa, p, i) == Xe[i] #cached
            get_grad_equality_constraint!(M, Y, cccofa, p, i) == Xe[i] # cached
            @test Y == Xe[i]
            @test get_count(cccofa, :GradEqualityConstraint, i) == 1
            get_grad_equality_constraint!(M, Y, cccofa, -p, i) == Xe2[i] # counts
            @test Y == Xe2[i]
            get_grad_equality_constraint!(M, Y, cccofa, -p, i) == Xe2[i] # cached
            @test Y == Xe2[i]
            @test get_grad_equality_constraint(M, cccofa, -p, i) == Xe2[i] #cached
            @test get_count(cccofa, :GradEqualityConstraint, i) == 2
        end
        for j in 1:2
            @test get_grad_inequality_constraint(M, cccofa, p, j) == Xi[j] # cached
            @test get_count(ccofa, :GradInequalityConstraint, j) == 1
            @test get_grad_inequality_constraint!(M, Y, cccofa, p, j) == Xi[j] # cached
            @test get_count(ccofa, :GradInequalityConstraint, j) == 1
            get_grad_inequality_constraint!(M, Y, cccofa, -p, j) # counts
            @test Y == Xi[j]
            @test get_grad_inequality_constraint(M, cccofa, -p, j) == Xi2[j] # cached
            @test get_count(ccofa, :GradInequalityConstraint, j) == 2
        end
    end
end
