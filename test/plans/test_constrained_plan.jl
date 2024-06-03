using LRUCache, Manopt, Manifolds, ManifoldsBase, Test

include("../utils/dummy_types.jl")

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
        nothing, [grad_g1, grad_g2]; jacobian_type=ComponentVectorialType()
    ) == 2
    @test Manopt._number_of_constraints(
        [g1, g2], nothing; jacobian_type=ComponentVectorialType()
    ) == 2
    # Equality Constraints
    h(M, p) = [2 * p[3] - 1]
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
    cofa = ConstrainedManifoldObjective(
        f, grad_f, g, grad_g, h, grad_h; inequality_constraints=2, equality_constraints=1
    )
    cofaA = ConstrainedManifoldObjective( # Array representation tangent vector
        f,
        grad_f,
        g,
        grad_gA,
        h,
        grad_hA;
        inequality_constraints=2,
        equality_constraints=1,
    )
    cofm = ConstrainedManifoldObjective(
        f,
        grad_f!,
        g,
        grad_g!,
        h,
        grad_h!;
        evaluation=InplaceEvaluation(),
        inequality_constraints=2,
        equality_constraints=1,
    )
    cova = ConstrainedManifoldObjective(
        f,
        grad_f,
        [g1, g2],
        [grad_g1, grad_g2],
        [h1],
        [grad_h1];
        inequality_constraints=2,
        equality_constraints=1,
    )
    covm = ConstrainedManifoldObjective(
        f,
        grad_f!,
        [g1, g2],
        [grad_g1!, grad_g2!],
        [h1],
        [grad_h1!];
        evaluation=InplaceEvaluation(),
        inequality_constraints=2,
        equality_constraints=1,
    )
    @test repr(cofa) === "ConstrainedManifoldObjective{AllocatingEvaluation}"
    @test repr(cofm) === "ConstrainedManifoldObjective{InplaceEvaluation}"
    @test repr(cova) === "ConstrainedManifoldObjective{AllocatingEvaluation}"
    @test repr(covm) === "ConstrainedManifoldObjective{InplaceEvaluation}"
    @test Manopt.get_cost_function(cofa) === f
    @test Manopt.get_gradient_function(cofa) === grad_f
    @test equality_constraints_length(cofa) == 1
    @test inequality_constraints_length(cofa) == 2
    @test Manopt.get_unconstrained_objective(cofa) isa ManifoldGradientObjective
    cop = ConstrainedManoptProblem(M, cofa)
    cop2 = ConstrainedManoptProblem(
        M,
        cofaA;
        gradient_equality_range=ArrayPowerRepresentation(),
        gradient_inequality_range=ArrayPowerRepresentation(),
    )
    cofha = ConstrainedManifoldObjective(
        f,
        grad_f,
        g,
        grad_g,
        h,
        grad_h;
        hess_f=hess_f,
        hess_g=hess_g,
        hess_h=hess_h,
        inequality_constraints=2,
        equality_constraints=1,
    )
    cofhm = ConstrainedManifoldObjective(
        f,
        grad_f!,
        g,
        grad_g!,
        h,
        grad_h!;
        hess_f=hess_f!,
        hess_g=hess_g!,
        hess_h=hess_h!,
        evaluation=InplaceEvaluation(),
        inequality_constraints=2,
        equality_constraints=1,
    )
    covha = ConstrainedManifoldObjective(
        f,
        grad_f,
        [g1, g2],
        [grad_g1, grad_g2],
        [h1],
        [grad_h1];
        hess_f=hess_f,
        hess_g=[hess_g1, hess_g2],
        hess_h=[hess_h1],
        inequality_constraints=2,
        equality_constraints=1,
    )
    covhm = ConstrainedManifoldObjective(
        f,
        grad_f!,
        [g1, g2],
        [grad_g1!, grad_g2!],
        [h1],
        [grad_h1!];
        hess_f=hess_f!,
        hess_g=[hess_g1!, hess_g2!],
        hess_h=[hess_h1!],
        evaluation=InplaceEvaluation(),
        inequality_constraints=2,
        equality_constraints=1,
    )

    p = [1.0, 2.0, 3.0]
    c = [[0.0, -3.0], [5.0]]
    gg = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    gh = [[0.0, 0.0, 2.0]]
    gf = 2 * p
    X = [1.0, 0.0, 0.0]
    hf = [2.0, 2.0, 2.0]
    hg = [X, -X]
    hh = [[0.0, 0.0, 0.0]]

    @testset "ConstrainedManoptProblem speecial cases" begin
        @test get_equality_constraint(cop, p, :) == c[2]
        @test get_inequality_constraint(cop, p, :) == c[1]
        @test get_grad_equality_constraint(cop, p, :) == gh
        @test get_grad_inequality_constraint(cop, p, :) == gg
        X = zero_vector(M, p)
        get_grad_equality_constraint!(cop, X, p, 1)
        @test X == gh[1]
        get_grad_inequality_constraint!(cop, X, p, 1)
        @test X == gg[1]
        @test get_equality_constraint(cop2, p, :) == c[2]
        @test get_inequality_constraint(cop2, p, :) == c[1]
        @test get_grad_equality_constraint(cop2, p, :) == cat(gh...; dims=2)
        @test get_grad_inequality_constraint(cop2, p, :) == cat(gg...; dims=2)
        get_grad_equality_constraint!(cop2, X, p, 1)
        @test X == gh[1]
        get_grad_inequality_constraint!(cop2, X, p, 1)
        @test X == gg[1]
    end
    @testset "ConstrainedObjective with Hessian" begin
        # Function accessors
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
            f, grad_f!; evaluation=InplaceEvaluation()
        )
        co1f = ConstrainedManifoldObjective(f, grad_f!; g=g, grad_g=grad_g, M=M)
        @test get_grad_equality_constraint(M, co1f, p, :) == []
        @test get_grad_inequality_constraint(M, co1f, p, :) == gg
        @test get_equality_constraint(M, co1f, p, :) == []
        @test get_inequality_constraint(M, co1f, p, :) == c[1]

        co1v = ConstrainedManifoldObjective(
            f, grad_f!; g=[g1, g2], grad_g=[grad_g1, grad_g2]
        )
        @test get_grad_equality_constraint(M, co1v, p, :) == []
        @test get_grad_inequality_constraint(M, co1v, p, :) == gg
        @test get_equality_constraint(M, co1v, p, :) == []
        @test get_inequality_constraint(M, co1v, p, :) == c[1]

        co2f = ConstrainedManifoldObjective(f, grad_f!; h=h, grad_h=grad_h, M=M)
        @test get_grad_equality_constraint(M, co2f, p, :) == gh
        @test get_grad_inequality_constraint(M, co2f, p, :) == []
        @test get_equality_constraint(M, co2f, p, :) == c[2]
        @test get_inequality_constraint(M, co2f, p, :) == []

        co2v = ConstrainedManifoldObjective(f, grad_f!; h=[h1], grad_h=[grad_h1])
        @test get_grad_equality_constraint(M, co2v, p, :) == gh
        @test get_grad_inequality_constraint(M, co2v, p, :) == []
        @test get_equality_constraint(M, co2v, p, :) == c[2]
        @test get_inequality_constraint(M, co2v, p, :) == []
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
            end
        end
    end
    @testset "Exact Penalties Cost & Grad" begin
        u = 1.0
        ρ = 0.1
        for P in [cofa, cofm, cova, covm]
            @testset "$P" begin
                EPCe = ExactPenaltyCost(P, ρ, u; smoothing=LogarithmicSumOfExponentials())
                EPGe = ExactPenaltyGrad(P, ρ, u; smoothing=LogarithmicSumOfExponentials())
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
                EPCh = ExactPenaltyCost(P, ρ, u; smoothing=LinearQuadraticHuber())
                EPGh = ExactPenaltyGrad(P, ρ, u; smoothing=LinearQuadraticHuber())
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
        for obj in [cofa, cofm, cova, covm]
            ddo = DummyDecoratedObjective(obj)
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
