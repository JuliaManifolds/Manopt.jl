using Manopt, ManifoldsBase, Test

include("../utils/dummy_types.jl")

@testset "Constrained Plan" begin
    M = ManifoldsBase.DefaultManifold(3)
    # Cost
    f(::ManifoldsBase.DefaultManifold, p) = norm(p)^2
    grad_f(M, p) = 2 * p
    grad_f!(M, X, p) = (X .= 2 * p)
    # Inequality constraints
    g(M, p) = [p[1] - 1, -p[2] - 1]
    # # Function
    grad_g(M, p) = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    function grad_g!(M, X, p)
        X[1] .= [1.0, 0.0, 0.0]
        X[2] .= [0.0, -1.0, 0.0]
        return X
    end
    # # Vectorial
    g1(M, p) = p[1] - 1
    grad_g1(M, p) = [1.0, 0.0, 0.0]
    grad_g1!(M, X, p) = (X .= [1.0, 0.0, 0.0])
    g2(M, p) = -p[2] - 1
    grad_g2(M, p) = [0.0, -1.0, 0.0]
    grad_g2!(M, X, p) = (X .= [0.0, -1.0, 0.0])
    # Equality Constraints
    h(M, p) = [2 * p[3] - 1]
    h1(M, p) = 2 * p[3] - 1
    grad_h(M, p) = [[0.0, 0.0, 2.0]]
    function grad_h!(M, X, p)
        X[1] .= [0.0, 0.0, 2.0]
        return X
    end
    grad_h1(M, p) = [0.0, 0.0, 2.0]
    grad_h1!(M, X, p) = (X .= [0.0, 0.0, 2.0])
    cofa = ConstrainedManifoldObjective(f, grad_f, g, grad_g, h, grad_h)
    cofm = ConstrainedManifoldObjective(
        f, grad_f!, g, grad_g!, h, grad_h!; evaluation=InplaceEvaluation()
    )
    cova = ConstrainedManifoldObjective(
        f, grad_f, [g1, g2], [grad_g1, grad_g2], [h1], [grad_h1]
    )
    covm = ConstrainedManifoldObjective(
        f,
        grad_f!,
        [g1, g2],
        [grad_g1!, grad_g2!],
        [h1],
        [grad_h1!];
        evaluation=InplaceEvaluation(),
    )
    @test repr(cofa) ===
        "ConstrainedManifoldObjective{AllocatingEvaluation,FunctionConstraint}."
    @test repr(cofm) ===
        "ConstrainedManifoldObjective{InplaceEvaluation,FunctionConstraint}."
    @test repr(cova) ===
        "ConstrainedManifoldObjective{AllocatingEvaluation,VectorConstraint}."
    @test repr(covm) === "ConstrainedManifoldObjective{InplaceEvaluation,VectorConstraint}."

    p = [1.0, 2.0, 3.0]
    c = [[0.0, -3.0], [5.0]]
    gg = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    gh = [[0.0, 0.0, 2.0]]
    gf = 2 * p

    @testset "Partial Constructors" begin
        # At least one constraint necessary
        @test_throws ErrorException ConstrainedManifoldObjective(f, grad_f)
        @test_throws ErrorException ConstrainedManifoldObjective(
            f, grad_f!; evaluation=InplaceEvaluation()
        )
        co1f = ConstrainedManifoldObjective(f, grad_f!; g=g, grad_g=grad_g)
        @test get_constraints(M, co1f, p) == [c[1], []]
        @test get_grad_equality_constraints(M, co1f, p) == []
        @test get_grad_inequality_constraints(M, co1f, p) == gg

        co1v = ConstrainedManifoldObjective(
            f, grad_f!; g=[g1, g2], grad_g=[grad_g1, grad_g2]
        )
        @test get_constraints(M, co1v, p) == [c[1], []]
        @test get_grad_equality_constraints(M, co1v, p) == []
        @test get_grad_inequality_constraints(M, co1v, p) == gg

        co2f = ConstrainedManifoldObjective(f, grad_f!; h=h, grad_h=grad_h)
        @test get_constraints(M, co2f, p) == [[], c[2]]
        @test get_grad_equality_constraints(M, co2f, p) == gh
        @test get_grad_inequality_constraints(M, co2f, p) == []

        co2v = ConstrainedManifoldObjective(f, grad_f!; h=[h1], grad_h=[grad_h1])
        @test get_constraints(M, co2v, p) == [[], c[2]]
        @test get_grad_equality_constraints(M, co2v, p) == gh
        @test get_grad_inequality_constraints(M, co2v, p) == []
    end

    for co in [cofa, cofm, cova, covm]
        @testset "$co" begin
            dmp = DefaultManoptProblem(M, co)
            @test get_constraints(dmp, p) == c
            @test get_equality_constraints(dmp, p) == c[2]
            @test get_equality_constraint(dmp, p, 1) == c[2][1]
            @test get_inequality_constraints(dmp, p) == c[1]
            @test get_inequality_constraint(dmp, p, 1) == c[1][1]
            @test get_inequality_constraint(dmp, p, 2) == c[1][2]

            @test get_grad_equality_constraints(dmp, p) == gh
            Xh = [zeros(3)]
            @test get_grad_equality_constraints!(dmp, Xh, p) == gh
            @test Xh == gh
            X = zeros(3)
            @test get_grad_equality_constraint(dmp, p, 1) == gh[1]
            @test get_grad_equality_constraint!(dmp, X, p, 1) == gh[1]
            @test X == gh[1]

            @test get_grad_inequality_constraints(dmp, p) == gg
            Xg = [zeros(3), zeros(3)]
            @test get_grad_inequality_constraints!(dmp, Xg, p) == gg
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
    @testset "Exact Penaltiy Cost & Grad" begin
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
    @testset "Objetive Decorator passthrough" begin
        for obj in [cofa, cofm, cova, covm]
            ddo = DummyDecoratedObjective(obj)
            @test get_constraints(M, ddo, p) == get_constraints(M, obj, p)
            @test get_equality_constraints(M, ddo, p) == get_equality_constraints(M, obj, p)
            @test get_inequality_constraints(M, ddo, p) ==
                get_inequality_constraints(M, obj, p)
            Xe = get_grad_equality_constraints(M, ddo, p)
            Ye = get_grad_equality_constraints(M, obj, p)
            @test Ye == Xe
            get_grad_equality_constraints!(M, Xe, ddo, p)
            get_grad_equality_constraints!(M, Ye, obj, p)
            @test Ye == Xe
            for i in 1:1 #num of equ constr
                @test get_equality_constraint(M, ddo, p, i) ==
                    get_equality_constraint(M, obj, p, i)
                X = get_grad_equality_constraint(M, ddo, p, i)
                Y = get_grad_equality_constraint(M, obj, p, i)
                @test X == Y
                X = get_grad_equality_constraint!(M, X, ddo, p, i)
                Y = get_grad_equality_constraint!(M, Y, obj, p, i)
                @test X == Y
            end
            for j in 1:2 # num eq constr
                @test get_inequality_constraint(M, ddo, p, j) ==
                    get_inequality_constraint(M, obj, p, j)
                X = get_grad_inequality_constraint(M, ddo, p, j)
                Y = get_grad_inequality_constraint(M, obj, p, j)
                @test X == Y
                X = get_grad_inequality_constraint!(M, X, ddo, p, j)
                Y = get_grad_inequality_constraint!(M, Y, obj, p, j)
                @test X == Y
            end
            Xe = get_grad_inequality_constraints(M, ddo, p)
            Ye = get_grad_inequality_constraints(M, obj, p)
            @test Ye == Xe
            get_grad_inequality_constraints!(M, Xe, ddo, p)
            get_grad_inequality_constraints!(M, Ye, obj, p)
            @test Ye == Xe
        end
    end
    @testset "Counting" begin
        ccofa = Manopt.objective_count_factory(
            M,
            cofa,
            [
                :Constraints,
                :InequalityConstraints,
                :InequalityConstraint,
                :EqualityConstraints,
                :EqualityConstraint,
            ],
        )
        @test get_constraints(M, ccofa, p) == get_constraints(M, cofa, p)
        @test get_count(ccofa, :Constraints) == 1
        @test get_equality_constraints(M, ccofa, p) == get_equality_constraints(M, cofa, p)
        @test get_count(ccofa, :EqualityConstraints) == 1
        @test get_equality_constraint(M, ccofa, p, 1) ==
            get_equality_constraint(M, cofa, p, 1)
        @test get_count(ccofa, :EqualityConstraint) == 1
        @test get_count(ccofa, :EqualityConstraint, 1) == 1
        @test get_inequality_constraints(M, ccofa, p) ==
            get_inequality_constraints(M, cofa, p)
        @test get_count(ccofa, :InequalityConstraints) == 1
        @test get_inequality_constraint(M, ccofa, p, 1) ==
            get_inequality_constraint(M, cofa, p, 1)
        @test get_inequality_constraint(M, ccofa, p, 2) ==
            get_inequality_constraint(M, cofa, p, 2)
        @test get_count(ccofa, :InequalityConstraint) == [1, 1]
        @test get_count(ccofa, :InequalityConstraint, 1) == 1
        @test get_count(ccofa, :InequalityConstraint, 1) == 1
        @test get_count(ccofa, :InequalityConstraint, [1, 2, 3]) == -1
    end
end
