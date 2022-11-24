using Manopt, ManifoldsBase, Test

@testset "Constrained Plan" begin
    M = ManifoldsBase.DefaultManifold(3)
    # Cost
    F(::ManifoldsBase.DefaultManifold, p) = norm(p)^2
    gradF(M, p) = 2 * p
    gradF!(M, X, p) = (X .= 2 * p)
    # Inequality constraints
    G(M, p) = [p[1] - 1, -p[2] - 1]
    # # Function
    gradG(M, p) = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    function gradG!(M, X, p)
        X[1] .= [1.0, 0.0, 0.0]
        X[2] .= [0.0, -1.0, 0.0]
        return X
    end
    # # Vectorial
    G1(M, p) = p[1] - 1
    gradG1(M, p) = [1.0, 0.0, 0.0]
    gradG1!(M, X, p) = (X .= [1.0, 0.0, 0.0])
    G2(M, p) = -p[2] - 1
    gradG2(M, p) = [0.0, -1.0, 0.0]
    gradG2!(M, X, p) = (X .= [0.0, -1.0, 0.0])
    # Equality Constraints
    H(M, p) = [2 * p[3] - 1]
    H1(M, p) = 2 * p[3] - 1
    gradH(M, p) = [[0.0, 0.0, 2.0]]
    function gradH!(M, X, p)
        X[1] .= [0.0, 0.0, 2.0]
        return X
    end
    gradH1(M, p) = [0.0, 0.0, 2.0]
    gradH1!(M, X, p) = (X .= [0.0, 0.0, 2.0])
    Pfa = ConstrainedProblem(M, F, gradF, G, gradG, H, gradH)
    Pfm = ConstrainedProblem(
        M, F, gradF!, G, gradG!, H, gradH!; evaluation=MutatingEvaluation()
    )
    Pva = ConstrainedProblem(M, F, gradF, [G1, G2], [gradG1, gradG2], [H1], [gradH1])
    Pvm = ConstrainedProblem(
        M,
        F,
        gradF!,
        [G1, G2],
        [gradG1!, gradG2!],
        [H1],
        [gradH1!];
        evaluation=MutatingEvaluation(),
    )
    @test repr(Pfa) === "ConstrainedProblem{AllocatingEvaluation,FunctionConstraint}."
    @test repr(Pfm) === "ConstrainedProblem{MutatingEvaluation,FunctionConstraint}."
    @test repr(Pva) === "ConstrainedProblem{AllocatingEvaluation,VectorConstraint}."
    @test repr(Pvm) === "ConstrainedProblem{MutatingEvaluation,VectorConstraint}."

    p = [1.0, 2.0, 3.0]
    c = [[0.0, -3.0], [5.0]]
    gg = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    gh = [[0.0, 0.0, 2.0]]
    gf = 2 * p

    @testset "Partial Constructors" begin
        # At least one constraint necessary
        @test_throws ErrorException ConstrainedProblem(M, F, gradF)
        @test_throws ErrorException ConstrainedProblem(
            M, F, gradF!; evaluation=MutatingEvaluation()
        )
        p1f = ConstrainedProblem(M, F, gradF!; G=G, gradG=gradG)
        @test get_constraints(p1f, p) == [c[1], []]
        @test get_grad_equality_constraints(p1f, p) == []
        @test get_grad_inequality_constraints(p1f, p) == gg

        p1v = ConstrainedProblem(M, F, gradF!; G=[G1, G2], gradG=[gradG1, gradG2])
        @test get_constraints(p1v, p) == [c[1], []]
        @test get_grad_equality_constraints(p1v, p) == []
        @test get_grad_inequality_constraints(p1v, p) == gg

        p2f = ConstrainedProblem(M, F, gradF!; H=H, gradH=gradH)
        @test get_constraints(p2f, p) == [[], c[2]]
        @test get_grad_equality_constraints(p2f, p) == gh
        @test get_grad_inequality_constraints(p2f, p) == []

        p2v = ConstrainedProblem(M, F, gradF!; H=[H1], gradH=[gradH1])
        @test get_constraints(p2v, p) == [[], c[2]]
        @test get_grad_equality_constraints(p2v, p) == gh
        @test get_grad_inequality_constraints(p2v, p) == []
    end

    for P in [Pfa, Pfm, Pva, Pvm]
        @testset "$P" begin
            @test get_constraints(P, p) == c
            @test get_equality_constraints(P, p) == c[2]
            @test get_equality_constraint(P, p, 1) == c[2][1]
            @test get_inequality_constraints(P, p) == c[1]
            @test get_inequality_constraint(P, p, 1) == c[1][1]
            @test get_inequality_constraint(P, p, 2) == c[1][2]

            @test get_grad_equality_constraints(P, p) == gh
            Xh = [zeros(3)]
            @test get_grad_equality_constraints!(P, Xh, p) == gh
            @test Xh == gh
            X = zeros(3)
            @test get_grad_equality_constraint(P, p, 1) == gh[1]
            @test get_grad_equality_constraint!(P, X, p, 1) == gh[1]
            @test X == gh[1]

            @test get_grad_inequality_constraints(P, p) == gg
            Xg = [zeros(3), zeros(3)]
            @test get_grad_inequality_constraints!(P, Xg, p) == gg
            @test Xg == gg
            @test get_grad_inequality_constraint(P, p, 1) == gg[1]
            @test get_grad_inequality_constraint!(P, X, p, 1) == gg[1]
            @test X == gg[1]
            @test get_grad_inequality_constraint(P, p, 2) == gg[2]
            @test get_grad_inequality_constraint!(P, X, p, 2) == gg[2]
            @test X == gg[2]

            @test get_gradient(P, p) == gf
            @test get_gradient!(P, X, p) == gf
            @test X == gf
        end
    end
    @testset "Augmented Lagrangian Cost & Grad" begin
        μ = [1.0, 1.0]
        λ = [1.0]
        ρ = 0.1
        cg = sum(max.([0.0, 0.0], c[1] .+ μ ./ ρ) .^ 2)
        ch = sum((c[2] .+ λ ./ ρ) .^ 2)
        ac = F(M, p) + ρ / 2 * (cg + ch)
        agg = sum((c[1] .* ρ .+ μ) .* gg .* (c[1] .+ μ ./ ρ .> 0))
        agh = sum((c[2] .* ρ .+ λ) .* gh)
        ag = gf + agg + agh
        X = zero_vector(M, p)
        for P in [Pfa, Pfm, Pva, Pvm]
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
        for P in [Pfa, Pfm, Pva, Pvm]
            @testset "$P" begin
                EPCe = ExactPenaltyCost(P, ρ, u; smoothing=LogarithmicSumOfExponentials())
                EPGe = ExactPenaltyGrad(P, ρ, u; smoothing=LogarithmicSumOfExponentials())
                # LogExp Cost
                v1 = sum(u .* log.(1 .+ exp.(c[1] ./ u))) # cost g
                v2 = sum(u .* log.(exp.(c[2] ./ u) .+ exp.(-c[2] ./ u))) # cost h
                @test EPCe(M, p) ≈ F(M, p) + ρ * (v1 + v2)
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
                @test EPCh(M, p) ≈ F(M, p) + ρ * (w1 + w2 + w3)
                wg1 = sum(gg .* (c[1] .>= u) .* ρ)
                wg2 = sum(gg .* (c[1] ./ u .* (0 .<= c[1] .< u)) .* ρ)
                wg3 = sum(gh .* (c[2] ./ sqrt.(c[2] .^ 2 .+ u^2)) .* ρ)
                @test EPGh(M, p) ≈ gf + wg1 .+ wg2 .+ wg3
            end
        end
    end
end
