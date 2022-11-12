using Manopt, ManifoldsBase, Test

@testset "Constrained Plan" begin
    M = ManifoldsBase.DefaultManifold(3)
    # Cost
    F(M, p) = norm(M, p)^2
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
    c = [[0, -3], [5]]
    gg = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    gh = [[0.0, 0.0, 2.0]]
    gf = 2 * p
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
    @testset "Augmented Lagrangian Cost & Grad" begin end
end
