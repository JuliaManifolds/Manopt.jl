using ManifoldsBase, Manopt, Test, LinearAlgebra

@testset "Frank Wolfe Method" begin
    M = ManifoldsBase.DefaultManifold(3)
    A = [1 2 1; 0 2 1; 0 1 1; 1 1 0]
    b = [1, 2, 1, 1]
    #
    #
    #
    f(M, p) = norm(A * p - b)^2
    grad_f!(M, X, p) = (X .= transpose(A) * (A * p - b))
    grad_f(M, p) = transpose(A) * (A * p - b)
    function oracle!(M, q, p, X)
        i = argmax(X)
        q .= p
        return q[i] = p[i] - sign(X[i])
    end
    function oracle(M, p, X)
        X
        i = argmax(X)
        q = copy(p)
        q[i] = p[i] - sign(X[i])
        return q
    end
    p = ones(3)
    @testset "Basics and access functions" begin
        p = ones(3)
        X = ones(3)
        FC = FrankWolfeCost(p, X)
        @test FC(M, p) == 0
        FG = FrankWolfeGradient(p, X)
        Y = similar(X)
        FG(M, Y, p)
        @test FG(M, p) == Y
        s = FrankWolfeState(M, p, oracle!, InplaceEvaluation())
        @test Manopt.get_message(s) == ""
        @test startswith(repr(s), "# Solver state for `Manopt.jl`s Frank Wolfe Method\n")
        set_iterate!(s, 2 .* p)
        @test get_iterate(s) == 2 .* p
        dmp = DefaultManoptProblem(M, ManifoldGradientObjective(FC, FG))
        gds = GradientDescentState(M)
        s2 = FrankWolfeState(M, p, dmp, gds)
        @test Manopt.get_message(s2) == ""
    end
    @testset "Two small Test runs" begin
        @testset "Testing with an Oracle" begin
            p2a = Frank_Wolfe_method(
                M, f, grad_f!, p; sub_problem=oracle!, evaluation=InplaceEvaluation()
            )
            @test f(M, p2a) < f(M, p)
            p2b = Frank_Wolfe_method(M, f, grad_f, p; sub_problem=oracle)
            @test f(M, p2b) ≈ f(M, p2a)
            p2c = copy(M, p)
            Frank_Wolfe_method!(M, f, grad_f, p2c; sub_problem=oracle)
            @test f(M, p2c) < f(M, p)
            p2d = Frank_Wolfe_method(M, f, grad_f; sub_problem=oracle)
            @test f(M, p2d) < f(M, p)
        end
        @testset "Testing with an Subsolver" begin
            # This is not a useful run since the subproblem is not constraint
            p3 = Frank_Wolfe_method(
                M,
                f,
                grad_f!,
                p;
                evaluation=InplaceEvaluation(),
                stopping_criterion=StopAfterIteration(1),
            )
            @test is_point(M, p3)
            p3b = Frank_Wolfe_method(
                M,
                f,
                grad_f,
                p;
                evaluation=AllocatingEvaluation(),
                stopping_criterion=StopAfterIteration(1),
            )
            #so we can just test that the subproblem is delivering a point.
            @test is_point(M, p3b)
        end
        @testset "Number test" begin
            # I have no good idea for a test, so this merely
            # Checks the call, since that it works was already tested
            M = Euclidean()
            f(M, p) = P
            grad_f(M, p) = zero_vector(M, p)
            oracle(M, p, X) = X
            Frank_Wolfe_method(M, f, grad_f; sub_problem=oracle)
            # and since the gradient is zero and oracle hence returns zero, we stay at zero
            @test 0.0 == Frank_Wolfe_method(M, f, grad_f, 0.0; sub_problem=oracle)
        end
    end
end
