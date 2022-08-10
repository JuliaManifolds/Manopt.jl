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
    @testset "Basics and access functions" begin end
    @testset "Two small Test runs" begin
        @testset "Testing with an Oracle" begin
            p2a = Frank_Wolfe_method(
                M, f, grad_f!, p; subtask=oracle!, evaluation=MutatingEvaluation()
            )
            @test f(M, p2a) < f(M, p)
            p2b = Frank_Wolfe_method(M, f, grad_f, p; subtask=oracle)
            @test f(M, p2b) ≈ f(M, p2a)
            o2 = Frank_Wolfe_method(M, f, grad_f, p; subtask=oracle, return_options=true)
            p2c = get_solver_result(o2)
            @test f(M, p2c) ≈ f(M, p2a)
        end
        @testset "Testing with an Subsolver" begin
            # This is not a useful run since the subproblem is not constraint
            p3 = Frank_Wolfe_method(
                M,
                f,
                grad_f!,
                p;
                evaluation=MutatingEvaluation(),
                stopping_criterion=StopAfterIteration(1),
            )
            #so we can just test that the subproblem is delivering a point.
            @test is_point(M, p3)
        end
    end
end
