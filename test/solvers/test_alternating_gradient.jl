using Manopt, Manifolds, Test, RecursiveArrayTools

@testset "Alternating Gradient Descent" begin
    # Note that this is merely an alternating gradient descent toy example
    M = Sphere(2)
    N = M × M
    data = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    function f(N, p)
        return 1 / 2 * (
            distance(N[1], p[N, Val(1)], data[1])^2 +
                distance(N[2], p[N, Val(2)], data[2])^2
        )
    end
    grad_f1(N, p) = -log(N[1], p[N, 1], data[1])
    grad_f1!(N, X, p) = (X .= -log(N[1], p[N, 1], data[1]))
    grad_f2(N, p) = -log(N[2], p[N, 2], data[2])
    grad_f2!(N, X, p) = (X .= -log(N[2], p[N, 2], data[2]))
    grad_f(N, p) = ArrayPartition([-log(N[i], p[N, i], data[i]) for i in [1, 2]]...)
    function grad_f!(N, X, p)
        log!(N[1], X[N, 1], p[N, 1], data[1])
        log!(N[2], X[N, 2], p[N, 2], data[2])
        return X .*= -1
    end
    p = ArrayPartition([0.0, 0.0, 1.0], [0.0, 0.0, 1.0])

    @testset "Test gradient access" begin
        objf = ManifoldAlternatingGradientObjective(f, grad_f)
        Pf = DefaultManoptProblem(N, objf)
        objv = ManifoldAlternatingGradientObjective(f, [grad_f1, grad_f2])
        Pv = DefaultManoptProblem(N, objv)
        objf! = ManifoldAlternatingGradientObjective(
            f, grad_f!; evaluation = InplaceEvaluation()
        )
        Pf! = DefaultManoptProblem(N, objf!)
        objv! = ManifoldAlternatingGradientObjective(
            f, [grad_f1!, grad_f2!]; evaluation = InplaceEvaluation()
        )
        Pv! = DefaultManoptProblem(N, objv!)
        for P in [Pf, Pv, Pf!, Pv!]
            X = zero_vector(N, p)
            @test get_gradient(P, p)[N, 1] == grad_f(N, p)[N, 1]
            @test get_gradient(P, p)[N, 2] == grad_f(N, p)[N, 2]
            get_gradient!(P, X, p)
            @test X[N, 1] == grad_f(N, p)[N, 1]
            @test X[N, 2] == grad_f(N, p)[N, 2]
            @test get_gradient(P, p, 1) == grad_f(N, p)[N, 1]
            @test get_gradient(P, p, 2) == grad_f(N, p)[N, 2]
            X = zero_vector(N, p)
            get_gradient!(P, X[N, 1], p, 1)
            @test X[N, 1] == grad_f(N, p)[N, 1]
            get_gradient!(P, X[N, 2], p, 2)
            @test X[N, 2] == grad_f(N, p)[N, 2]
        end
    end
    @testset "Test high level interface" begin
        q = allocate(p)
        copyto!(N, q, p)
        q2 = allocate(p)
        copyto!(N, q2, p)
        q3 = alternating_gradient_descent(
            N,
            f,
            [grad_f1!, grad_f2!],
            p;
            order_type = :Linear,
            evaluation = InplaceEvaluation(),
        )
        r = alternating_gradient_descent!(
            N,
            f,
            [grad_f1!, grad_f2!],
            q;
            order_type = :Linear,
            evaluation = InplaceEvaluation(),
            return_state = true,
        )
        @test startswith(
            repr(r), "# Solver state for `Manopt.jl`s Alternating Gradient Descent Solver"
        )
        # r has the same message as the internal stepsize
        @test Manopt.get_message(r) == Manopt.get_message(r.stepsize)
        @test isapprox(N, q3, q)
    end
end
