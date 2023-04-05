using Manopt, Manifolds, Test

@testset "Stepsize" begin
    @test Manopt.get_message(ConstantStepsize(1.0)) == ""
    s = ArmijoLinesearch()
    @test startswith(repr(s), "ArmijoLineseach() with keyword parameters\n")
    s_stat = Manopt.status_summary(s)
    @test startswith(s_stat, "ArmijoLineseach() with keyword parameters\n")
    @test endswith(s_stat, "of 1.0")
    @test Manopt.get_message(s) == ""

    s2 = NonmonotoneLinesearch()
    @test startswith(repr(s2), "NonmonotoneLinesearch() with keyword arguments\n")
    @test Manopt.get_message(s2) == ""

    s2b = NonmonotoneLinesearch(Euclidean(2)) # with manifold -> faster storage
    @test startswith(repr(s2b), "NonmonotoneLinesearch() with keyword arguments\n")

    s3 = WolfePowellBinaryLinesearch()
    @test Manopt.get_message(s3) == ""
    @test startswith(repr(s3), "WolfePowellBinaryLinesearch(DefaultManifold(), ")
    # no stepsize yet so repr and summary are the same
    @test repr(s3) == Manopt.status_summary(s3)
    s4 = WolfePowellLinesearch()
    @test startswith(repr(s4), "WolfePowellLinesearch(DefaultManifold(), ")
    # no stepsize yet so repr and summary are the same
    @test repr(s4) == Manopt.status_summary(s4)
    @test Manopt.get_message(s4) == ""
    @testset "Linesearch safeguards" begin
        M = Euclidean(2)
        f(M, p) = sum(p .^ 2)
        grad_f(M, p) = sum(2 .* p)
        p = [2.0, 2.0]
        s1 = Manopt.linesearch_backtrack(
            M, f, p, grad_f(M, p), 1.0, 1.0, 0.99; max_decrease_steps=10
        )
        @test startswith(s1[2], "Max decrease")
        s2 = Manopt.linesearch_backtrack(
            M, f, p, grad_f(M, p), 1.0, 1.0, 0.5, ExponentialRetraction(), grad_f(M, p);
        )
        @test startswith(s2[2], "The search direction")
        s3 = Manopt.linesearch_backtrack(
            M, f, p, grad_f(M, p), 1.0, 1.0, 0.5; stop_when_stepsize_less=0.75
        )
        @test startswith(s3[2], "Min step size (0.75)")
        # cheating for increase
        s4 = Manopt.linesearch_backtrack(
            M, f, p, grad_f(M, p), 1e-12, 0, 0.5; stop_when_stepsize_larger=0.1
        )
        @test startswith(s4[2], "Max step size (0.1)")
        s5 = Manopt.linesearch_backtrack(
            M, f, p, grad_f(M, p), 1e-12, 0, 0.5; max_increase_steps=1
        )
        @test startswith(s5[2], "Max increase steps (1)")
    end
end
