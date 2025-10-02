using Manopt, Manifolds, Test
@testset "Stochastic Gradient Descent" begin
    M = Sphere(2)
    # 5 point mean
    p = [0.0, 0.0, 1.0]
    s = 1.0
    pts = [
        exp(M, p, X) for
            X in [zeros(3), [s, 0.0, 0.0], [-s, 0.0, 0.0], [0.0, s, 0.0], [0.0, -s, 0.0]]
    ]
    f(M, y) = 1 / 2 * sum([distance(M, y, x)^2 for x in pts])
    grad_f(M, y) = sum([-log(M, y, x) for x in pts])
    sgrad_f1(M, y) = [-log(M, y, x) for x in pts]
    function sgrad_f1!(M, X, y)
        for i in 1:length(pts)
            log!(M, X[i], y, pts[i])
            X[i] .*= -1
        end
        return X
    end
    sgrad_f2 = [((M, y) -> -log(M, y, x)) for x in pts]
    sgrad_f2! = [
        function f!(M, X, y)
                log!(M, X, y, x)
                X .*= -1
                return X
        end for x in pts
    ]

    @testset "Constructors" begin
        sg = StochasticGradient(M; p = p)()
        @test sg.X == zero_vector(M, p)

        msgo1 = ManifoldStochasticGradientObjective(sgrad_f1)
        dmp1 = DefaultManoptProblem(M, msgo1)
        msgo1i = ManifoldStochasticGradientObjective(
            sgrad_f1!; evaluation = InplaceEvaluation()
        )
        dmp1i = DefaultManoptProblem(M, msgo1i)
        msgo2 = ManifoldStochasticGradientObjective(sgrad_f2)
        dmp2 = DefaultManoptProblem(M, msgo2)
        msgo2i = ManifoldStochasticGradientObjective(
            sgrad_f2!; evaluation = InplaceEvaluation()
        )
        dmp2i = DefaultManoptProblem(M, msgo2i)
        @test get_gradient(dmp1, p, 1) == zeros(3)
        @test get_gradient(dmp1, p) == zeros(3)
        @test get_gradient(dmp2, p, 1) == zeros(3)
        @test get_gradient(dmp2, p) == zeros(3)
        X = zero_vector(M, p)
        get_gradient!(dmp1, X, p)
        @test X == zeros(3)
        get_gradient!(dmp2, X, p)
        @test X == zeros(3)
        for pr in [dmp1, dmp2, dmp2i]
            X = zero_vector(M, p)
            get_gradient!(pr, X, p, 1)
            @test X == get_gradient(pr, p, 1)
            get_gradient!(pr, X, p)
            @test X == get_gradient(pr, p)
        end
        @test get_gradients(dmp1, p) == get_gradients(dmp2, p)
        @test get_gradients(dmp2i, p) == get_gradients(dmp2, p)
        Z = get_gradients(dmp2i, p)
        Z2 = similar.(Z)
        get_gradients!(dmp2i, Z2, p)
        @test Z == Z2
        Z3 = similar.(Z)
        get_gradients!(dmp1, Z3, p)
        @test Z == Z3
        Z4 = similar.(Z)
        get_gradients!(dmp2, Z4, p)
        @test Z == Z4
        Z5 = similar.(Z)
        get_gradients!(dmp1i, Z5, p)
        @test Z == Z5
        X = zero_vector(M, p)
        @test_throws ErrorException get_gradient!(dmp1i, X, p)
        @test_throws ErrorException get_gradients(dmp1i, p)
        @test_throws ErrorException get_gradient!(dmp1i, Z4, p, 1)
        sgds = StochasticGradientDescentState(
            M; p = deepcopy(p), X = zero_vector(M, p), direction = StochasticGradient(; p = p)(M)
        )
        sgds.order = collect(1:5)
        sgds.order_type = :Linear
        initialize_solver!(dmp1, sgds)
        sgds.order_type = :Linear
        step_solver!(dmp1, sgds, 1)
        @test sgds.p == exp(M, p, get_gradient(dmp1, p, 1))
        @test startswith(
            repr(sgds), "# Solver state for `Manopt.jl`s Stochastic Gradient Descent\n"
        )
    end
    @testset "Comparing Stochastic Methods" begin
        q1 = stochastic_gradient_descent(M, sgrad_f1, p; order_type = :Linear)
        @test is_point(M, q1, true)
        s1 = stochastic_gradient_descent(
            M, sgrad_f1, p; order_type = :Linear, return_state = true
        )
        q1a = get_solver_result(s1)
        @test q1 == q1a
        q2 = stochastic_gradient_descent(M, sgrad_f1, p; order_type = :FixedRandom)
        @test is_point(M, q2, true)
        q3 = stochastic_gradient_descent(M, sgrad_f1, p; order_type = :Random)
        @test is_point(M, q3, true)
        q4 = copy(M, p)
        stochastic_gradient_descent!(M, sgrad_f1, q4; order_type = :Random)
        @test is_point(M, q4, true)
        q5 = stochastic_gradient_descent(
            M,
            sgrad_f1,
            p;
            order_type = :Random,
            direction = AverageGradient(M; p = p, n = 10, direction = StochasticGradient()),
        )
        @test is_point(M, q5, true)
        q6 = stochastic_gradient_descent(
            M,
            sgrad_f1,
            p;
            order_type = :Random,
            direction = MomentumGradient(; p = p, direction = StochasticGradient()),
        )
        @test is_point(M, q6, true)
    end
    @testset "Comparing different starts" begin
        msgo2 = ManifoldStochasticGradientObjective(sgrad_f1)
        q1 = stochastic_gradient_descent(M, msgo2, p)
        q2 = copy(M, p)
        stochastic_gradient_descent!(M, msgo2, q2)
    end
    @testset "Circle example" begin
        Mc = Circle()
        pc = 0.0
        data = [-π / 4, 0.0, π / 4]
        fc(y) = 1 / 2 * sum([distance(M, y, x)^2 for x in data])
        sgrad_fc(M, y) = [-log(M, y, x) for x in data]
        q1 = stochastic_gradient_descent(Mc, sgrad_fc)
        q2 = stochastic_gradient_descent(Mc, sgrad_fc, pc)
        #For this case in-place does not exist.
        sgrad_fc2 = [((M, y) -> -log(M, y, x)) for x in data]
        q3 = stochastic_gradient_descent(Mc, sgrad_fc2, pc)
        q4 = stochastic_gradient_descent(Mc, sgrad_fc2, pc; evaluation = InplaceEvaluation())
        s = stochastic_gradient_descent(Mc, sgrad_fc2, pc; return_state = true)
        q5 = get_solver_result(s)[]
        @test all([is_point(Mc, q, true) for q in [q1, q2, q3, q4, q5]])
    end
end
