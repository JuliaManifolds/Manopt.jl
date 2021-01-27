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
    F(y) = 1 / 2 * sum([distance(M, y, x)^2 for x in pts])
    gradF(y) = sum([-log(M, y, x) for x in pts])
    sgradF1(y) = [-log(M, y, x) for x in pts]
    sgradF2 = [(y -> -log(M, y, x)) for x in pts]
    @testset "Constructors" begin
        p1 = StochasticGradientProblem(M, sgradF1)
        p2 = StochasticGradientProblem(M, sgradF2)
        @test get_gradient(p1, 1, p) == zeros(3)
        @test get_gradient(p2, 1, p) == zeros(3)
        @test get_gradients(p1, p) == get_gradients(p2, p)
        o = StochasticGradientDescentOptions(deepcopy(p))
        o.order = collect(1:5)
        o.order_type = :Linear
        initialize_solver!(p1, o)
        o.order_type = :Linear
        step_solver!(p1, o, 1)
        @test o.x == exp(M, p, get_gradient(p1, 1, p))
    end
    @testset "Momentum and Average Processor Constructors" begin
        p1 = StochasticGradientProblem(M, sgradF1)
        p2 = GradientProblem(M, F, gradF)
        m1 = MomentumGradient(M, p, StochasticGradient())
        m2 = MomentumGradient(p1, p)
        @test m1.direction == m2.direction #both use StochasticGradient
        m3 = MomentumGradient(M, p)
        m4 = MomentumGradient(p2, p)
        @test m3.direction == m4.direction #both use Gradient
        a1 = AverageGradient(M, p, 10, StochasticGradient())
        a2 = AverageGradient(p1, p, 10)
        @test a1.direction == a2.direction #both use StochasticGradient
        a3 = AverageGradient(M, p, 10)
        a4 = AverageGradient(p2, p, 10)
        @test a3.direction == a4.direction #both use Gradient
    end
    @testset "Comparing Stochastic Methods" begin
        x1 = stochastic_gradient_descent(M, sgradF1, p; order_type=:Linear)
        @test norm(x1) ≈ 1
        o1 = stochastic_gradient_descent(
            M, sgradF1, p; order_type=:Linear, return_options=true
        )
        x1a = get_solver_result(o1)
        @test x1 == x1a
        x2 = stochastic_gradient_descent(M, sgradF1, p; order_type=:FixedRandom)
        @test norm(x2) ≈ 1
        x3 = stochastic_gradient_descent(M, sgradF1, p; order_type=:Random)
        @test norm(x3) ≈ 1
        x4 = stochastic_gradient_descent(
            M,
            sgradF1,
            p;
            order_type=:Random,
            direction=AverageGradient(M, p, 10, StochasticGradient()),
        )
        @test norm(x4) ≈ 1
        x5 = stochastic_gradient_descent(
            M,
            sgradF1,
            p;
            order_type=:Random,
            direction=MomentumGradient(M, p, StochasticGradient()),
        )
        @test norm(x5) ≈ 1
    end
end
