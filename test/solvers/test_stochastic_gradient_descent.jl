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
    gradF(M, y) = sum([-log(M, y, x) for x in pts])
    sgradF1(M, y) = [-log(M, y, x) for x in pts]
    function sgradF1!(M, X, y)
        for i in 1:length(pts)
            log!(M, X[i], y, pts[i])
            X[i] .*= -1
        end
        return X
    end
    sgradF2 = [((M, y) -> -log(M, y, x)) for x in pts]
    sgradF2! = [function f!(M, X, y)
        log!(M, X, y, x)
        X .*= -1
        return X
    end for x in pts]

    @testset "Constructors" begin
        p1 = StochasticGradientProblem(M, sgradF1)
        p1e = StochasticGradientProblem(M, sgradF1!; evaluation=InplaceEvaluation())
        p2 = StochasticGradientProblem(M, sgradF2)
        p2m = StochasticGradientProblem(M, sgradF2!; evaluation=InplaceEvaluation())
        @test get_gradient(p1, 1, p) == zeros(3)
        @test get_gradient(p2, 1, p) == zeros(3)
        for pr in [p1, p2, p2m]
            X = zero_vector(M, p)
            get_gradient!(pr, X, 1, p)
            @test X == get_gradient(pr, 1, p)
        end
        @test get_gradients(p1, p) == get_gradients(p2, p)
        @test get_gradients(p2m, p) == get_gradients(p2, p)
        Z = get_gradients(p2m, p)
        Z2 = similar.(Z)
        get_gradients!(p2m, Z2, p)
        @test Z == Z2
        Z3 = similar.(Z)
        get_gradients!(p1, Z3, p)
        @test Z == Z3
        Z4 = similar.(Z)
        get_gradients!(p2, Z4, p)
        @test Z == Z4
        Z5 = similar.(Z)
        get_gradients!(p1e, Z5, p)
        @test Z == Z5
        @test_throws ErrorException get_gradients(p1e, p)
        @test_throws ErrorException get_gradient!(p1e, Z4, 1, p)
        o = StochasticGradientDescentState(
            M, deepcopy(p), zero_vector(M, p); direction=StochasticGradient(deepcopy(p))
        )
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
        m1 = MomentumGradient(M, p, StochasticGradient(zero_vector(M, p)))
        m2 = MomentumGradient(p1, p)
        @test typeof(m1.direction) == typeof(m2.direction) #both use StochasticGradient
        m3 = MomentumGradient(M, p)
        m4 = MomentumGradient(p2, p)
        @test m3.direction == m4.direction #both use Gradient
        a1 = AverageGradient(M, p, 10, StochasticGradient(zero_vector(M, p)))
        a2 = AverageGradient(p1, p, 10)
        @test typeof(a1.direction) == typeof(a2.direction) #both use StochasticGradient
        a3 = AverageGradient(M, p, 10)
        a4 = AverageGradient(p2, p, 10)
        @test a3.direction == a4.direction #both use Gradient
    end
    @testset "Comparing Stochastic Methods" begin
        x1 = stochastic_gradient_descent(M, sgradF1, p; order_type=:Linear)
        @test norm(x1) ≈ 1
        o1 = stochastic_gradient_descent(
            M, sgradF1, p; order_type=:Linear, return_state=true
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
            direction=AverageGradient(M, p, 10, StochasticGradient(zero_vector(M, p))),
        )
        @test norm(x4) ≈ 1
        x5 = stochastic_gradient_descent(
            M,
            sgradF1,
            p;
            order_type=:Random,
            direction=MomentumGradient(M, p, StochasticGradient(zero_vector(M, p))),
        )
        @test norm(x5) ≈ 1
    end
end
