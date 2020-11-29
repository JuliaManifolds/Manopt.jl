using Manopt, Manifolds, Test
@testset "Stochastic Gradient Descent" begin
    M = Sphere(2)
    # 5 point mean
    p = [0.0, 0.0, 1.0]
    s = 1.0
    pts = [exp(M, p, X) for X ∈ [zeros(3), [s, 0.0, 0.0], [-s, 0.0, 0.0], [0.0, s, 0.0], [0.0, -s, 0.0]] ]
    F(y) = 1/2 * sum([distance(M, y, x)^2 for x ∈ pts])
    ∇F(y) = sum( [ -log(M, y, x) for x ∈ pts] )
    s∇F1(y) = [ -log(M, y, x) for x ∈ pts ]
    s∇F2 = [ ( y -> -log(M,y,x)) for x ∈ pts ]
    @testset "Constructors" begin
        p1 = StochasticGradientProblem(M, s∇F1)
        p2 = StochasticGradientProblem(M, s∇F2)
        @test get_gradient(p1, 1, p) == zeros(3)
        @test get_gradient(p2, 1, p) == zeros(3)
        @test get_gradients(p1, p) == get_gradients(p2, p)
        o = StochasticGradientDescentOptions(deepcopy(p))
        o.order = collect(1:5)
        o.order_type = :Linear
        initialize_solver!(p1,o)
        o.order_type = :Linear
        step_solver!(p1,o,1)
        @test o.x == exp(M, p, get_gradient(p1, 1, p))
    end
    @testset "Comparing Stochastic Methods" begin
        x1 = stochastic_gradient_descent(M, s∇F1, deepcopy(p); order_type=:Linear)
        @test norm(x1) ≈ 1
        x2 = stochastic_gradient_descent(M, s∇F1, deepcopy(p); order_type=:FixedRandom)
        @test norm(x2) ≈ 1
        x3 = stochastic_gradient_descent(M, s∇F1, deepcopy(p); order_type=:Random)
        @test norm(x3) ≈ 1
        x4 = stochastic_gradient_descent(M, s∇F1, deepcopy(p); order_type=:Random, direction = AverageGradient(M, p, 10, StochasticGradient()))
        @test norm(x4) ≈ 1
        x5 = stochastic_gradient_descent(M, s∇F1, deepcopy(p); order_type=:Random, direction = MomentumGradient(M, p, StochasticGradient()))
        @test norm(x5) ≈ 1
    end
end