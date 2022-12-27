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
    f(y) = 1 / 2 * sum([distance(M, y, x)^2 for x in pts])
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
    sgrad_f2! = [function f!(M, X, y)
        log!(M, X, y, x)
        X .*= -1
        return X
    end for x in pts]

    @testset "Constructors" begin
        msgo1 = ManifoldStochasticGradientObjective(sgrad_f1)
        dmp1 = DefaultManoptProblem(M, msgo1)
        msgo1i = ManifoldStochasticGradientObjective(
            sgrad_f1!; evaluation=InplaceEvaluation()
        )
        dmp1i = DefaultManoptProblem(M, msgo1i)
        msgo2 = ManifoldStochasticGradientObjective(sgrad_f2)
        dmp2 = DefaultManoptProblem(M, msgo2)
        msgo2i = ManifoldStochasticGradientObjective(
            sgrad_f2!; evaluation=InplaceEvaluation()
        )
        dmp2i = DefaultManoptProblem(M, msgo2i)
        @test get_gradient(dmp1, p, 1) == zeros(3)
        @test get_gradient(dmp2, p, 1) == zeros(3)
        for pr in [dmp1, dmp2, dmp2i]
            X = zero_vector(M, p)
            get_gradient!(pr, X, p, 1)
            @test X == get_gradient(pr, p, 1)
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
        @test_throws ErrorException get_gradients(dmp1i, p)
        @test_throws ErrorException get_gradient!(dmp1i, Z4, p, 1)
        sgds = StochasticGradientDescentState(
            M, deepcopy(p), zero_vector(M, p); direction=StochasticGradient(deepcopy(p))
        )
        sgds.order = collect(1:5)
        sgds.order_type = :Linear
        initialize_solver!(dmp1, sgds)
        sgds.order_type = :Linear
        step_solver!(dmp1, sgds, 1)
        @test sgds.p == exp(M, p, get_gradient(dmp1, p, 1))
    end
    @testset "Comparing Stochastic Methods" begin
        x1 = stochastic_gradient_descent(M, sgrad_f1, p; order_type=:Linear)
        @test norm(x1) ≈ 1
        o1 = stochastic_gradient_descent(
            M, sgrad_f1, p; order_type=:Linear, return_state=true
        )
        x1a = get_solver_result(o1)
        @test x1 == x1a
        x2 = stochastic_gradient_descent(M, sgrad_f1, p; order_type=:FixedRandom)
        @test norm(x2) ≈ 1
        x3 = stochastic_gradient_descent(M, sgrad_f1, p; order_type=:Random)
        @test norm(x3) ≈ 1
        x4 = stochastic_gradient_descent(
            M,
            sgrad_f1,
            p;
            order_type=:Random,
            direction=AverageGradient(
                M, p; n=10, direction=StochasticGradient(zero_vector(M, p))
            ),
        )
        @test norm(x4) ≈ 1
        x5 = stochastic_gradient_descent(
            M,
            sgrad_f1,
            p;
            order_type=:Random,
            direction=MomentumGradient(
                M, p; direction=StochasticGradient(zero_vector(M, p))
            ),
        )
        @test norm(x5) ≈ 1
    end
end
