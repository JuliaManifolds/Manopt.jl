using Manopt, Manifolds, Test
@testset "Alternating Gradient Descent" begin
    # Note that this is mereely an alternating gradient descent toy example
    M = Sphere(2)
    N = M × M
    data = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    function F(N, x)
        return 1 / 2 *
               (distance(N[1], x[N, 1], data[1])^2 + distance(N[2], x[N, 2], data[2])^2)
    end
    gradF1(N, x) = -log(N[1], x[N, 1], data[1])
    gradF1!(N, Y, x) = (Y .= -log(N[1], x[N, 1], data[1]))
    gradF2(N, x) = -log(N[2], x[N, 2], data[2])
    gradF2!(N, Y, x) = (Y .= -log(N[2], x[N, 2], data[2]))
    gradF(N, x) = ProductRepr([-log(N[i], x[N, i], data[i]) for i in [1, 2]]...)
    function gradF!(N, Y, x)
        log!(N[1], Y[N, 1], x[N, 1], data[1])
        log!(N[2], Y[N, 2], x[N, 2], data[2])
        return Y .*= -1
    end
    x = ProductRepr([0.0, 0.0, 1.0], [0.0, 0.0, 1.0])

    @testset "Test gradient access" begin
        Pf = AlternatingGradientProblem(N, F, gradF)
        Pv = AlternatingGradientProblem(N, F, [gradF1, gradF2])
        Pf! = AlternatingGradientProblem(N, F, gradF!; evaluation=MutatingEvaluation())
        Pv! = AlternatingGradientProblem(
            N, F, [gradF1!, gradF2!]; evaluation=MutatingEvaluation()
        )
        for P in [Pf, Pv, Pf!, Pv!]
            Y = zero_tangent_vector(N, x)
            @test get_gradient(P, x)[N, 1] == gradF(N, x)[N, 1]
            @test get_gradient(P, x)[N, 2] == gradF(N, x)[N, 2]
            get_gradient!(P, Y, x)
            @test Y[N, 1] == gradF(N, x)[N, 1]
            @test Y[N, 2] == gradF(N, x)[N, 2]
            @test get_gradient(P, 1, x) == gradF(N, x)[N, 1]
            @test get_gradient(P, 2, x) == gradF(N, x)[N, 2]
            Y = zero_tangent_vector(N, x)
            get_gradient!(P, Y[N, 1], 1, x)
            @test Y[N, 1] == gradF(N, x)[N, 1]
            get_gradient!(P, Y[N, 2], 2, x)
            @test Y[N, 2] == gradF(N, x)[N, 2]
        end
    end
    @testset "Test high level interface" begin
        y2 = allocate(x)
        copyto!(N, y2, x)
        y3 = allocate(x)
        copyto!(N, y3, x)
        y = alternating_gradient_descent(
            N, F, [gradF1!, gradF2!], x; order_type=:Linear, evaluation=MutatingEvaluation()
        )
        alternating_gradient_descent!(
            N,
            F,
            [gradF1!, gradF2!],
            y2;
            order_type=:Linear,
            evaluation=MutatingEvaluation(),
        )
        @test isapprox(M, y[N, 1], data[1]; atol=10^-3)
        @test isapprox(M, y[N, 2], data[2]; atol=10^-3)
        @test isapprox(N, y, y2)
        o = alternating_gradient_descent!(
            N,
            F,
            [gradF1!, gradF2!],
            y3;
            evaluation=MutatingEvaluation(),
            order_type=:Linear,
            return_options=true,
        )
        @test isapprox(N, y, o.x; atol=10^-3)
    end
end
