using LinearAlgebra, Manopt, Manifolds, Test
include("../utils/dummy_types.jl")

@testset "Stochastic Gradient Plan" begin
    M = Sphere(2)
    # 5 point mean
    p = [0.0, 0.0, 1.0]
    s = 1.0
    pts = [
        exp(M, p, X) for
        X in [zeros(3), [s, 0.0, 0.0], [-s, 0.0, 0.0], [0.0, s, 0.0], [0.0, -s, 0.0]]
    ]
    f(M, y) = 1 / 2 * sum([distance(M, y, x)^2 for x in pts])
    sgrad_f1(M, y) = [-log(M, y, x) for x in pts]
    sgrad_f2 = [((M, y) -> -log(M, y, x)) for x in pts]
    msgo1 = ManifoldStochasticGradientObjective(sgrad_f1)
    msgo2 = ManifoldStochasticGradientObjective(sgrad_f2)
    @testset "Objetive Decorator passthrough" begin
        X = zero_vector(M, p)
        Y = zero_vector(M, p)
        Xa = [zero_vector(M, p) for p in pts]
        Ya = [zero_vector(M, p) for p in pts]
        for obj in [msgo1, msgo2]
            ddo = DummyDecoratedObjective(obj)
            @test get_gradients(M, obj, p) == get_gradients(M, ddo, p)
            get_gradients!(M, Xa, obj, p)
            get_gradients!(M, Ya, ddo, p)
            @test Xa == Ya
            for i in 1:length(sgrad_f2)
                @test get_gradient(M, obj, p, i) == get_gradient(M, ddo, p, i)
                get_gradient!(M, X, obj, p, i)
                get_gradient!(M, Y, ddo, p, i)
                @test X == Y
            end
        end
    end
end
