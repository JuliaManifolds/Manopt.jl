using Manifolds, Manopt, Test

@testset "Counting Objective test" begin
    M = Sphere(2)
    A = [2.0 1.0 0.0; 1.0 2.0 1.0; 0.0 1.0 2.0]
    f(M, p) = p' * A * p
    grad_f(M, p) = project(M, p, 2 * A * p)
    obj = ManifoldGradientObjective(f, grad_f)
    c_obj = CountObjective(obj, [:Cost, :Gradient])

    p = [1.0, 0.0, 0.0]
    X = [1.0, 1.0, 0.0]
    get_cost(M, c_obj, p)
    @test get_count(c_obj, :Cost) == 1
    @test get_count(c_obj, :NonExistent) == -1
    Y = similar(X)
    get_gradient(M, c_obj, p)
    get_gradient!(M, Y, c_obj, p)
    # both are counted
    @test get_count(c_obj, :Gradient) == 2
    get_gradient(M, obj, p)
    # others do not affect the counter
    @test get_count(c_obj, :Gradient) == 2
end
