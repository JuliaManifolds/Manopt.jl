using Manifolds, Manopt, Test

include("../utils/dummy_types.jl")

@testset "Counting Objective test" begin
    M = Sphere(2)
    A = [2.0 1.0 0.0; 1.0 2.0 1.0; 0.0 1.0 2.0]
    f(M, p) = p' * A * p
    grad_f(M, p) = project(M, p, 2 * A * p)
    obj = ManifoldGradientObjective(f, grad_f)
    c_obj = ManifoldCountObjective(M, obj, [:Cost, :Gradient])
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
    # also decorated objects can be wrapped to be counted
    ro = DummyDecoratedObjective(obj)
    c_obj2 = ManifoldCountObjective(M, ro, [:Gradient])
    get_gradient(M, c_obj2, p)
    @test get_count(c_obj2, :Gradient) == 1
    @test_throws ErrorException get_count(ro, :Cost) # Errors since no CountObj
    @test get_count(c_obj2, :Cost) == -1 # Does not count cost
    @test_throws ErrorException get_count(c_obj2, :Cost, :error)
    @test startswith(repr(c_obj), "## Statistics")
    @test startswith(Manopt.status_summary(c_obj), "## Statistics")
    # also if we get any (nonspecific tuple) - i.e. if the second is a point
    @test startswith(repr((c_obj, p)), "## Statistics")
    # but this also includes the hint, how to access the result
    @test endswith(repr((c_obj, p)), "on this variable.")
end
