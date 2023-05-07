using Manifolds, Manopt, Test

@testset "Counting Objective test" begin
    M = Sphere(2)
    A = [2.0 1.0; 1.0 2.0]
    f(M, p) = p' * A * p
    grad_f(M, p) = project(M, p, 2 * A * p)
    obj = ManifoldGradientObjective(f, grad_f)
    c_obj = CountObjective(obj, [:Cost, :Gradient])
end
