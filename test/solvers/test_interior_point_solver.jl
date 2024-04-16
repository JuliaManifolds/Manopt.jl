using Manifolds, Manopt, LinearAlgebra, Random, Test

# <<<<<<< Updated upstream
# @testset "IP Newton Tests" begin
#     @test 1 == 1
#     a = 1 / 2
#     @test a ≈ 0.5 atol = 1e-9
# end
# =======
include("../../src/solvers/interior_point_Newton.jl")
# include("../../src/plans/interior_point_Newton_plan.jl")


A = Symmetric(rand(3,3))

function f(M, p)
    return 0.5 * p' * A * p
end

function grad_f(M, p)
    return ( A - f(M, p)*I ) * p
end

function g(M, p)
    return p - ones(3)
end

function grad_g(M, p)
    return [1 0 0;
            0 1 0;
            0 0 1]
end

m = 3
n = 0
M = Sphere(2)
N = M × ℝ^m × ℝ^n × ℝ^m
q = rand(N)
X = rand(N, vector_at = q)
co = ConstrainedManifoldObjective(f, grad_f, g, grad_g)
problem = InteriorPointProblem(M, co, (m,n))
params = InteriorPointParams(problem)
L = InteriorPointLagrangian(problem, params)
state = InteriorPointState(L)

# @testset "IP Newton Tests" begin
#     @test 1==1
#     a = 1/2
#     @test a ≈ 0.5 atol=1e-9
# end
