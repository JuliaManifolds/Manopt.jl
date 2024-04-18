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


A = [2 0 0; 0 2 0; 0 0 1]

function f(M, p)
    return 0.5*p'*A*p
end

function grad_f(M, p)
    return (I - p*p')*A*p
end

function Hess_f(M, p, X)
    return (I - p*p')*A*X - f(M, p)*X
end

function g(M, p)
    return -[p[3]]
end

function grad_g(M, p)
    return [0 0 -1] * (I - p*p')
end

m = 1
n = 0
M = Sphere(2)

co = ConstrainedManifoldObjective(f, grad_f, g, grad_g)
ho = ManifoldHessianObjective(f, grad_f, Hess_f)
problem = ConstrainedProblem(M, co, ho, (m,n))
state = InteriorPointState(problem)
p = interior_point_Newton(problem)



# @testset "IP Newton Tests" begin
#     @test 1==1
#     a = 1/2
#     @test a ≈ 0.5 atol=1e-9
# end
