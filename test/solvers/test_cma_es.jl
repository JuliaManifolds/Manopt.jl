using Revise
using Manopt, ManifoldsBase, Manifolds, Test
using Random

"""
    griewank(::AbstractManifold, p)

Compute value of the [Griewank test function](https://en.wikipedia.org/wiki/Griewank_function).
"""
function griewank(::AbstractManifold, p)
    sumsq = 0.0
    prodcos = 1.0
    for (i, xi) in enumerate(p)
        sumsq += xi^2
        prodcos *= cos(xi / sqrt(i))
    end
    return 1 + sumsq / 4000 - prodcos
end

@testset "CMA-ES" begin
    @testset "Euclidean CMA-ES" begin
        M = Euclidean(2)

        p1 = cma_es(M, griewank, [10.0, 10.0]; σ=10.0)

        o = cma_es(M, griewank, [10.0, 10.0]; return_state=true)
        @test startswith(
            repr(o),
            "# Solver state for `Manopt.jl`s Covariance Matrix Adaptation Evolutionary Strategy",
        )
        g = get_solver_result(o)
    end
    @testset "Spherical CMA-ES" begin
        Random.seed!(42)
        M = Sphere(2)
    end
end
