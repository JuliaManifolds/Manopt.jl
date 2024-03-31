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

function divergent_example(::AbstractManifold, p)
    return -norm(p)
end

function poorly_conditioned_example(::AbstractManifold, p)
    return p' * [1e12 0; 0 -1e-6] * p
end

@testset "CMA-ES" begin
    @testset "Euclidean CMA-ES" begin
        M = Euclidean(2)

        p1 = cma_es(M, griewank, [10.0, 10.0]; σ=10.0, rng=MersenneTwister(123))
        @test griewank(M, p1) < 0.1

        o = cma_es(M, griewank, [10.0, 10.0]; return_state=true)
        @test startswith(
            repr(o),
            "# Solver state for `Manopt.jl`s Covariance Matrix Adaptation Evolutionary Strategy",
        )

        o_d = cma_es(
            M,
            divergent_example,
            [10.0, 10.0];
            σ=10.0,
            rng=MersenneTwister(123),
            return_state=true,
        )
        @test only(get_active_stopping_criteria(o_d.stop)) isa Manopt.TolXUpCondition

        o_d = cma_es(
            M,
            poorly_conditioned_example,
            [10.0, 10.0];
            σ=10.0,
            rng=MersenneTwister(123),
            return_state=true,
        )
        condcov_sc = only(get_active_stopping_criteria(o_d.stop))
        @test condcov_sc isa Manopt.CMAESConditionCov
        @test !Manopt.indicates_convergence(condcov_sc)
        @test startswith(repr(condcov_sc), "CMAESConditionCov(")
    end
    @testset "Spherical CMA-ES" begin
        M = Sphere(2)

        p1 = cma_es(M, griewank, [0.0, 1.0, 0.0]; σ=1.0, rng=MersenneTwister(123))
        @test griewank(M, p1) < 0.17
    end
end
