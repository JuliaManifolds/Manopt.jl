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
    return p' * [1.0e12 0; 0 -1.0e-6] * p
end

flat_example(::AbstractManifold, p) = 0.0

@testset "CMA-ES" begin
    @testset "Euclidean CMA-ES" begin
        M = Euclidean(2)

        p1 = cma_es(M, griewank, [10.0, 10.0]; σ = 10.0, rng = MersenneTwister(123))
        @test griewank(M, p1) < 0.1

        p1 = cma_es(M, griewank; σ = 10.0, rng = MersenneTwister(123))
        @test griewank(M, p1) < 0.2

        p1 = [10.0, 10.0]
        cma_es!(M, griewank, p1; σ = 10.0, rng = MersenneTwister(123))
        @test griewank(M, p1) < 0.2

        o = cma_es(M, griewank, [10.0, 10.0]; return_state = true)
        @test startswith(
            repr(o),
            "# Solver state for `Manopt.jl`s Covariance Matrix Adaptation Evolutionary Strategy",
        )

        o_d = cma_es(
            M,
            divergent_example,
            [10.0, 10.0];
            σ = 10.0,
            rng = MersenneTwister(123),
            return_state = true,
        )
        div_sc = only(get_active_stopping_criteria(o_d.stop))
        @test div_sc isa StopWhenPopulationDiverges
        @test !Manopt.indicates_convergence(div_sc)
        @test startswith(repr(div_sc), "StopWhenPopulationDiverges(")

        o_d = cma_es(
            M,
            poorly_conditioned_example,
            [10.0, 10.0];
            σ = 10.0,
            rng = MersenneTwister(123),
            return_state = true,
        )
        condcov_sc = only(get_active_stopping_criteria(o_d.stop))
        @test condcov_sc isa StopWhenCovarianceIllConditioned
        @test !Manopt.indicates_convergence(condcov_sc)
        @test startswith(repr(condcov_sc), "StopWhenCovarianceIllConditioned(")

        o_flat = cma_es(
            M,
            flat_example,
            [10.0, 10.0];
            σ = 10.0,
            stopping_criterion = StopAfterIteration(500) |
                StopWhenBestCostInGenerationConstant{Float64}(5),
            rng = MersenneTwister(123),
            return_state = true,
        )
        flat_sc = only(get_active_stopping_criteria(o_flat.stop))
        @test flat_sc isa StopWhenBestCostInGenerationConstant
        @test Manopt.indicates_convergence(flat_sc)
        @test startswith(repr(flat_sc), "StopWhenBestCostInGenerationConstant(")

        o_flat = cma_es(
            M,
            flat_example,
            [10.0, 10.0];
            σ = 10.0,
            stopping_criterion = StopAfterIteration(500) |
                StopWhenEvolutionStagnates(5, 100, 0.3),
            rng = MersenneTwister(123),
            return_state = true,
        )
        flat_sc = only(get_active_stopping_criteria(o_flat.stop))
        @test flat_sc isa StopWhenEvolutionStagnates
        @test Manopt.indicates_convergence(flat_sc)
        @test startswith(repr(flat_sc), "StopWhenEvolutionStagnates(")

        o_flat = cma_es(
            M,
            flat_example,
            [10.0, 10.0];
            σ = 10.0,
            stopping_criterion = StopAfterIteration(1000) |
                StopWhenPopulationStronglyConcentrated(1.0e-5),
            rng = MersenneTwister(12),
            return_state = true,
        )
        flat_sc = only(get_active_stopping_criteria(o_flat.stop))
        @test flat_sc isa StopWhenPopulationStronglyConcentrated
        @test Manopt.indicates_convergence(flat_sc)
        @test startswith(repr(flat_sc), "StopWhenPopulationStronglyConcentrated(")

        o_flat = cma_es(
            M,
            flat_example,
            [10.0, 10.0];
            σ = 10.0,
            stopping_criterion = StopAfterIteration(500) |
                StopWhenPopulationCostConcentrated(1.0e-5, 5),
            rng = MersenneTwister(123),
            return_state = true,
        )
        flat_sc = only(get_active_stopping_criteria(o_flat.stop))
        @test flat_sc isa StopWhenPopulationCostConcentrated
        @test Manopt.indicates_convergence(flat_sc)
        @test startswith(repr(flat_sc), "StopWhenPopulationCostConcentrated(")

        # test handling of negative covariance matrix eigenvalues
        @test_warn "Covariance matrix has nonpositive eigenvalues" o_flat = cma_es(
            M,
            flat_example,
            [10.0, 10.0];
            σ = 10.0,
            stopping_criterion = StopAfterIteration(10000) |
                StopWhenPopulationStronglyConcentrated(1.0e-14),
            rng = MersenneTwister(12),
            return_state = true,
        )
        flat_sc = only(get_active_stopping_criteria(o_flat.stop))
        @test flat_sc isa StopWhenPopulationStronglyConcentrated
        @test Manopt.indicates_convergence(flat_sc)
        @test startswith(repr(flat_sc), "StopWhenPopulationStronglyConcentrated(")
    end
    @testset "Spherical CMA-ES" begin
        M = Sphere(2)

        p1 = cma_es(M, griewank, [0.0, 1.0, 0.0]; σ = 1.0, rng = MersenneTwister(123))
        @test griewank(M, p1) < 0.17
    end
    @testset "Special Stopping Criteria" begin
        sc1 = StopWhenBestCostInGenerationConstant{Float64}(10)
        sc2 = StopWhenEvolutionStagnates(1, 2, 0.5)
        @test contains(Manopt.status_summary(sc2), "not yet filled")
        sc3 = StopWhenPopulationStronglyConcentrated(0.1)
        sc4 = StopWhenPopulationCostConcentrated(0.1, 5)
        sc5 = StopWhenCovarianceIllConditioned(1.0e-5)
        sc6 = StopWhenPopulationDiverges(0.1)
        for sc in [sc1, sc2, sc3, sc4, sc5, sc6]
            @test get_reason(sc) == ""
            # Manually set is active
            sc.at_iteration = 10
            @test length(get_reason(sc)) > 0
        end
    end
end
