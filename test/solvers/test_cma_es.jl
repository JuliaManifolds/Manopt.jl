using Manopt, ManifoldsBase, Manifolds, Test
using LinearAlgebra, Random

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
        @test griewank(M, p1) < 0.25

        p1 = [10.0, 10.0]
        r1 = cma_es!(M, griewank, p1; σ = 10.0, rng = MersenneTwister(123))
        @test r1 === p1 # in-place: the passed point holds the result (best visited)
        @test griewank(M, p1) < 0.2

        o = cma_es(M, griewank, [10.0, 10.0]; return_state = true)
        @test startswith(
            Manopt.status_summary(o; context = :default),
            "# Solver state for `Manopt.jl`s Covariance Matrix Adaptation Evolutionary Strategy",
        )
        @test contains(
            Manopt.status_summary(o; context = :inline), "covariance matrix adaptation"
        )

        @testset "Callbacks" begin
            sk_record = Tuple{Symbol, Int}[]
            cb(symbol, problem, state, k) = push!(sk_record, (symbol, k))
            cma_es(
                M, griewank, [10.0, 10.0];
                callbacks = cb,
                stopping_criterion = StopAfterIteration(1),
                rng = MersenneTwister(123),
            )
            @test sk_record == [
                (:BeforeInit, 0),
                (:Init, 0),
                (:BeforeStop, 0),
                (:BeforeStep, 1),
                (:Step, 1),
                (:BeforeStop, 1),
                (:Stop, 1),
            ]
        end

        o_d = cma_es(
            M, divergent_example, [10.0, 10.0]; σ = 10.0, rng = MersenneTwister(123),
            return_state = true,
        )
        div_sc = only(get_active_stopping_criteria(o_d.stop))
        @test div_sc isa StopWhenPopulationDiverges
        @test !Manopt.indicates_convergence(div_sc)
        @test startswith(repr(div_sc), "StopWhenPopulationDiverges(")

        o_d = cma_es(
            M, poorly_conditioned_example, [10.0, 10.0];
            σ = 10.0, rng = MersenneTwister(123), return_state = true,
        )
        condcov_sc = only(get_active_stopping_criteria(o_d.stop))
        @test condcov_sc isa StopWhenCovarianceIllConditioned
        @test !Manopt.indicates_convergence(condcov_sc)
        @test startswith(repr(condcov_sc), "StopWhenCovarianceIllConditioned(")

        o_flat = cma_es(
            M, flat_example, [10.0, 10.0]; σ = 10.0,
            stopping_criterion = StopAfterIteration(500) | StopWhenBestCostInGenerationConstant{Float64}(5),
            rng = MersenneTwister(123), return_state = true,
        )
        flat_sc = only(get_active_stopping_criteria(o_flat.stop))
        @test flat_sc isa StopWhenBestCostInGenerationConstant
        @test Manopt.indicates_convergence(flat_sc)
        @test startswith(repr(flat_sc), "StopWhenBestCostInGenerationConstant(")

        o_flat = cma_es(
            M, flat_example, [10.0, 10.0]; σ = 10.0,
            stopping_criterion = StopAfterIteration(500) | StopWhenEvolutionStagnates(5, 100, 0.3),
            rng = MersenneTwister(123), return_state = true,
        )
        flat_sc = only(get_active_stopping_criteria(o_flat.stop))
        @test flat_sc isa StopWhenEvolutionStagnates
        @test Manopt.indicates_convergence(flat_sc)
        @test startswith(repr(flat_sc), "StopWhenEvolutionStagnates(")

        o_flat = cma_es(
            M, flat_example, [10.0, 10.0]; σ = 10.0,
            stopping_criterion = StopAfterIteration(1000) | StopWhenPopulationStronglyConcentrated(1.0e-5),
            rng = MersenneTwister(12), return_state = true,
        )
        flat_sc = only(get_active_stopping_criteria(o_flat.stop))
        @test flat_sc isa StopWhenPopulationStronglyConcentrated
        @test Manopt.indicates_convergence(flat_sc)
        @test startswith(repr(flat_sc), "StopWhenPopulationStronglyConcentrated(")

        o_flat = cma_es(
            M, flat_example, [10.0, 10.0]; σ = 10.0,
            stopping_criterion = StopAfterIteration(500) | StopWhenPopulationCostConcentrated(1.0e-5, 5),
            rng = MersenneTwister(123), return_state = true,
        )
        flat_sc = only(get_active_stopping_criteria(o_flat.stop))
        @test flat_sc isa StopWhenPopulationCostConcentrated
        @test Manopt.indicates_convergence(flat_sc)
        @test startswith(repr(flat_sc), "StopWhenPopulationCostConcentrated(")

        # test handling of negative covariance matrix eigenvalues
        @test_warn "Covariance matrix has nonpositive eigenvalues" o_flat = cma_es(
            M, flat_example, [10.0, 10.0]; σ = 10.0,
            stopping_criterion = StopAfterIteration(10000) | StopWhenPopulationStronglyConcentrated(1.0e-14),
            rng = MersenneTwister(13), return_state = true,
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
        # with filled histories the summary reports the two medians
        push!(sc2.best_history, 1.0); push!(sc2.best_history, 1.0)
        push!(sc2.median_history, 2.0); push!(sc2.median_history, 2.0)
        @test contains(Manopt.status_summary(sc2), "the best mean did not decrease 1.0 <= 1.0")
        @test contains(Manopt.status_summary(sc2; context = :inline), "1.0 <= 1.0 && 2.0 <= 2.0")
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
        # the two tolerances are independent, so switching one of them off is allowed
        M = Euclidean(2)
        @test Manopt.default_cma_es_stopping_criterion(M, 6; tol_fun = 0) isa StoppingCriterion
        @test Manopt.default_cma_es_stopping_criterion(M, 6; tol_x = 0) isa StoppingCriterion
        # a reused criterion starts counting from zero again
        st = Manopt.get_state(
            cma_es(
                M, flat_example, [10.0, 10.0]; σ = 10.0, rng = MersenneTwister(123),
                stopping_criterion = StopAfterIteration(20) | StopWhenBestCostInGenerationConstant{Float64}(5),
                return_state = true,
            )
        )
        @test startswith(repr(st), "CMAESState(M, ")
        sc7 = only(get_active_stopping_criteria(st.stop))
        @test sc7.iterations_since_change > 0
        sc7(DefaultManoptProblem(M, ManifoldCostObjective(flat_example)), st, 0)
        @test sc7.iterations_since_change == 0
        # the same for the history of the best costs
        mp_flat = DefaultManoptProblem(M, ManifoldCostObjective(flat_example))
        sc8 = StopWhenPopulationCostConcentrated(1.0e-5, 3)
        @test [sc8(mp_flat, st, k) for k in 1:3] == [false, false, true]
        sc8(mp_flat, st, 0)
        @test length(sc8.best_value_history) == 0
        # a criterion is active only once it stopped, not when its condition holds
        sc9 = StopWhenBestCostInGenerationConstant{Float64}(3)
        sc9(mp_flat, st, 0)
        for k in 1:4 # the first call records the cost, the next three count
            sc9(mp_flat, st, k)
        end
        @test sc9.iterations_since_change == 3
        @test !Manopt.is_active_stopping_criterion(sc9) # the counter reached 3, the next call fires
        @test sc9(mp_flat, st, 5)
        @test Manopt.is_active_stopping_criterion(sc9)
        @test [sc8(mp_flat, st, k) for k in 1:3] == [false, false, true]
    end
    @testset "The covariance matrix of the state is kept" begin
        M2 = Euclidean(2)
        C0 = [4.0 1.0; 1.0 2.0]
        st = CMAESState(
            M2, [2.0, 2.0], 2, 5, 1.5, 0.1, 0.2, 0.3, 0.4, 1.0, 1.2, StopAfterIteration(1),
            copy(C0), 1.0, [0.6, 0.4, 0.0, -0.3, -0.7],
        )
        Manopt.initialize_solver!(DefaultManoptProblem(M2, ManifoldCostObjective((M, p) -> sum(abs2, p))), st)
        @test st.covariance_matrix == C0
        @test st.deviations ≈ sqrt.(eigvals(Symmetric(C0)))
        @test st.covariance_matrix_cond ≈ cond(C0)
    end
    @testset "Objectives and numbers as points" begin
        M = Euclidean(2)
        p0 = [1.0, 1.0]
        # the allocating entry also takes an objective, like the in-place one
        mco = ManifoldCostObjective(griewank)
        q = cma_es(M, mco, p0; rng = MersenneTwister(123))
        q2 = copy(M, p0)
        cma_es!(M, mco, q2; rng = MersenneTwister(123))
        @test isapprox(M, q, q2)
        # and a manifold whose points are numbers works
        Mc = Circle()
        fc(N, r) = (r - 0.3)^2
        qc = cma_es(Mc, fc, 0.5; rng = MersenneTwister(1))
        @test qc isa Float64
        @test isapprox(Mc, qc, 0.3; atol = 1.0e-6)
        # the number types of the point and of σ are free
        fq(M, p) = sum(abs2, p .- 1)
        kw = (; stopping_criterion = StopAfterIteration(20))
        q64 = cma_es(M, fq, [2.0, 2.0]; rng = MersenneTwister(42), kw...)
        q32 = cma_es(M, fq, Float32[2.0, 2.0]; rng = MersenneTwister(42), kw...)
        @test q32 isa Vector{Float32}
        @test q32 ≈ q64
        @test cma_es(M, fq, [2.0, 2.0]; σ = 1.0f0, rng = MersenneTwister(42), kw...) == q64
        @test cma_es(M, fq, [2.0, 2.0]; σ = 1, rng = MersenneTwister(42), kw...) == q64
    end
end
