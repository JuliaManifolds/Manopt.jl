using Manopt, ManifoldsBase, Manifolds, Test
using LinearAlgebra: I, tr

@testset "Riemannian Augmented Lagrangian Method" begin
    @testset "Test RALM with a nonneg. PCA" begin
        d = 20
        M = Sphere(d - 1)
        S = [ones(4)..., zeros(d - 4)...]
        v0 = project(M, S)
        Z = v0 * v0'
        f(M, p) = -tr(transpose(p) * Z * p) / 2
        grad_f(M, p) = project(M, p, -transpose.(Z) * p / 2 - Z * p / 2)
        g(M, p) = -p # in other words p ≥ 0
        mI = -Matrix{Float64}(I, d, d)
        grad_g(M, p) = [project(M, p, mI[:, i]) for i in 1:d]
        p0 = project(M, ones(d))
        # This run (and the following as well) seem to stall after iteration 3 (previously 14)
        # not reaching the minimiser as before
        sol = augmented_Lagrangian_method(M, f, grad_f, p0; g = g, grad_g = grad_g)
        @test distance(M, sol, v0) < 8 * 1.0e-4
        sol2 = copy(M, p0)
        augmented_Lagrangian_method!(M, f, grad_f, sol2; g = g, grad_g = grad_g)
        @test sol2 == sol
        sol3 = copy(M, p0)
        augmented_Lagrangian_method!(
            M, f, grad_f, sol3;
            g = g, grad_g = grad_g, gradient_inequality_range = NestedPowerRepresentation(),
        )
        @test sol3 ≈ sol atol = 5.0e-5
        co = ConstrainedManifoldObjective(f, grad_f; g = g, grad_g = grad_g, M = M)
        mp = DefaultManoptProblem(M, co)
        # dummy ALM problem
        sp = DefaultManoptProblem(M, ManifoldCostObjective(f))
        ss = NelderMeadState(M)
        alms = AugmentedLagrangianMethodState(M, co, sp, ss; p = p0)
        set_iterate!(alms, M, 2 .* p0)
        @test Manopt.get_message(alms) == ""
        @test get_iterate(alms) == 2 .* p0
        @test startswith(
            Manopt.status_summary(alms; context = :default),
            "# Solver state for `Manopt.jl`s Augmented Lagrangian Method\n"
        )
        @test Manopt.get_sub_problem(alms) === sp
        @test Manopt.get_sub_state(alms) === ss
        # With dummy closed form solution
        almsc = AugmentedLagrangianMethodState(M, co, f)
        @test almsc.sub_state isa Manopt.ClosedFormSubSolverState

        alm_record = Tuple{Symbol, Int}[]
        alm_cb(symbol, problem, state, k) = append!(alm_record, [(symbol, k)])
        augmented_Lagrangian_method(
            M, f, grad_f, p0;
            g = g, grad_g = grad_g, stopping_criterion = StopAfterIteration(1), callbacks = alm_cb,
        )
        @test alm_record == [
            (:BeforeInit, 0), (:Init, 0), (:BeforeStop, 0),
            (:BeforeStep, 1), (:Subsolver, 1), (:Step, 1), (:BeforeStop, 1), (:Stop, 1),
        ]
        almsc_cb = AugmentedLagrangianMethodState(
            M, co, f; callbacks = Dict(:Step => alm_cb),
        )
        @test occursin("active callbacks", Manopt.status_summary(almsc_cb; context = :default))
    end
    @testset "Numbers" begin
        M = Euclidean()
        f(M, p) = (p + 5)^2
        grad_f(M, p) = 2 * p + 10
        g(M, p) = -p # in other words p ≥ 0
        grad_g(M, p) = -1.0
        s = augmented_Lagrangian_method(
            M, f, grad_f, 4.0;
            g = g, grad_g = grad_g,
            stopping_criterion = StopAfterIteration(20), return_state = true,
        )
        q = get_solver_result(s)[]
        @test q isa Real
        @test f(M, q) < f(M, 4)
    end
end
