using Manopt, ManifoldsBase, Manifolds, Test
using LinearAlgebra: I, tr

@testset "Test REPM with a nonneg. PCA" begin
    d = 4
    M = Sphere(d - 1)
    S = [ones(2)..., zeros(d - 2)...]
    v0 = project(M, S)
    Z = v0 * v0'
    f(M, p) = -tr(transpose(p) * Z * p) / 2
    grad_f(M, p) = project(M, p, -transpose.(Z) * p / 2 - Z * p / 2)
    g(M, p) = -p # inequality constraint p ≥ 0
    mI = -Matrix{Float64}(I, d, d)
    grad_g(M, p) = [project(M, p, mI[:, i]) for i in 1:d]
    p0 = project(M, [ones(2)..., zeros(d - 3)..., 0.1])
    sol_lse = exact_penalty_method(M, f, grad_f, p0; g = g, grad_g = grad_g)
    sol_lse2 = exact_penalty_method(M, f, grad_f; g = g, grad_g = grad_g)
    sol_lqh = exact_penalty_method(
        M, f, grad_f, p0; g = g, grad_g = grad_g, smoothing = LinearQuadraticHuber()
    )
    sol_lqh2 = copy(M, p0)
    exact_penalty_method!(
        M, f, grad_f, sol_lqh2; g = g, grad_g = grad_g, smoothing = LinearQuadraticHuber()
    )
    sol_lqh3 = copy(M, p0)
    exact_penalty_method!(
        M, f, grad_f, sol_lqh3;
        g = g, grad_g = grad_g, smoothing = LinearQuadraticHuber(),
        gradient_inequality_range = NestedPowerRepresentation(),
    )
    # allocating entry must forward inequality_constraints (was a copy-paste of equality_constraints)
    g!(M, V, p) = (V .= -p; V)
    grad_g!(M, X, p) = (
        for i in 1:d
            copyto!(X[i], project(M, p, mI[:, i]))
        end; X
    )
    grad_f!(M, X, p) = copyto!(X, grad_f(M, p))
    sol_ip = exact_penalty_method(
        M, f, grad_f!, p0; g = g!, grad_g = grad_g!,
        inequality_constraints = d, evaluation = InplaceEvaluation(),
    )
    a_tol_emp = 8.0e-2
    @test isapprox(M, v0, sol_lse; atol = a_tol_emp)
    @test isapprox(M, v0, sol_ip; atol = a_tol_emp)
    @test isapprox(M, v0, sol_lse2; atol = a_tol_emp)
    @test isapprox(M, v0, sol_lqh; atol = a_tol_emp)
    @test isapprox(M, v0, sol_lqh2; atol = a_tol_emp)
    @test isapprox(M, v0, sol_lqh3; atol = a_tol_emp)
    # Dummy options
    mco = ManifoldCostObjective(f)
    dmp = DefaultManoptProblem(M, mco)
    epms = ExactPenaltyMethodState(M, dmp, NelderMeadState(M); p = p0)
    @test Manopt.get_message(epms) == ""
    set_iterate!(epms, M, 2 .* p0)
    @test get_iterate(epms) == 2 .* p0
    @test startswith(Manopt.status_summary(epms; context = :default), "# Solver state for `Manopt.jl`s Exact Penalty Method\n")
    @test startswith(repr(epms), "ExactPenaltyMethodState($(dmp)")
    # With dummy closed form solution
    epmsc = ExactPenaltyMethodState(M, f)
    @test epmsc.sub_state isa Manopt.ClosedFormSubSolverState
    epmsc2 = ExactPenaltyMethodState(M, f, AllocatingEvaluation())
    @test epmsc2.sub_state isa Manopt.ClosedFormSubSolverState
    @testset "closed form sub solver can take a step" begin
        # the closed form state could be built but had no `step_solver!` method at all
        co = ConstrainedManifoldObjective(f, grad_f; g = g, grad_g = grad_g, M = M)
        cmp = DefaultManoptProblem(M, co)
        step(M, p) = exp(M, p, -0.05 .* grad_f(M, p))
        closed_a(M, ρ, u, p) = step(M, p)                       # allocating
        closed_i!(M, q, ρ, u, p) = copyto!(M, q, step(M, p))    # in place
        ea = ExactPenaltyMethodState(M, closed_a; p = copy(M, p0))
        ei = ExactPenaltyMethodState(M, closed_i!, InplaceEvaluation(); p = copy(M, p0))
        @test ea.sub_problem isa Manopt.InplaceManifoldFunction
        @test ei.sub_problem === closed_i!
        for s in (ea, ei)
            @test Manopt.step_solver!(cmp, s, 1) === s
            @test is_point(M, get_iterate(s))
        end
        @test isapprox(M, get_iterate(ea), get_iterate(ei))
        @test isapprox(M, get_iterate(ea), step(M, p0))
        @test ea.ϵ < 1.0e-3 # the tolerance update ran
    end
    # that is errors with just Manifold + State
    @test_throws ErrorException ExactPenaltyMethodState(M, Manopt.Test.DummyState())
    @testset "Numbers" begin
        Me = Euclidean()
        fe(M, p) = (p + 5)^2
        grad_fe(M, p) = 2 * p + 10
        ge(M, p) = -p # inequality constraint p ≥ 0
        grad_ge(M, p) = -1
        s = exact_penalty_method(
            Me, fe, grad_fe, 4.0;
            g = ge, grad_g = grad_ge, stopping_criterion = StopAfterIteration(20),
            return_state = true,
        )
        q = get_solver_result(s)[]
        @test q isa Real
        @test fe(Me, q) < fe(Me, 4.0)
    end
    @testset "Callbacks" begin
        sk_record = Tuple{Symbol, Int}[]
        cb(symbol, problem, state, k) = append!(sk_record, [(symbol, k)])
        exact_penalty_method!(
            M, f, grad_f, sol_lqh2; g = g, grad_g = grad_g, smoothing = LinearQuadraticHuber(),
            callbacks = cb, stopping_criterion = StopAfterIteration(1)
        )
        @test sk_record == [
            (:BeforeInit, 0), (:Init, 0), (:BeforeStop, 0),
            (:BeforeStep, 1), (:BeforeSubsolver, 1), (:Subsolver, 1), (:Step, 1), (:BeforeStop, 1), (:Stop, 1),
        ]
    end
end
