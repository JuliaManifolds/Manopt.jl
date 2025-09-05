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
        M,
        f,
        grad_f,
        sol_lqh3;
        g = g,
        grad_g = grad_g,
        smoothing = LinearQuadraticHuber(),
        gradient_inequality_range = NestedPowerRepresentation(),
    )
    a_tol_emp = 8.0e-2
    @test isapprox(M, v0, sol_lse; atol = a_tol_emp)
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
    @test startswith(repr(epms), "# Solver state for `Manopt.jl`s Exact Penalty Method\n")
    # With dummy closed form solution
    epmsc = ExactPenaltyMethodState(M, f)
    @test epmsc.sub_state isa Manopt.ClosedFormSubSolverState
    @testset "Numbers" begin
        Me = Euclidean()
        fe(M, p) = (p + 5)^2
        grad_fe(M, p) = 2 * p + 10
        ge(M, p) = -p # inequality constraint p ≥ 0
        grad_ge(M, p) = -1
        s = exact_penalty_method(
            Me,
            fe,
            grad_fe,
            4.0;
            g = ge,
            grad_g = grad_ge,
            stopping_criterion = StopAfterIteration(20),
            return_state = true,
        )
        q = get_solver_result(s)[]
        @test q isa Real
        @test fe(M, q) < fe(M, 4.0)
    end
end
