using ManifoldsBase, Manifolds, Manopt, Test, RecursiveArrayTools

@testset "InteriorPointNewtonState" begin
    M = ManifoldsBase.DefaultManifold(3)
    # Cost
    f(::ManifoldsBase.DefaultManifold, p) = norm(p)^2
    grad_f(M, p) = 2 * p
    hess_f(M, p, X) = [2.0, 2.0, 2.0]
    # Inequality constraints
    g(M, p) = [p[1] - 1, -p[2] - 1]
    # # Function
    grad_g(M, p) = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    hess_g(M, p, X) = [copy(X), -copy(X)]

    h(M, p) = [2 * p[3] - 1]
    grad_h(M, p) = [[0.0, 0.0, 2.0]]
    hess_h(M, p, X) = [[0.0, 0.0, 0.0]]

    #A set of values for an example point and tangent
    p = [1.0, 2.0, 3.0]
    c = [[0.0, -3.0], [5.0]]
    fp = 14.0
    gg = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    gh = [[0.0, 0.0, 2.0]]
    gf = 2 * p
    X = [1.0, 0.0, 0.0]
    hf = [2.0, 2.0, 2.0]
    hg = [X, -X]
    hh = [[0.0, 0.0, 0.0]]

    μ = [1.0, 1.0]
    λ = [1.0]
    β = 7.0
    s = [1.0, 2.0]
    β = 6.0
    step_M = M × ℝ^2 × ℝ^1 × ℝ^2
    step_p = rand(step_M)
    step_p[step_M, 1] = p
    step_p[step_M, 2] = μ
    step_p[step_M, 3] = λ
    step_p[step_M, 4] = s
    sub_M = M × ℝ^1
    sub_p = rand(sub_M)
    sub_p[sub_M, 1] = p
    sub_p[sub_M, 2] = λ
    coh = ConstrainedManifoldObjective(
        f, grad_f; hess_f = hess_f,
        g = g, grad_g = grad_g, hess_g = hess_g,
        h = h, grad_h = grad_h, hess_h = hess_h,
        M = M,
    )
    sub_obj = SymmetricLinearSystemObjective(
        CondensedKKTVectorFieldJacobian(coh, μ, s, β), CondensedKKTVectorField(coh, μ, s, β)
    )
    sub_state = ConjugateResidualState(TangentSpace(sub_M, sub_p), sub_obj)
    dmp = DefaultManoptProblem(M, coh)
    ipns = InteriorPointNewtonState(
        M, coh, DefaultManoptProblem(sub_M, sub_obj), sub_state; p = p
    )
    # Getters & Setters
    @test length(Manopt.get_message(ipns)) == 0
    @test set_iterate!(ipns, M, 2 * p) == ipns
    @test get_iterate(ipns) == 2 * p
    @test set_gradient!(ipns, M, 3 * p) == ipns
    @test get_gradient(ipns) == 3 * p
    show_str = "# Solver state for `Manopt.jl`s Interior Point Newton Method\n"
    @test startswith(Manopt.status_summary(ipns; context = :default), show_str)
    #
    sc = StopWhenKKTResidualLess(1.0e-5)
    @test length(get_reason(sc)) == 0
    @test !sc(dmp, ipns, 1) #not yet reached
    @test Manopt.indicates_convergence(sc)
    @test startswith(repr(sc), "StopWhenKKTResidualLess(1.0e-5)")
    # Fake stop
    sc.residual = 1.0e-7
    sc.at_iteration = 1
    @test length(get_reason(sc)) > 0
    #
    ipcc = InteriorPointCentralityCondition(coh, 1.0)
    @test Manopt.set_parameter!(ipcc, :τ, step_M, step_p) == ipcc
    @test Manopt.set_parameter!(ipcc, :γ, 2.0) == ipcc
    @test Manopt.get_parameter(ipcc, :γ) == 2.0
    @test Manopt.get_parameter(ipcc, :τ1) == 2 / 3
    @test Manopt.get_parameter(ipcc, :τ2) ≈ 0.2809757 atol = 1.0e-7
    @test !ipcc(step_M, step_p)
    ipcc.τ1 = 0.01 # trick conditions so ipcc succeeds
    ipcc.τ2 = 0.01
    @test ipcc(step_M, step_p)
end
