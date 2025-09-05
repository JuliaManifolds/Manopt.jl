using LinearAlgebra, Manifolds, Manopt, Random, Test
import Manifolds: inner

@testset "Difference of Convex" begin
    g(M, p) = log(det(p))^4 + 1 / 4
    grad_g(M, p) = 4 * (log(det(p)))^3 * p
    function grad_g!(M, X, p)
        copyto!(M, X, p)
        X .*= 4 * (log(det(p)))^3
        return X
    end
    h(M, p) = log(det(p))^2
    grad_h(M, p) = 2 * log(det(p)) * p
    function grad_h!(M, X, p)
        copyto!(M, X, p)
        X .*= 2 * (log(det(p)))
        return X
    end
    f(M, p) = g(M, p) - h(M, p)
    grad_f(M, p) = grad_g(M, p) - grad_h(M, p)

    n = 2
    M = SymmetricPositiveDefinite(n)
    p0 = log(2) * Matrix{Float64}(I, n, n)

    @testset "DC States" begin
        p1 = Matrix{Float64}(I, n, n)
        X1 = ones(2, 2)

        dca_sub_cost = LinearizedDCCost(g, p0, X1)
        dca_sub_grad = LinearizedDCGrad(grad_g, p0, X1)
        dca_sub_grad! = LinearizedDCGrad(grad_g!, p0, X1; evaluation = InplaceEvaluation())
        X2 = dca_sub_grad!(M, p0)
        X3 = similar(X2)
        dca_sub_grad(M, X3, p0)
        @test X2 == X3
        dca_sub_objective = ManifoldGradientObjective(dca_sub_cost, dca_sub_grad)
        dca_sub_problem = DefaultManoptProblem(M, dca_sub_objective)
        dca_sub_state = GradientDescentState(M; p = copy(M, p0))

        dcs = DifferenceOfConvexState(M, dca_sub_problem, dca_sub_state; p = copy(M, p0))
        @test Manopt.get_message(dcs) == ""
        dcsc = DifferenceOfConvexState(M, f)
        @test dcsc.sub_state isa Manopt.ClosedFormSubSolverState

        set_iterate!(dcs, M, p1)
        @test dcs.p == p1
        set_gradient!(dcs, M, p1, X1)
        @test dcs.X == X1
        Manopt.set_parameter!(dcs, :SubProblem, :X, X1)
        Manopt.set_parameter!(dcs, :SubState, :X, X1)

        dcppa_sub_cost = ProximalDCCost(g, copy(M, p0), 1.0)
        dcppa_sub_grad = ProximalDCGrad(grad_g, copy(M, p0), 1.0)
        dcppa_sub_grad! = ProximalDCGrad(
            grad_g!, copy(M, p0), 1.0; evaluation = InplaceEvaluation()
        )
        Y1 = dcppa_sub_grad!(M, p0)
        Y2 = similar(Y1)
        dcppa_sub_grad(M, Y2, p0)
        @test Y1 == Y2

        dcppa_sub_objective = ManifoldGradientObjective(dcppa_sub_cost, dcppa_sub_grad)
        dcppa_sub_problem = DefaultManoptProblem(M, dcppa_sub_objective)
        dcppa_sub_state = GradientDescentState(M; p = copy(M, p0))

        dcps = DifferenceOfConvexProximalState( #Initialize with random point
            M,
            dcppa_sub_problem,
            dcppa_sub_state,
        )
        set_iterate!(dcps, M, p1)
        @test dcps.p == p1
        set_gradient!(dcps, M, p1, X1)
        @test dcps.X == X1
        # Dummy closed form sub
        dcpsc = DifferenceOfConvexProximalState(M, f)
        @test dcpsc.sub_state isa Manopt.ClosedFormSubSolverState

        dc_cost_a = ManifoldDifferenceOfConvexObjective(f, grad_h)
        @test_throws ErrorException difference_of_convex_algorithm(
            M, dc_cost_a, p1; grad_g = grad_g
        )
        @test_throws ErrorException difference_of_convex_algorithm(M, dc_cost_a, p1; g = g)
        dc_cost_i = ManifoldDifferenceOfConvexObjective(
            f, grad_h!; evaluation = InplaceEvaluation()
        )
        dcp = DefaultManoptProblem(M, dc_cost_a)

        X4 = get_subtrahend_gradient(dcp, p0)
        @test X4 == grad_h(M, p0)
        X5 = get_subtrahend_gradient(M, dc_cost_a, p0)
        @test X5 == grad_h(M, p0)
        X6 = get_subtrahend_gradient(M, dc_cost_i, p0)
        @test X6 == grad_h(M, p0)

        dcp_cost_a = ManifoldDifferenceOfConvexProximalObjective(grad_h)
        dcp_cost_i = ManifoldDifferenceOfConvexProximalObjective(
            grad_h!; evaluation = InplaceEvaluation()
        )
        dcpp = DefaultManoptProblem(M, dcp_cost_a)

        X4 = get_subtrahend_gradient(dcpp, p0)
        @test X4 == grad_h(M, p0)
        X5 = get_subtrahend_gradient(M, dcp_cost_a, p0)
        @test X5 == grad_h(M, p0)
        X6 = get_subtrahend_gradient(M, dcp_cost_i, p0)
        @test X6 == grad_h(M, p0)
    end
    @testset "Running the subsolver algorithms" begin
        p1 = difference_of_convex_algorithm(
            M, f, g, grad_h!, p0; grad_g = (grad_g!), evaluation = InplaceEvaluation()
        )
        p2 = difference_of_convex_algorithm(M, f, g, grad_h, p0; grad_g = grad_g)
        s1 = difference_of_convex_algorithm(
            M, f, g, grad_h, p0; grad_g = grad_g, gradient = grad_f, return_state = true
        )
        @test startswith(
            repr(s1), "# Solver state for `Manopt.jl`s Difference of Convex Algorithm\n"
        )
        p3 = get_solver_result(s1)
        @test Manopt.get_message(s1) == "" # no message in last step
        @test isapprox(M, p1, p2)
        @test isapprox(M, p2, p3)
        @test isapprox(f(M, p1), 0.0; atol = 1.0e-8)
        # not provided `grad_g` or problem nothing
        @test_throws ErrorException difference_of_convex_algorithm(
            M, f, g, grad_h, p0; sub_problem = nothing
        )
        @test_throws ErrorException difference_of_convex_algorithm(
            M, f, g, grad_h, p0; sub_hessian = nothing
        )
        @test_throws ErrorException difference_of_convex_algorithm(M, f, g, grad_h, p0)

        p4 = difference_of_convex_proximal_point(
            M, grad_h!, p0; g = g, grad_g = (grad_g!), evaluation = InplaceEvaluation()
        )
        p5 = difference_of_convex_proximal_point(M, grad_h, p0; g = g, grad_g = grad_g)
        p5b = difference_of_convex_proximal_point(M, grad_h; g = g, grad_g = grad_g)
        # using gradient descent
        p5c = difference_of_convex_proximal_point(
            M,
            grad_h,
            p0;
            g = g,
            grad_g = grad_g,
            sub_hess = nothing,
            stopping_criterion = StopAfterIteration(10), # is not that stable
        )
        s2 = difference_of_convex_proximal_point(
            M, grad_h, p0; g = g, grad_g = grad_g, gradient = grad_f, return_state = true
        )
        @test startswith(
            repr(s2),
            "# Solver state for `Manopt.jl`s Difference of Convex Proximal Point Algorithm\n",
        )
        p6 = get_solver_result(s2)
        @test Manopt.get_message(s2) == ""

        @test isapprox(M, p3, p4)
        @test isapprox(M, p4, p5)
        @test isapprox(M, p5, p6)
        @test isapprox(f(M, p5b), 0.0; atol = 2.0e-16) # bit might be a different min due to rand
        @test isapprox(f(M, p5c), 0.0; atol = 1.0e-10)
        @test isapprox(f(M, p4), 0.0; atol = 1.0e-14)

        Random.seed!(23)
        p7 = difference_of_convex_algorithm(M, f, g, grad_h; grad_g = grad_g)
        @test isapprox(f(M, p7), 0.0; atol = 1.0e-8)

        p8 = copy(M, p0) # Same call as p2 in-place
        difference_of_convex_algorithm!(M, f, g, grad_h, p8; grad_g = grad_g)
        @test isapprox(M, p8, p2)

        # using GD - only very imprecise
        p9 = difference_of_convex_algorithm(
            M, f, g, grad_h, p0; grad_g = grad_g, sub_hess = nothing
        )
        @test isapprox(M, p9, p2; atol = 1.0e-7)

        @test_throws ErrorException difference_of_convex_proximal_point(
            M, grad_h, p0; sub_problem = nothing
        )
        @test_throws ErrorException difference_of_convex_proximal_point(M, grad_h, p0)
        @test_throws ErrorException difference_of_convex_proximal_point(
            M, grad_h, p0; g = g, grad_g = grad_g, sub_grad = nothing
        )
        # both g and grad g required here
        @test_throws ErrorException difference_of_convex_proximal_point(M, grad_h, p0; g = g)
        @test_throws ErrorException difference_of_convex_proximal_point(
            M, grad_h, p0; grad_g = grad_g
        )
    end
    @testset "Running the closed form solution solvers" begin
        # make them a bit by providing sub solvers as functions
        function dca_sub(M, p, X)
            q = copy(M, p)
            lin_s = LinearizedDCCost(g, copy(M, p), copy(M, p, X))
            grad_lin_s = LinearizedDCGrad(grad_g, copy(M, p), copy(M, p, X))
            hess_lin_s = ApproxHessianFiniteDifference(M, copy(M, p), grad_lin_s)
            trust_regions!(M, lin_s, grad_lin_s, hess_lin_s, q)
            return q
        end
        p11 = difference_of_convex_algorithm(M, f, g, grad_h, p0; sub_problem = dca_sub)
        function dca_sub!(M, q, p, X)
            copyto!(M, q, p)
            lin_s = LinearizedDCCost(g, copy(M, p), copy(M, p, X))
            grad_lin_s = LinearizedDCGrad(grad_g, copy(M, p), copy(M, p, X))
            hess_lin_s = ApproxHessianFiniteDifference(M, copy(M, p), grad_lin_s)
            trust_regions!(M, lin_s, grad_lin_s, hess_lin_s, q)
            return q
        end
        p12 = difference_of_convex_algorithm(
            M, f, g, grad_h!, p0; sub_problem = (dca_sub!), evaluation = InplaceEvaluation()
        )
        @test isapprox(M, p11, p12)
        @test f(M, p11) ≈ 0.0 atol = 1.0e-15

        # fake them a bit by providing sub solvers as functions
        function prox_g(M, λ, p)
            q = copy(M, p)
            prox = ProximalDCCost(g, copy(M, p), λ)
            grad_prox = ProximalDCGrad(grad_g, copy(M, p), λ)
            hess_prox = ApproxHessianFiniteDifference(M, copy(M, p), grad_prox)
            trust_regions!(M, prox, grad_prox, hess_prox, q)
            return q
        end
        p13 = difference_of_convex_proximal_point(M, grad_h, p0; prox_g = prox_g)
        p13b = copy(M, p0)
        difference_of_convex_proximal_point!(M, grad_h, p13b; prox_g = prox_g)
        @test isapprox(M, p13, p13b)
        function prox_g!(M, q, λ, p)
            copyto!(M, q, p)
            prox = ProximalDCCost(g, copy(M, p), λ)
            grad_prox = ProximalDCGrad(grad_g, copy(M, p), λ)
            hess_prox = ApproxHessianFiniteDifference(M, copy(M, p), grad_prox)
            trust_regions!(M, prox, grad_prox, hess_prox, q)
            return q
        end
        p14 = difference_of_convex_proximal_point(
            M, grad_h!, p0; prox_g = (prox_g!), evaluation = InplaceEvaluation()
        )
        @test isapprox(M, p13, p14)
        @test f(M, p13) ≈ 0.0 atol = 1.0e-15
    end
    @testset "On positive numbers" begin
        # Define in Manifolds.jl?
        Manifolds.inner(M::PositiveNumbers, p, X, Y) = inner(M, p[], X[], Y[])
        Mp = PositiveNumbers()
        pp = 1.0
        q = difference_of_convex_algorithm(Mp, f, g, grad_h, pp; grad_g = grad_g)
        @test pp == q # started in a minimizer
        q2 = difference_of_convex_proximal_point(Mp, grad_h, pp; g = g, grad_g = grad_g)
        @test pp == q2 # started in a in a minimizer
    end
end
