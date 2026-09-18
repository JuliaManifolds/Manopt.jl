using Manifolds, Manopt, Test, Random, ManifoldDiff
using Manopt: get_cost_function, get_gradient_function, get_differential_function
using LinearAlgebra: Symmetric

@testset "Counting" begin
    @testset "Basics" begin
        M = Sphere(2)
        A = [2.0 1.0 0.0; 1.0 2.0 1.0; 0.0 1.0 2.0]
        f(M, p) = p' * A * p
        grad_f(M, p) = project(M, p, 2 * A * p)
        obj = ManifoldGradientObjective(f, grad_f)
        c_obj = ManifoldCountObjective(M, obj, [:Cost, :Gradient, :Differential])
        # function access functions are different since the right is still counting.
        @test get_cost_function(obj) != get_cost_function(c_obj)
        @test get_gradient_function(obj) != get_gradient_function(c_obj)
        @test get_differential_function(obj) != get_differential_function(c_obj)
        p = [1.0, 0.0, 0.0]
        X = [1.0, 1.0, 0.0]
        get_cost(M, c_obj, p)
        @test get_count(c_obj, :Cost) == 1
        @test get_count(c_obj, :NonExistent) == -1
        Y = similar(X)
        get_gradient(M, c_obj, p)
        get_gradient!(M, Y, c_obj, p)
        # both are counted
        @test get_count(c_obj, :Gradient) == 2
        get_gradient(M, obj, p)
        # others do not affect the counter
        @test get_count(c_obj, :Gradient) == 2
        # Costgrad is counted
        Manopt.get_cost_and_gradient(M, c_obj, p)
        Manopt.get_cost_and_gradient!(M, Y, c_obj, p)
        @test get_count(c_obj, :Cost) == 3
        @test get_count(c_obj, :Gradient) == 4
        @test get_differential(M, c_obj, p, X) == get_differential(M, obj, p, X)
        @test get_count(c_obj, :Differential) == 1
        # the combined accessor counts both
        @test Manopt.get_cost_and_differential(M, c_obj, p, X) == Manopt.get_cost_and_differential(M, obj, p, X)
        @test get_count(c_obj, :Cost) == 4
        @test get_count(c_obj, :Differential) == 2
        # also decorated objects can be wrapped to be counted
        ro = Manopt.Test.DummyDecoratedObjective(obj)
        c_obj2 = ManifoldCountObjective(M, ro, [:Gradient])
        get_gradient(M, c_obj2, p)
        @test get_count(c_obj2, :Gradient) == 1
        @test_throws ErrorException get_count(ro, :Cost) # Errors since no CountObj
        @test get_count(c_obj2, :Cost) == -1 # Does not count cost
        @test_throws ErrorException get_count(c_obj2, :Cost, :error)
        # Dummy Hessian to call plain costgrad (half)
        obj3 = ManifoldHessianObjective(f, grad_f, (M, p, X) -> X)
        c_obj3 = ManifoldCountObjective(M, obj3, [:Gradient])
        # These fall back to cost and gradient separately, and they count
        @test Manopt.get_cost_and_gradient(M, c_obj3, p) == (f(M, p), grad_f(M, p))
        @test Manopt.get_cost_and_gradient!(M, zero_vector(M, p), c_obj3, p) ==
            (f(M, p), grad_f(M, p))
        @test get_count(c_obj3, :Gradient) == 2
        @test get_count(c_obj3, :Cost) == -1 # nonexistent
        @test startswith(repr(c_obj), "ManifoldCountObjective(ManifoldFirstOrderObjective")
        status_obj = Manopt.status_summary(c_obj)
        @test contains(status_obj, "## Statistics")
        io = IOBuffer()
        Manopt.status_summary(io, c_obj)
        @test String(take!(io)) == status_obj
        @test contains(Manopt.status_summary(c_obj; context = :inline), "(statistics:")
        rc_obj = Manopt.Test.DummyDecoratedObjective(c_obj)
        @test get_count(rc_obj, :Gradient) == 4 #still works if count is encapsulated
        @test_throws ErrorException get_count(obj, :Gradient) # no count objective
        @test get_count(rc_obj, :Gradient, 1) == 4 #still works if count is encapsulated
        @test_throws ErrorException get_count(obj, :Gradient, 1) # no count objective
        # test fallbacks
        @test get_count(c_obj, :None, 1) == -1
        @test get_count(c_obj, :Gradient, 2) == -1 # out of range
        @test get_count(c_obj, :Gradient, [2, 1]) == -1 #non-fitting dimensions
        reset_counters!(c_obj)
        @test get_count(c_obj, :Gradient) == 0
        @test get_count(c_obj, :Cost) == 0
        reset_counters!(rc_obj) # also works on decorated counters
        @test_throws ErrorException reset_counters!(obj) # errors on non-counter ones
    end
    @testset "Counting a single proximal map" begin
        M = Hyperbolic(2)
        p = [0.0, 0.0, 1.0]
        p0 = [1.0, 0.0, √2]
        g(M, q) = distance(M, q, p)^2
        grad_g(M, q) = -2 * log(M, q, p)
        h(M, q) = distance(M, q, p)
        prox_h(M, λ, q) = ManifoldDiff.prox_distance(M, λ, p, q, 1)
        f(M, q) = g(M, q) + h(M, q)
        ob = ManifoldProximalGradientObjective(f, g, grad_g, prox_h)
        c_ob = ManifoldCountObjective(M, ob, [:ProximalMap]; p = p0)
        @test get_count(c_ob, :ProximalMap) == 0
        get_proximal_map(M, c_ob, 1.0, p0)
        @test get_count(c_ob, :ProximalMap) == 1
        # and through the solver
        c_ob2, q = proximal_gradient_method(
            M, f, g, grad_g, p0;
            prox_nonsmooth = prox_h, count = [:ProximalMap],
            stopping_criterion = StopAfterIteration(3), return_objective = true,
        )
        @test get_count(c_ob2, :ProximalMap) == 6
    end
    @testset "Function passthrough" begin
        Random.seed!(42)
        n = 4
        A = Symmetric(randn(n, n))
        M = Sphere(n - 1)
        p = [1.0, zeros(n - 1)...]
        X = [0.0, 1.0, zeros(n - 2)...]
        f(M, p) = 0.5 * p' * A * p
        grad_f(M, p) = A * p - (p' * A * p) * p
        Hess_f(M, p, X) = A * X - (p' * A * X) .* p - (p' * A * p) .* X
        obj = ManifoldHessianObjective(f, grad_f, Hess_f)
        # the subgradient function of a counting objective counts
        sgo = ManifoldSubgradientObjective(f, grad_f)
        sgc = ManifoldCountObjective(M, sgo, [:SubGradient])
        @test Manopt.get_subgradient_function(sgc)(M, p) == grad_f(M, p)
        @test get_count(sgc, :SubGradient) == 1
        @test Manopt.get_subgradient_function(sgc, true) === grad_f
        c_obj = ManifoldCountObjective(M, obj, [:Cost, :Gradient, :Hessian])
        # undecorated / recursive cost -> exactly f
        @test Manopt.get_cost_function(obj) === Manopt.get_cost_function(c_obj, true)
        # PLain get_diff passthrough
        @test get_differential(M, c_obj, p, X) == get_differential(M, obj, p, X)
        # otherwise different
        f1 = get_cost_function(c_obj)
        @test f1 != f
        @test f1(M, p) == f(M, p)
        @test get_count(c_obj, :Cost) == 1 # still counted
        # The same for gradient
        @test Manopt.get_gradient_function(obj) ===
            Manopt.get_gradient_function(c_obj, true)
        grad_f1 = Manopt.get_gradient_function(c_obj)
        @test grad_f1 != grad_f
        @test grad_f1(M, p) == grad_f(M, p)
        @test get_count(c_obj, :Gradient) == 1 # still counted
        # And Hessian
        @test Manopt.get_hessian_function(obj) === Manopt.get_hessian_function(c_obj, true)
        Hess_f1 = Manopt.get_hessian_function(c_obj)
        @test Hess_f1 != Hess_f
        @test Hess_f1(M, p, X) == Hess_f(M, p, X)
        @test get_count(c_obj, :Hessian) == 1 # still counted
        #
        # And all three for mutating again
        grad_f!(M, X, p) = (X .= A * p - (p' * A * p) * p)
        Hess_f!(M, Y, p, X) = (Y .= A * X - (p' * A * X) .* p - (p' * A * p) .* X)
        obj_i = ManifoldHessianObjective(
            f, grad_f!, Hess_f!; evaluation = InplaceEvaluation()
        )
        c_obj_i = ManifoldCountObjective(M, obj_i, [:Cost, :Gradient, :Hessian])
        @test Manopt.get_cost_function(obj_i) === Manopt.get_cost_function(c_obj_i, true)
        f2 = Manopt.get_cost_function(c_obj_i)
        @test f2 != f
        @test f2(M, p) == f(M, p)
        @test get_count(c_obj_i, :Cost) == 1 # still counted
        # The same for gradient
        @test Manopt.get_gradient_function(obj_i) ===
            Manopt.get_gradient_function(c_obj_i, true)
        grad_f1! = Manopt.get_gradient_function(c_obj_i; evaluation = InplaceEvaluation())
        @test grad_f1! != grad_f!
        Y = similar(X)
        Z = similar(X)
        @test grad_f1!(M, Y, p) == grad_f!(M, Z, p)
        @test get_count(c_obj_i, :Gradient) == 1 # still counted
        # And Hessian
        @test Manopt.get_hessian_function(obj_i) ===
            Manopt.get_hessian_function(c_obj_i, true)
        Hess_f1! = Manopt.get_hessian_function(c_obj_i; evaluation = InplaceEvaluation())
        @test Hess_f1! != Hess_f!
        @test Hess_f1!(M, Y, p, X) == Hess_f!(M, Z, p, X)
        @test get_count(c_obj_i, :Hessian) == 1 # still counted
    end
end
