s = joinpath(@__DIR__, "..", "ManoptTestSuite.jl")
!(s in LOAD_PATH) && (push!(LOAD_PATH, s))

using Manifolds, Manopt, ManoptTestSuite, Test, Random
using Manopt: get_cost_function, get_gradient_function
using LinearAlgebra: Symmetric

@testset "Counting Objective test" begin
    @testset "Basics" begin
        M = Sphere(2)
        A = [2.0 1.0 0.0; 1.0 2.0 1.0; 0.0 1.0 2.0]
        f(M, p) = p' * A * p
        grad_f(M, p) = project(M, p, 2 * A * p)
        obj = ManifoldFirstOrderObjective(f, grad_f)
        c_obj = ManifoldCountObjective(M, obj, [:Cost, :Gradient])
        # function access functions are different since the right is still counting.
        @test get_cost_function(obj) != get_cost_function(c_obj)
        @test get_gradient_function(obj) != get_gradient_function(c_obj)
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
        # also decorated objects can be wrapped to be counted
        ro = ManoptTestSuite.DummyDecoratedObjective(obj)
        c_obj2 = ManifoldCountObjective(M, ro, [:Gradient])
        get_gradient(M, c_obj2, p)
        @test get_count(c_obj2, :Gradient) == 1
        @test_throws ErrorException get_count(ro, :Cost) # Errors since no CountObj
        @test get_count(c_obj2, :Cost) == -1 # Does not count cost
        @test_throws ErrorException get_count(c_obj2, :Cost, :error)
        @test startswith(repr(c_obj), "## Statistics")
        @test startswith(Manopt.status_summary(c_obj), "## Statistics")
        # also for the `repr` call
        @test startswith(repr((c_obj, p)), "## Statistics")
        # but this also includes the hint, how to access the result
        @test endswith(repr((c_obj, p)), "on this variable.")
        rc_obj = ManoptTestSuite.DummyDecoratedObjective(c_obj)
        @test get_count(rc_obj, :Gradient) == 2 #still works if count is encapsulated
        @test_throws ErrorException get_count(obj, :Gradient) # no count objective
        @test get_count(rc_obj, :Gradient, 1) == 2 #still works if count is encapsulated
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
    @testset "Function passthrough" begin
        Random.seed!(42)
        n = 4
        A = Symmetric(randn(n, n))
        M = Sphere(n - 1)
        p = [1.0, zeros(n - 1)...]
        X = [0.0, 1.0, zeros(n - 2)...]
        f(M, p) = 0.5 * p' * A * p
        grad_f(M, p) = A * p - (p' * A * p) * p
        Hess_f(M, p, X) = A * X + (p' * A * X) .* p + (p' * A * p) .* X
        obj = ManifoldHessianObjective(f, grad_f, Hess_f)
        c_obj = ManifoldCountObjective(M, obj, [:Cost, :Gradient, :Hessian])
        # undecorated / recursive cost -> exactly f
        @test Manopt.get_cost_function(obj) === Manopt.get_cost_function(c_obj, true)
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
        Hess_f!(M, Y, p, X) = (Y .= A * X + (p' * A * X) .* p + (p' * A * p) .* X)
        obj_i = ManifoldHessianObjective(
            f, grad_f!, Hess_f!; evaluation=InplaceEvaluation()
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
        grad_f1! = Manopt.get_gradient_function(c_obj_i)
        @test grad_f1! != grad_f!
        Y = similar(X)
        Z = similar(X)
        @test grad_f1!(M, Y, p) == grad_f!(M, Z, p)
        @test get_count(c_obj_i, :Gradient) == 1 # still counted
        # And Hessian
        @test Manopt.get_hessian_function(obj_i) ===
            Manopt.get_hessian_function(c_obj_i, true)
        Hess_f1! = Manopt.get_hessian_function(c_obj_i)
        @test Hess_f1 != Hess_f
        @test Hess_f1!(M, Y, p, X) == Hess_f!(M, Z, p, X)
        @test get_count(c_obj_i, :Hessian) == 1 # still counted
    end
end
