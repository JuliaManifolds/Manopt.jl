using LinearAlgebra, Manifolds, Manopt, Test, Random

@testset "Scaled Objective" begin
    n = 4
    Random.seed!(42)
    A = Symmetric(randn(n, n))
    f(M, p) = 0.5 * p' * A * p
    ∇f(M, p) = A * p
    ∇²f(M, p, X) = A * X
    ∇f!(M, X, p) = (X .= A * p)
    ∇²f!(M, Y, p, X) = (Y .= A * X)
    M = Euclidean(n)
    p = [1.0, zeros(n - 1)...]
    X = [0.0, 1.0, zeros(n - 2)...]
    obj = ManifoldHessianObjective(f, ∇f, ∇²f)
    obj! = ManifoldHessianObjective(f, ∇f!, ∇²f!; evaluation = InplaceEvaluation())
    neg_obj = -obj
    @test neg_obj isa ScaledManifoldObjective
    s = repr(neg_obj)
    @test startswith(s, "ScaledManifoldObjective(ManifoldHessianObjective(f, ∇f")
    @test endswith(s, "-1)")
    @test repr(neg_obj) == s
    scaled_obj = -1 * obj
    @test scaled_obj == neg_obj
    scaled_obj! = -1.0 * obj!
    # just verify that this also works for double decorated ones.
    deco_obj = ScaledManifoldObjective(ManifoldCountObjective(M, obj, [:Cost]), 0.5)

    #
    # Test and compare all accessors
    #
    for (s, o) in zip([scaled_obj, scaled_obj!], [obj, obj!])
        @test get_cost(M, s, p) == -f(M, p)
        @test get_gradient(M, s, p) == -∇f(M, p)
        Y = zero_vector(M, p)
        get_gradient!(M, Y, s, p)
        @test Y == -∇f(M, p)
        @test get_hessian(M, s, p, X) == -∇²f(M, p, X)
        get_hessian!(M, Y, s, p, X)
        @test Y == -∇²f(M, p, X)

        # Function accessors
        @test Manopt.get_cost_function(o) === Manopt.get_cost_function(s, true)
        f2 = Manopt.get_cost_function(s)
        @test f2 != f
        @test f2(M, p) == -f(M, p)
        # The same for gradient
        @test Manopt.get_gradient_function(o) === Manopt.get_gradient_function(s, true)
    end
    # Special cases for alloc and inplace
    grad_f1 = Manopt.get_gradient_function(scaled_obj)
    @test grad_f1 != ∇f
    @test grad_f1(M, p) == -∇f(M, p)
    Hess_f1 = Manopt.get_hessian_function(scaled_obj)
    @test Hess_f1 != ∇²f
    @test Hess_f1(M, p, X) == -∇²f(M, p, X)
    # Special cases for alloc and inplace
    Y = similar(X)
    Z = similar(X)
    grad_f1! = Manopt.get_gradient_function(scaled_obj!)
    @test grad_f1 != ∇f!
    @test grad_f1!(M, Y, p) == -∇f!(M, Z, p)
    @test Y == -Z

    Hess_f1 = Manopt.get_hessian_function(scaled_obj)
    @test Hess_f1 != ∇²f
    @test Hess_f1(M, p, X) == -∇²f(M, p, X)
    Hess_f1! = Manopt.get_hessian_function(scaled_obj!)
    @test Hess_f1 != ∇²f!
    @test Hess_f1!(M, Y, p, X) == -∇²f!(M, Z, p, X)
    @test Y == -Z
end
