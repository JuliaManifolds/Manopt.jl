using LRUCache, Manifolds, Manopt, Test, Random

@testset "Hessian access functions" begin
    M = Euclidean(2)
    f(M, p) = 1
    grad_f(M, p) = zeros(2)
    grad_f!(M, X, p) = copyto!(M, X, p, zeros(2))
    Hess_f(M, p, X) = 0.5 * X
    Hess_f!(M, Y, p, X) = copyto!(M, Y, p, 0.5 * X)
    precon(M, p, X) = X
    precon!(M, Y, p, X) = copyto!(M, Y, p, X)

    mho1 = ManifoldHessianObjective(f, grad_f, Hess_f)
    mho2 = ManifoldHessianObjective(f, grad_f, Hess_f, precon)
    mho3 = ManifoldHessianObjective(f, grad_f!, Hess_f!; evaluation = InplaceEvaluation())
    mho4 = ManifoldHessianObjective(f, grad_f, Hess_f)

    p = zeros(2)
    X = ones(2)

    for mho in [mho1, mho2, mho3, mho4]
        mp = DefaultManoptProblem(M, mho)
        Y = similar(X)
        # Gradient
        @test get_gradient(mp, p) == zeros(2)
        get_gradient!(mp, Y, p)
        @test Y == zeros(2)
        # check differential default
        @test get_differential(mp, p, X; gradient = Y) == 0
        @test get_differential(mp, p, X) == 0
        # Hessian
        @test get_hessian(mp, p, X) == 0.5 * X
        get_hessian!(mp, Y, p, X)
        @test Y == 0.5 * X
        # precondition
        @test get_preconditioner(mp, p, X) == X
        get_preconditioner!(mp, Y, p, X)
        @test Y == X
    end
    @testset "Objective Decorator passthrough" begin
        Y1 = zero_vector(M, p)
        Y2 = zero_vector(M, p)
        for obj in [mho1, mho2, mho3, mho4]
            ddo = Manopt.Test.DummyDecoratedObjective(obj)
            @test get_hessian(M, obj, p, X) == get_hessian(M, ddo, p, X)
            get_hessian!(M, Y1, obj, p, X)
            get_hessian!(M, Y2, ddo, p, X)
            @test Y1 == Y2
            @test get_preconditioner(M, obj, p, X) == get_preconditioner(M, ddo, p, X)
            get_preconditioner!(M, Y1, obj, p, X)
            get_preconditioner!(M, Y2, ddo, p, X)
            @test Y1 == Y2
            @test Manopt.get_hessian_function(ddo) == Manopt.get_hessian_function(obj)
        end
    end
    @testset "Counting Objective" begin
        Y1 = zero_vector(M, p)
        Y2 = zero_vector(M, p)
        for obj in [mho1, mho2, mho3, mho4]
            cobj = Manopt.objective_count_factory(M, obj, [:Hessian, :Preconditioner])
            ddo = Manopt.Test.DummyDecoratedObjective(obj)
            @test get_hessian(M, obj, p, X) == get_hessian(M, cobj, p, X)
            get_hessian!(M, Y1, obj, p, X)
            get_hessian!(M, Y2, cobj, p, X)
            @test Y1 == Y2
            @test get_preconditioner(M, obj, p, X) == get_preconditioner(M, cobj, p, X)
            get_preconditioner!(M, Y1, obj, p, X)
            get_preconditioner!(M, Y2, cobj, p, X)
            @test Y1 == Y2
            @test get_count(cobj, :Hessian) == 2
            @test get_count(cobj, :Preconditioner) == 2
        end
    end
    @testset "LRU Cache Objective" begin
        Y = zero_vector(M, p)
        for obj in [mho1, mho2, mho3, mho4]
            cobj = Manopt.objective_count_factory(M, obj, [:Hessian, :Preconditioner])
            ccobj = Manopt.objective_cache_factory(
                M, cobj, (:LRU, [:Hessian, :Preconditioner])
            )
            Z = get_hessian(M, obj, p, X)
            @test get_hessian(M, ccobj, p, X) == Z
            @test get_hessian(M, ccobj, p, X) == Z #cached
            get_hessian!(M, Y, ccobj, p, X) #cached
            @test Y == Z
            @test get_count(ccobj, :Hessian) == 1
            Z = get_hessian(M, obj, -p, -X)
            get_hessian!(M, Y, ccobj, -p, -X) #cached
            @test Y == Z
            @test get_hessian(M, ccobj, -p, -X) == Z #cached
            @test get_count(ccobj, :Hessian) == 2

            Z = get_preconditioner(M, obj, p, X)
            @test get_preconditioner(M, ccobj, p, X) == Z
            @test get_preconditioner(M, ccobj, p, X) == Z #cached
            get_preconditioner!(M, Y, ccobj, p, X) #cached
            @test Y == Z
            @test get_count(ccobj, :Preconditioner) == 1
            Z = get_preconditioner(M, obj, -p, -X)
            get_preconditioner!(M, Y, ccobj, -p, -X)
            @test Y == Z
            get_preconditioner!(M, Y, ccobj, -p, -X) # Cached
            @test Y == Z
            @test get_preconditioner(M, ccobj, -p, -X) == Z #cached
            @test get_preconditioner(M, ccobj, -p, -X) == Z #cached
            @test get_count(ccobj, :Preconditioner) == 2
        end
    end
    @testset "Fixedrank – nonmatrix point/vector types" begin
        m = 5
        n = 3
        k = 2
        M = FixedRankMatrices(m, n, k)
        L = randn(m, k)
        R = randn(n, k)
        A = L * R'
        # Generate a mask
        P = zeros(m, n)
        P[1:9] .= 1
        PA = P .* A
        f2(M, p) = 0.5 * norm(P .* embed(M, p) - PA)^2
        # Project converts the Gradient in the Embedding to an fixed rank matrices vector
        grad_f2(M, p) = project(M, p, P .* embed(M, p) - PA)
        grad_f2!(M, X, p) = project!(M, X, p, P .* embed(M, p) - PA)
        Random.seed!(42)
        p0 = rand(M)
        q1 = trust_regions(M, f2, grad_f2, p0)
        q2 = trust_regions(M, f2, grad_f2!, p0; evaluation = InplaceEvaluation())
        @test isapprox(M, q1, q2)
    end
end
