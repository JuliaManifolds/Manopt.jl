using LinearAlgebra, LRUCache, Manifolds, Manopt, Test, Random

# Three dummy functors that are just meant to count their calls
mutable struct TestCostCount
    i::Int
end
TestCostCount() = TestCostCount(0)
function (tcc::TestCostCount)(M, p)
    tcc.i += 1
    return norm(p)
end
mutable struct TestGradCount
    i::Int
end
TestGradCount() = TestGradCount(0)
function (tgc::TestGradCount)(M, p)
    tgc.i += 1
    return copy(p) # for cache, copy
end
function (tgc::TestGradCount)(M, X, p)
    tgc.i += 1
    X .= p
    return X
end
mutable struct TestCostGradCount
    i::Int
end
TestCostGradCount() = TestCostGradCount(0)
function (tcgc::TestCostGradCount)(M, p)
    tcgc.i += 1
    return norm(p), p
end
function (tcgc::TestCostGradCount)(M, X, p)
    tcgc.i += 1
    X .= p
    return norm(p), X
end

@testset "Test Caches" begin
    @testset "Test Factory" begin
        M = Euclidean(3)
        # allocating
        mgoa = ManifoldGradientObjective(TestCostCount(0), TestGradCount(0))
        s1 = objective_cache_factory(M, mgoa, :Simple)
        @test s1 isa SimpleManifoldCachedObjective
        @test objective_cache_factory(M, mgoa, :none) == mgoa
        # pass a keyword
        s2 = objective_cache_factory(M, mgoa, (:Simple, [], [:initialized => false]))
        @test s2 isa SimpleManifoldCachedObjective
        @test Manopt.is_objective_decorator(s2)
        # but not initialized
        @test !s2.X_valid
        @test !s2.c_valid
        # test fallbacks that do not decorate
        @test objective_cache_factory(M, mgoa, :none) == mgoa
        @test objective_cache_factory(M, mgoa, (:none, [])) == mgoa
        @test objective_cache_factory(M, mgoa, (:none, [], [])) == mgoa
    end
    @testset "SimpleManifoldCachedObjective" begin
        M = Euclidean(3)
        p = zeros(3)
        q = ones(3)
        r = 2 * ones(3)
        s = 3 * ones(3)
        X = zero_vector(M, p)
        # allocating
        mgoa = ManifoldGradientObjective(TestCostCount(0), TestGradCount(0))
        # Init to copy of p - init cache
        sco1 = Manopt.SimpleManifoldCachedObjective(M, mgoa; p = copy(M, p))
        @test repr(sco1) == "SimpleManifoldCachedObjective{AllocatingEvaluation,$(mgoa)}"
        @test startswith(
            repr((sco1, 1.0)),
            """## Cache
            A `SimpleManifoldCachedObjective`""",
        )
        @test startswith(
            repr((sco1, Manopt.Test.DummyState())),
            """Manopt.Test.DummyState(Float64[])

            ## Cache
            A `SimpleManifoldCachedObjective`""",
        )
        # evaluated on init -> 1
        @test sco1.objective.functions[:cost].i == 1
        @test sco1.objective.functions[:gradient].i == 1
        @test get_gradient(M, sco1, p) == p
        get_gradient!(M, X, sco1, p)
        @test X == zero_vector(M, p)
        @test get_cost(M, sco1, p) == norm(p)
        # still at 1
        @test sco1.objective.functions[:cost].i == 1
        @test sco1.objective.functions[:gradient].i == 1
        @test get_gradient(M, sco1, q) == q # triggers an evaluation
        get_gradient!(M, X, sco1, q) # same point, copies
        @test X == q
        @test get_cost(M, sco1, q) == norm(q)
        @test sco1.objective.functions[:cost].i == 2
        @test sco1.objective.functions[:gradient].i == 2
        # first `grad!`
        get_gradient!(M, X, sco1, r) # triggers an evaluation
        @test get_gradient(M, sco1, r) == X # cached
        @test X == r
        @test sco1.objective.functions[:gradient].i == 3
        @test Manopt.get_cost_function(sco1) != Manopt.get_cost_function(mgoa)
        @test Manopt.get_gradient_function(sco1) != Manopt.get_gradient_function(mgoa)
        # test cost_grad, back to q to trigger first
        c, X = Manopt.get_cost_and_gradient(M, sco1, q) # trigger
        @test X == q
        @test c == norm(q)
        @test sco1.objective.functions[:cost].i == 3
        @test sco1.objective.functions[:gradient].i == 4
        c, _ = Manopt.get_cost_and_gradient!(M, X, sco1, q) # cached
        @test X == q
        @test c == norm(q)
        @test sco1.objective.functions[:cost].i == 3
        @test sco1.objective.functions[:gradient].i == 4
        # Diff via grad - with caching (since not recursive)
        df = Manopt.get_differential_function(sco1)
        d = df(M, r, X) #norm <r, X> since grad is r, triggers, but does not cache grad
        @test d == dot(X, r)
        @test sco1.objective.functions[:gradient].i == 5
        get_gradient(M, sco1, r)
        @test sco1.objective.functions[:gradient].i == 6
        d = get_differential(M, sco1, r, X) # ...so that this is cached
        @test d == dot(X, r)
        @test sco1.objective.functions[:gradient].i == 6

        mgoi = ManifoldGradientObjective(
            TestCostCount(0), TestGradCount(0); evaluation = InplaceEvaluation()
        )
        sco2 = Manopt.SimpleManifoldCachedObjective(M, mgoi; initialized = false)
        # not evaluated on init -> this is the first
        @test sco2.objective.functions[:cost].i == 0
        @test sco2.objective.functions[:gradient].i == 0
        @test get_gradient(M, sco2, p) == p
        @test get_cost(M, sco2, p) == norm(p)
        # both evaluated once
        @test sco2.objective.functions[:cost].i == 1
        @test sco2.objective.functions[:gradient].i == 1
        # new point -> 2
        @test get_gradient(M, sco2, q) == q
        get_gradient!(M, X, sco2, q) # cached
        @test X == q
        @test get_cost(M, sco2, q) == norm(q)
        @test sco2.objective.functions[:cost].i == 2
        @test sco2.objective.functions[:gradient].i == 2
        # Just gradient anew
        get_gradient!(M, X, sco2, r)
        @test sco2.objective.functions[:gradient].i == 3
        @test get_gradient(M, sco2, r) == X # cached
        @test sco2.objective.functions[:gradient].i == 3
        @test X == r
        # Costgrad, here first inplace
        c, _ = Manopt.get_cost_and_gradient!(M, X, sco2, q) # trigger
        @test X == q
        @test c == norm(q)
        @test sco2.objective.functions[:cost].i == 3
        @test sco2.objective.functions[:gradient].i == 4
        c, X = Manopt.get_cost_and_gradient(M, sco2, q) # cached
        @test X == q
        @test c == norm(q)
        @test sco2.objective.functions[:cost].i == 3
        @test sco2.objective.functions[:gradient].i == 4

        mcgoa = ManifoldCostGradientObjective(TestCostGradCount(0))
        sco3 = Manopt.SimpleManifoldCachedObjective(M, mcgoa; initialized = false)
        # not evaluated on init -> still zero
        @test sco3.objective.functions[:costgradient].i == 0
        @test get_gradient(M, sco3, p) == p
        get_gradient!(M, X, sco3, p)
        @test X == p
        @test get_cost(M, sco3, p) == norm(p)
        # for seperate calls this is a 2

        @test sco3.objective.functions[:costgradient].i == 2
        @test get_gradient(M, sco3, q) == q
        get_gradient!(M, X, sco3, q) # cached
        @test X == q
        @test get_cost(M, sco3, q) == norm(q)
        @test sco3.objective.functions[:costgradient].i == 4
        get_gradient!(M, X, sco3, r)
        @test X == r
        @test get_gradient(M, sco3, r) == r # cached
        @test get_cost(M, sco3, r) == norm(r)
        @test sco3.objective.functions[:costgradient].i == 6
        @test get_cost(M, sco3, s) == norm(s)
        get_gradient!(M, X, sco3, s)
        @test X == s
        @test get_gradient(M, sco3, s) == s # cached
        @test sco3.objective.functions[:costgradient].i == 8

        mcgoi = ManifoldCostGradientObjective(
            TestCostGradCount(0); evaluation = InplaceEvaluation()
        )
        sco4 = Manopt.SimpleManifoldCachedObjective(M, mcgoi; p = p)
        # evaluated on init -> evaluates twice
        @test sco4.objective.functions[:costgradient].i == 2
        @test get_gradient(M, sco4, p) == p
        get_gradient!(M, X, sco4, p) # cached
        @test X == p
        @test get_cost(M, sco4, p) == norm(p)
        # still at 2
        @test sco4.objective.functions[:costgradient].i == 2
        @test get_gradient(M, sco4, q) == q
        get_gradient!(M, X, sco4, q) #cached
        @test X == q
        @test get_cost(M, sco4, q) == norm(q)
        @test sco4.objective.functions[:costgradient].i == 4
        get_gradient!(M, X, sco4, r)
        @test X == r
        @test get_gradient(M, sco4, r) == r # cached
        @test sco4.objective.functions[:costgradient].i == 5
        @test get_cost(M, sco4, s) == norm(s)
        get_gradient!(M, X, sco4, s)
        @test X == s
        @test get_gradient(M, sco4, s) == s # cached
        @test sco4.objective.functions[:costgradient].i == 7
    end
    @testset "ManifoldCachedObjective on Cost&Grad" begin
        M = Sphere(2)
        A = [2.0 1.0 0.0; 1.0 2.0 1.0; 0.0 1.0 2.0]
        f(M, p) = p' * A * p
        grad_f(M, p) = 2 * A * p
        o = ManifoldGradientObjective(f, grad_f)
        co = ManifoldCountObjective(M, o, [:Cost, :Gradient, :Differential])
        lco = objective_cache_factory(M, co, (:LRU, [:Cost, :Gradient, :Differential]))
        @test startswith(repr(lco), "## Cache\n  * ")
        @test startswith(
            repr((lco, Manopt.Test.DummyState())),
            "Manopt.Test.DummyState(Float64[])\n\n## Cache\n  * ",
        )
        ro = Manopt.Test.DummyDecoratedObjective(o)
        #undecorated works as well
        lco2 = objective_cache_factory(M, o, (:LRU, [:Cost, :Gradient]))
        @test Manopt.get_cost_function(lco2) != Manopt.get_cost_function(o)
        @test Manopt.get_gradient_function(lco2) != Manopt.get_gradient_function(o)
        @test Manopt.get_differential_function(lco2) != Manopt.get_differential_function(o)
        p = [1.0, 0.0, 0.0]
        a = get_count(lco, :Cost) # usually 1 since creating `lco` calls that once
        @test get_cost(M, lco, p) == 2.0
        @test get_cost(M, lco, p) == 2.0
        # but the second was cached so no cost `eval`
        @test get_count(lco, :Cost) == a + 1
        # Gradient
        b = get_count(lco, :Gradient)
        X = get_gradient(M, lco, p)
        @test X == grad_f(M, p)
        # make sure this is safe, by modifying X
        X .= [1.0, 0.0, 1.0]
        # does not affect the cache
        @test get_gradient(M, lco, p) == grad_f(M, p)
        X = get_gradient(M, lco, p) # restore X
        Y = similar(X)
        #Update Y in-place but without evaluating the gradient but taking it from the cache
        get_gradient!(M, Y, lco, p)
        @test Y == X
        @test get_count(lco, :Gradient) == b + 1
        # Differential
        c = get_count(lco, :Differential)
        @test get_differential(M, lco, p, X) == inner(M, p, X, Y)
        @test get_count(lco, :Differential) == c + 1
        d = get_differential(M, lco, p, X) # cached
        @test get_count(lco, :Differential) == c + 1
        # A second point to check cost grad cache
        # Staying at p eval cost_grad comes at no cost.
        a2 = get_count(lco, :Cost)
        b2 = get_count(lco, :Gradient)
        c, X = Manopt.get_cost_and_gradient(M, lco, p)
        @test c == 2.0
        @test X == Y
        c, _ = Manopt.get_cost_and_gradient!(M, X, lco, p)
        @test c == 2.0
        @test X == Y
        @test get_count(lco, :Cost) == a2
        @test get_count(lco, :Gradient) == b2
        q = p .+ 1
        c2 = get_cost(M, o, q)
        X2 = get_gradient(M, o, q)
        c, X = Manopt.get_cost_and_gradient(M, lco, q) #miss
        @test c == c2
        @test X == X2
        c, _ = Manopt.get_cost_and_gradient!(M, X, lco, q) # cached
        @test c == c2
        @test X == X2
        # one of these was cached
        @test get_count(lco, :Cost) == a2 + 1
        @test get_count(lco, :Gradient) == b2 + 1
        # yet again the other way around
        q = q .+ 1
        c2 = get_cost(M, o, q)
        X2 = get_gradient(M, o, q)
        c, _ = Manopt.get_cost_and_gradient!(M, X, lco, q) # miss
        @test c == c2
        @test X == X2
        c, X = Manopt.get_cost_and_gradient(M, lco, q) # cached
        @test c == c2
        @test X == X2
        # one of these was cached
        @test get_count(lco, :Cost) == a2 + 2
        @test get_count(lco, :Gradient) == b2 + 2

        #
        # CostGrad
        f_f_grad(M, p) = (p' * A * p, 2 * A * p)
        f_f_grad!(M, X, p) = (p' * A * p, X .= 2 * A * p)
        o2a = ManifoldCostGradientObjective(f_f_grad)
        co2a = ManifoldCountObjective(M, o2a, [:Cost, :Gradient])
        #pass size
        lco2a = objective_cache_factory(M, co2a, (:LRU, [:Cost, :Gradient], 10))
        o2i = ManifoldCostGradientObjective(f_f_grad!; evaluation = InplaceEvaluation())
        co2i = ManifoldCountObjective(M, o2i, [:Cost, :Gradient])
        # pass keyword
        lco2i = objective_cache_factory(
            M, co2i, (:LRU, [:Cost, :Gradient], [:cache_size => 10])
        )
        #
        c = get_count(lco2a, :Cost) # usually 1 since creating `lco`` calls that once
        @test get_cost(M, lco2a, p) == 2.0
        @test get_cost(M, lco2a, p) == 2.0
        # but the second was cached so no cost evaluation
        @test get_count(lco2a, :Cost) == c + 1
        d = get_count(lco2a, :Gradient)
        X = get_gradient(M, lco2a, p)
        @test X == f_f_grad(M, p)[2]
        Y = similar(X)
        #Update Y in-place but without evaluating the gradient but taking it from the cache
        get_gradient!(M, Y, lco, p)
        @test Y == X
        # But is Y also fixed in there ? note that a reference to the cache was returned.
        Y .+= 1
        Z = similar(Y)
        get_gradient!(M, Z, lco, p)
        @test Z == X
        get_gradient!(M, Y, lco, -p) #trigger cache with in-place
        @test Y == -X
        # Similar with
        # Gradient not yet cached from cost (fornow) so one new evaluations
        @test get_count(lco2a, :Gradient) == d + 1
        # Trigger caching on `costgrad`
        X = get_gradient(M, lco2a, -p)
        @test X == Y
        # Trigger caching on `costgrad!`
        get_gradient!(M, X, lco2i, -p)
        @test X == Y
        # Check default trigger
        @test_throws DomainError Manopt.init_caches(M, [:Cost], Nothing)
        @test_throws ErrorException Manopt.init_caches(M, [:None], LRU)
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
        c_obj = objective_cache_factory(
            M, obj, (:LRU, [:Cost, :Gradient, :Hessian], [:cache_size => 1])
        )
        # undecorated / recursive cost -> exactly f
        @test Manopt.get_cost_function(obj) === Manopt.get_cost_function(c_obj, true)
        # otherwise different
        f1 = Manopt.get_cost_function(c_obj)
        @test f1 != f
        @test f1(M, p) == f(M, p)
        # The same for gradient
        @test Manopt.get_gradient_function(obj) ===
            Manopt.get_gradient_function(c_obj, true)
        grad_f1 = Manopt.get_gradient_function(c_obj)
        @test grad_f1 != grad_f
        @test grad_f1(M, p) == grad_f(M, p)
        # And Hessian
        @test Manopt.get_hessian_function(obj) === Manopt.get_hessian_function(c_obj, true)
        Hess_f1 = Manopt.get_hessian_function(c_obj)
        @test Hess_f1 != Hess_f
        @test Hess_f1(M, p, X) == Hess_f(M, p, X)
        #
        # And all three for mutating again
        grad_f!(M, X, p) = (X .= A * p - (p' * A * p) * p)
        Hess_f!(M, Y, p, X) = (Y .= A * X - (p' * A * X) .* p - (p' * A * p) .* X)
        obj_i = ManifoldHessianObjective(
            f, grad_f!, Hess_f!; evaluation = InplaceEvaluation()
        )
        c_obj_i = objective_cache_factory(
            M, obj_i, (:LRU, [:Cost, :Gradient, :Hessian], [:cache_size => 1])
        )
        @test Manopt.get_cost_function(obj_i) === Manopt.get_cost_function(c_obj_i, true)
        f2 = Manopt.get_cost_function(c_obj_i)
        @test f2 != f
        @test f2(M, p) == f(M, p)
        # The same for gradient
        @test Manopt.get_gradient_function(obj_i) ===
            Manopt.get_gradient_function(c_obj_i, true)
        grad_f1! = Manopt.get_gradient_function(c_obj_i)
        @test grad_f1! != grad_f!
        Y = similar(X)
        Z = similar(X)
        @test grad_f1!(M, Y, p) == grad_f!(M, Z, p)
        # And Hessian
        @test Manopt.get_hessian_function(obj_i) ===
            Manopt.get_hessian_function(c_obj_i, true)
        Hess_f1! = Manopt.get_hessian_function(c_obj_i)
        @test Hess_f1 != Hess_f
        @test Hess_f1!(M, Y, p, X) == Hess_f!(M, Z, p, X)
        #
        # Simple
        obj_g = ManifoldGradientObjective(f, grad_f)
        s_obj = Manopt.SimpleManifoldCachedObjective(M, obj_g; p = similar(p), X = similar(X))
        # undecorated / recursive cost -> exactly f
        @test Manopt.get_cost_function(obj_g) === Manopt.get_cost_function(s_obj, true)
        # otherwise different
        f1 = Manopt.get_cost_function(s_obj)
        @test f1 != f
        @test f1(M, p) == f(M, p)
        # The same for gradient
        @test Manopt.get_gradient_function(obj_g) ===
            Manopt.get_gradient_function(s_obj, true)
        grad_f1 = Manopt.get_gradient_function(s_obj)
        @test grad_f1 != grad_f
        @test grad_f1(M, p) == grad_f(M, p)
        # Simple Mutating
        obj_g_i = ManifoldGradientObjective(f, grad_f!; evaluation = InplaceEvaluation())
        s_obj_i = Manopt.SimpleManifoldCachedObjective(
            M, obj_g_i; p = similar(p), X = similar(X)
        )
        @test Manopt.get_cost_function(obj_g_i) === Manopt.get_cost_function(s_obj_i, true)
        f2 = Manopt.get_cost_function(s_obj_i)
        @test f2 != f
        @test f2(M, p) == f(M, p)
        # The same for gradient
        @test Manopt.get_gradient_function(obj_g_i) ===
            Manopt.get_gradient_function(s_obj_i, true)
        grad_f1! = Manopt.get_gradient_function(s_obj_i)
        @test grad_f1! != grad_f!
        Y = similar(X)
        Z = similar(X)
        @test grad_f1!(M, Y, p) == grad_f!(M, Z, p)
    end
    # Other tests are included with their respective objective tests in the corresponding plans
end
