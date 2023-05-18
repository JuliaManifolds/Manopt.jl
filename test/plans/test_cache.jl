using LinearAlgebra, LRUCache, Manifolds, Manopt, Test

include("../utils/dummy_types.jl")

# Three dummy functors that are just meant to cound their calls
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
    return p
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
        mcgoa = ManifoldGradientObjective(TestCostCount(0), TestGradCount(0))
        sco1 = Manopt.SimpleManifoldCachedObjective(M, mgoa; p=p)
        # We evaluated on init -> 1
        @test get_gradient_function(sco1).i == 1
        @test get_cost_function(sco1).i == 1
        @test get_gradient(M, sco1, p) == p
        get_gradient!(M, X, sco1, p)
        @test X == zero_vector(M, p)
        @test get_cost(M, sco1, p) == norm(p)
        # stil at 1
        @test get_gradient_function(sco1).i == 1
        @test get_cost_function(sco1).i == 1
        @test get_gradient(M, sco1, q) == q # triggers an evaluation
        get_gradient!(M, X, sco1, q) # same point, copies
        @test X == q
        @test get_cost(M, sco1, q) == norm(q)
        @test get_cost_function(sco1).i == 2
        @test get_gradient_function(sco1).i == 2
        # first grad!
        get_gradient!(M, X, sco1, r) # triggers an evaluation
        @test get_gradient(M, sco1, r) == X # cached
        @test X == r
        @test get_gradient_function(sco1).i == 3

        mgoi = ManifoldGradientObjective(
            TestCostCount(0), TestGradCount(0); evaluation=InplaceEvaluation()
        )
        sco2 = Manopt.SimpleManifoldCachedObjective(M, mgoi; p=p, initialized=false)
        # We did not evaluate on init -> 1st eval
        @test get_gradient_function(sco2).i == 0
        @test get_cost_function(sco2).i == 0
        @test get_gradient(M, sco2, p) == p
        @test get_cost(M, sco2, p) == norm(p)
        # now 1
        @test get_gradient_function(sco2).i == 1
        @test get_cost_function(sco2).i == 1
        # new point -> 2
        @test get_gradient(M, sco2, q) == q
        get_gradient!(M, X, sco2, q) # cached
        @test X == q
        @test get_cost(M, sco2, q) == norm(q)
        @test get_gradient_function(sco2).i == 2
        @test get_cost_function(sco2).i == 2
        get_gradient!(M, X, sco2, r)
        @test get_gradient(M, sco2, r) == X # cached
        @test X == r

        mcgoa = ManifoldCostGradientObjective(TestCostGradCount(0))
        sco3 = Manopt.SimpleManifoldCachedObjective(M, mcgoa; p=p, initialized=false)
        # We do not evaluate on init -> still zero
        @test sco3.objective.costgrad!!.i == 0
        @test get_gradient(M, sco3, p) == p
        get_gradient!(M, X, sco3, p)
        @test X == p
        @test get_cost(M, sco3, p) == norm(p)
        # stil at 1
        @test sco3.objective.costgrad!!.i == 1
        @test get_gradient(M, sco3, q) == q
        get_gradient!(M, X, sco3, q) # cached
        @test X == q
        @test get_cost(M, sco3, q) == norm(q) # cached
        @test sco3.objective.costgrad!!.i == 2
        get_gradient!(M, X, sco3, r)
        @test X == r
        @test get_gradient(M, sco3, r) == r # cached
        @test get_cost(M, sco3, r) == norm(r) # cached
        @test sco3.objective.costgrad!!.i == 3
        @test get_cost(M, sco3, s) == norm(s)
        get_gradient!(M, X, sco3, s) # cached
        @test X == s
        @test get_gradient(M, sco3, s) == s # cached
        @test sco3.objective.costgrad!!.i == 4

        mcgoi = ManifoldCostGradientObjective(
            TestCostGradCount(0); evaluation=InplaceEvaluation()
        )
        sco4 = Manopt.SimpleManifoldCachedObjective(M, mcgoi; p=p)
        # We evaluated on init -> evaluates twice
        @test sco4.objective.costgrad!!.i == 2
        @test get_gradient(M, sco4, p) == p
        get_gradient!(M, X, sco4, p) # cached
        @test X == p
        @test get_cost(M, sco4, p) == norm(p)
        # stil at 2
        @test sco4.objective.costgrad!!.i == 2
        @test get_gradient(M, sco4, q) == q
        get_gradient!(M, X, sco4, q) #cached
        @test X == q
        @test get_cost(M, sco4, q) == norm(q)
        @test sco4.objective.costgrad!!.i == 3
        get_gradient!(M, X, sco4, r)
        @test X == r
        @test get_gradient(M, sco4, r) == r # cached
        @test sco4.objective.costgrad!!.i == 4
        @test get_cost(M, sco4, s) == norm(s)
        get_gradient!(M, X, sco4, s) # cached
        @test X == s
        @test get_gradient(M, sco4, s) == s # cached
        @test sco4.objective.costgrad!!.i == 5
    end
    @testset "ManifoldCachedObjective on Cost&Grad" begin
        M = Sphere(2)
        A = [2.0 1.0 0.0; 1.0 2.0 1.0; 0.0 1.0 2.0]
        f(M, p) = p' * A * p
        grad_f(M, p) = 2 * A * p
        o = ManifoldGradientObjective(f, grad_f)
        co = ManifoldCountObjective(M, o, [:Cost, :Gradient])
        lco = objective_cache_factory(M, co, (:LRU, [:Cost, :Gradient]))
        ro = DummyDecoratedObjective(o)
        #indecorated works as well
        lco2 = objective_cache_factory(M, o, (:LRU, [:Cost, :Gradient]))
        p = [1.0, 0.0, 0.0]
        a = get_count(lco, :Cost) # usually 1 since creating lco calls that once
        @test get_cost(M, lco, p) == 2.0
        @test get_cost(M, lco, p) == 2.0
        # but the second was cached so no cost eval
        @test get_count(lco, :Cost) == a + 1
        # Gradient
        b = get_count(lco, :Gradient)
        X = get_gradient(M, lco, p)
        @test X == grad_f(M, p)
        Y = similar(X)
        #Update Y inplace but without evaluating the gradient but taking it from the cache
        get_gradient!(M, Y, lco, p)
        @test Y == X
        @test get_count(lco, :Gradient) == b + 1
        #
        # CostGrad
        f_f_grad(M, p) = (p' * A * p, 2 * A * p)
        f_f_grad!(M, X, p) = (p' * A * p, X .= 2 * A * p)
        o2a = ManifoldCostGradientObjective(f_f_grad)
        co2a = ManifoldCountObjective(M, o2a, [:Cost, :Gradient])
        #pass size
        lco2a = objective_cache_factory(M, co2a, (:LRU, [:Cost, :Gradient], 10))
        o2i = ManifoldCostGradientObjective(f_f_grad!; evaluation=InplaceEvaluation())
        co2i = ManifoldCountObjective(M, o2i, [:Cost, :Gradient])
        # pass keyword
        lco2i = objective_cache_factory(
            M, co2i, (:LRU, [:Cost, :Gradient], [:cache_size => 10])
        )
        #
        c = get_count(lco2a, :Cost) # usually 1 since creating lco calls that once
        @test get_cost(M, lco2a, p) == 2.0
        @test get_cost(M, lco2a, p) == 2.0
        # but the second was cached so no cost eval
        @test get_count(lco2a, :Cost) == c + 1
        d = get_count(lco2a, :Gradient)
        X = get_gradient(M, lco2a, p)
        @test X == f_f_grad(M, p)[2]
        Y = similar(X)
        #Update Y inplace but without evaluating the gradient but taking it from the cache
        get_gradient!(M, Y, lco, p)
        @test Y == X
        get_gradient!(M, Y, lco, -p) #trigger cache with in-place
        @test Y == -X
        # Similar with
        # Gradient cached already so no new evals
        @test get_count(lco2a, :Gradient) == d
        # Trigger caching on costgrad
        X = get_gradient(M, lco2a, -p)
        @test X == Y
        # Trigger caching on costgrad!
        get_gradient!(M, X, lco2i, -p)
        @test X == Y
        # Check default trigger
        @test_throws DomainError Manopt.init_caches(M, [:Cost], Nothing)
    end
end
