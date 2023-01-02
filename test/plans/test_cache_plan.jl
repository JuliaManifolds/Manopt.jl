using LinearAlgebra, Manifolds, Manopt, Test

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
    @testset "SimpleCacheObjective" begin
        M = Euclidean(3)
        p = zeros(3)
        q = ones(3)
        r = 2 * ones(3)
        s = 3 * ones(3)
        X = zero_vector(M, p)
        # allocating
        mgoa = ManifoldGradientObjective(TestCostCount(0), TestGradCount(0))
        mcgoa = ManifoldGradientObjective(TestCostCount(0), TestGradCount(0))
        sco1 = Manopt.SimpleCacheObjective(M, mgoa; p=p)
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
        sco2 = Manopt.SimpleCacheObjective(M, mgoi; p=p, initialized=false)
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
        sco3 = Manopt.SimpleCacheObjective(M, mcgoa; p=p, initialized=false)
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
        sco4 = Manopt.SimpleCacheObjective(M, mcgoi; p=p)
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
end
