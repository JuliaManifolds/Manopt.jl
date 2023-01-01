using Manifolds, Manopt, Test

# Three dummy functors that are just meant to cound their calls
mutable struct TestCostCount
    i::Int
end
TestCostCount() = TestCostCount(0)
function (tcc::TestCostCount)(M, p)
    tcc.i += 1
    return 1.0
end
mutable struct TestGradCount
    i::Int
end
TestGradCount() = TestGradCount(0)
function (tgc::TestGradCount)(M, p)
    tgc.i += 1
    return zero_vector(M, p)
end
function (tgc::TestGradCount)(M, X, p)
    tgc.i += 1
    zero_vector!(M, X, p)
    return nothing
end
mutable struct TestCostGradCount
    i::Int
end
TestCostGradCount() = TestCostGradCount(0)
function (tcgc::TestCostGradCount)(M, p)
    tcgc.i += 1
    return 1.0, zero_vector(M, p)
end
function (tcgc::TestCostGradCount)(M, X, p)
    tcgc.i += 1
    zero_vector!(M, X, p)
    return 1.0, X
end

@testset "Test Caches" begin
    @testset "SimpleCacheObjective" begin
        M = Euclidean(3)
        p = zeros(3)
        q = ones(3)
        X = zero_vector(M, p)
        # allocating
        mgoa = ManifoldGradientObjective(TestCostCount(0), TestGradCount(0))
        mcgoa = ManifoldGradientObjective(TestCostCount(0), TestGradCount(0))
        sco1 = Manopt.SimpleCacheObjective(M, mgoa; p=p)
        # We evaluated on init -> 1
        @test get_gradient_function(sco1).i == 1
        @test get_cost_function(sco1).i == 1
        @test get_gradient(M, sco1, p) == zero_vector(M, p)
        get_gradient!(M, X, sco1, p)
        @test X == zero_vector(M, p)
        @test get_cost(M, sco1, p) == 1.0
        # stil at 1
        @test get_gradient_function(sco1).i == 1
        @test get_cost_function(sco1).i == 1
        @test get_gradient(M, sco1, q) == zero_vector(M, q)
        get_gradient!(M, X, sco1, q)
        @test X == zero_vector(M, q)
        @test get_cost(M, sco1, q) == 1.0
        @test get_gradient_function(sco1).i == 2
        @test get_cost_function(sco1).i == 2

        mgoi = ManifoldGradientObjective(
            TestCostCount(0), TestGradCount(0); evaluation=InplaceEvaluation()
        )
        sco2 = Manopt.SimpleCacheObjective(M, mgoi; p=p, initialized=false)
        # We did not evaluate on init -> ß
        @test get_gradient_function(sco2).i == 0
        @test get_cost_function(sco2).i == 0
        @test get_gradient(M, sco2, p) == zero_vector(M, p)
        @test get_cost(M, sco2, p) == 1.0
        # now 1
        @test get_gradient_function(sco2).i == 1
        @test get_cost_function(sco2).i == 1
        # new point -> 2
        @test get_gradient(M, sco2, q) == zero_vector(M, q)
        get_gradient!(M, X, sco2, q)
        @test X == zero_vector(M, q)
        @test get_cost(M, sco2, q) == 1.0
        @test get_gradient_function(sco2).i == 2
        @test get_cost_function(sco2).i == 2

        mcgoa = ManifoldCostGradientObjective(TestCostGradCount(0))
        sco3 = Manopt.SimpleCacheObjective(M, mcgoa; p=p, initialized=false)
        # We do not evaluate on init -> still zero
        @test sco3.objective.costgrad!!.i == 0
        @test get_gradient(M, sco3, p) == zero_vector(M, p)
        get_gradient!(M, X, sco3, p)
        @test X == zero_vector(M, p)
        @test get_cost(M, sco3, p) == 1.0
        # stil at 1
        @test sco3.objective.costgrad!!.i == 1
        @test get_gradient(M, sco3, q) == zero_vector(M, q)
        get_gradient!(M, X, sco3, q)
        @test X == zero_vector(M, q)
        @test get_cost(M, sco3, q) == 1.0
        @test sco3.objective.costgrad!!.i == 2

        mcgoi = ManifoldCostGradientObjective(
            TestCostGradCount(0); evaluation=InplaceEvaluation()
        )
        sco4 = Manopt.SimpleCacheObjective(M, mcgoi; p=p)
        # We evaluated on init -> evaluates twice
        @test sco4.objective.costgrad!!.i == 2
        @test get_gradient(M, sco4, p) == zero_vector(M, p)
        get_gradient!(M, X, sco4, p)
        @test X == zero_vector(M, p)
        @test get_cost(M, sco4, p) == 1.0
        # stil at 2
        @test sco4.objective.costgrad!!.i == 2
        @test get_gradient(M, sco4, q) == zero_vector(M, q)
        get_gradient!(M, X, sco4, q)
        @test X == zero_vector(M, q)
        @test get_cost(M, sco4, q) == 1.0
        @test sco4.objective.costgrad!!.i == 3
    end
end
