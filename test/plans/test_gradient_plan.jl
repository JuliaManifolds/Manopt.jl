s = joinpath(@__DIR__, "..", "ManoptTestSuite.jl")
!(s in LOAD_PATH) && (push!(LOAD_PATH, s))

using ManifoldsBase, Manopt, ManoptTestSuite, Test

@testset "Gradient Plan" begin
    io = IOBuffer()
    M = ManifoldsBase.DefaultManifold(2)
    p = [4.0, 2.0]
    gst = GradientDescentState(
        M;
        p=zero(p),
        stopping_criterion=StopAfterIteration(20),
        stepsize=Manopt.ConstantStepsize(M),
    )
    set_iterate!(gst, M, p)
    @test get_iterate(gst) == p
    set_gradient!(gst, M, p, [1.0, 0.0])
    @test isapprox(M, p, get_gradient(gst), [1.0, 0.0])
    f(M, q) = distance(M, q, p) .^ 2
    grad_f(M, q) = -2 * log(M, q, p)
    mgo = ManifoldFirstOrderObjective(f, grad_f)
    mp = DefaultManoptProblem(M, mgo)
    @test get_initial_stepsize(mp, gst) == 1.0
    @test get_stepsize(mp, gst, 1) == 1.0
    @test get_last_stepsize(mp, gst, 1) == 1.0
    # Check Fallbacks of Problem
    @testset "Subgradient is Gradient" begin
        @test get_subgradient(M, mgo, p) == get_gradient(M, mgo, p)
        X = zero_vector(M, p)
        Y = similar(X)
        @test get_subgradient!(M, Y, mgo, p) == get_gradient!(M, X, mgo, p)
        @test X == Y
    end
    @test get_cost(mp, gst.p) == 0.0
    @test get_gradient(mp, gst.p) == zero_vector(M, p)
    @test_throws MethodError get_proximal_map(mp, 1.0, gst.p, 1)
    @testset "Debug Gradient" begin
        a1 = DebugGradient(; long=false, io=io)
        a1(mp, gst, 1)
        @test String(take!(io)) == "grad f(p):[1.0, 0.0]"
        a1a = DebugGradient(; prefix="s:", io=io)
        a1a(mp, gst, 1)
        @test String(take!(io)) == "s:[1.0, 0.0]"
        a2 = DebugGradientNorm(; long=false, io=io)
        a2(mp, gst, 1)
        @test String(take!(io)) == "|grad f(p)|:1.0"
        a2a = DebugGradientNorm(; prefix="s:", io=io)
        a2a(mp, gst, 1)
        @test String(take!(io)) == "s:1.0"
        a3 = DebugStepsize(; long=false, io=io)
        a3(mp, gst, 1)
        @test String(take!(io)) == "s:1.0"
        a3a = DebugStepsize(; prefix="S:", io=io)
        a3a(mp, gst, 1)
        @test String(take!(io)) == "S:1.0"
    end
    @testset "Record Gradient" begin
        b1 = RecordGradient(gst.X)
        b1(mp, gst, 1)
        @test b1.recorded_values == [gst.X]
        b2 = RecordGradientNorm()
        b2(mp, gst, 1)
        @test b2.recorded_values == [1.0]
        b3 = RecordStepsize()
        b3(mp, gst, 1)
        b3(mp, gst, 2)
        b3(mp, gst, 3)
        @test b3.recorded_values == [1.0, 1.0, 1.0]
    end
    @testset "CostGradObjective" begin
        costgrad(M, p) = (f(M, p), grad_f(M, p))
        mcgo = ManifoldCostGradientObjective(costgrad)
        f2 = Manopt.get_cost_function(mcgo)
        @test f(M, p) == f2(M, p)
        @test f(M, p) == get_cost(M, mcgo, p)
        grad_f2 = Manopt.get_gradient_function(mcgo)
        X = grad_f(M, p)
        @test isapprox(M, p, X, grad_f2(M, p))
        @test isapprox(M, p, X, get_gradient(M, mcgo, p))
        Y = zero_vector(M, p)
        get_gradient!(M, Y, mcgo, p)
        @test isapprox(M, p, X, Y)

        grad_f!(M, X, q) = -2 * log!(M, X, q, p)
        costgrad!(M, X, p) = (f(M, p), grad_f!(M, X, p))
        mcgo! = ManifoldCostGradientObjective(costgrad!; evaluation=InplaceEvaluation())
        @test isapprox(M, p, X, get_gradient(M, mcgo!, p))
        get_gradient!(M, Y, mcgo!, p)
        @test isapprox(M, p, X, Y)
        cmcgo = ManifoldCountObjective(M, mcgo, [:Cost, :Gradient])
        @test get_cost(M, cmcgo, p) == get_cost(M, mcgo, p)
        @test get_gradient(M, cmcgo, p) == get_gradient(M, mcgo, p)
        get_gradient!(M, Y, cmcgo, p)
        get_gradient!(M, X, mcgo, p)
        # Verify that both were called 3 times
        @test get_count(cmcgo, :Gradient) == 3
        @test get_count(cmcgo, :Cost) == 3
    end
    @testset "Objective Decorator passthrough" begin
        ddo = ManoptTestSuite.DummyDecoratedObjective(mgo)
        @test get_cost(M, mgo, p) == get_cost(M, ddo, p)
        @test get_gradient(M, mgo, p) == get_gradient(M, ddo, p)
        X = zero_vector(M, p)
        Y = zero_vector(M, p)
        get_gradient!(M, X, ddo, p)
        get_gradient!(M, Y, ddo, p)
        @test X == Y
        @test Manopt.get_gradient_function(ddo) == Manopt.get_gradient_function(mgo)
        @test Manopt.get_cost_function(ddo) == Manopt.get_cost_function(mgo)
    end
end
