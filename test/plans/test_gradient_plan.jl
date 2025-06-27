s = joinpath(@__DIR__, "..", "ManoptTestSuite.jl")
!(s in LOAD_PATH) && (push!(LOAD_PATH, s))

using ManifoldsBase, Manopt, ManoptTestSuite, Test

@testset "Gradient Plan" begin
    io = IOBuffer()
    M = ManifoldsBase.DefaultManifold(2)
    q = [4.0, 2.0]
    f(M, p) = distance(M, p, q) .^ 2
    grad_f(M, p) = -2 * log(M, p, q)
    grad_f!(M, X, p) = (X .= -2 * log(M, p, q))
    diff_f(M, p, X) = inner(M, p, grad_f(M, p), X)
    p = [1.0, 2.0]
    X = [0.2, 0.3]
    gst = GradientDescentState(
        M;
        p=zero(p),
        stopping_criterion=StopAfterIteration(20),
        stepsize=Manopt.ConstantStepsize(M),
    )
    set_iterate!(gst, M, q)
    @test get_iterate(gst) == q
    set_gradient!(gst, M, p, [1.0, 0.0])
    @test isapprox(M, p, get_gradient(gst), [1.0, 0.0])
    mgo = ManifoldGradientObjective(f, grad_f)
    mp = DefaultManoptProblem(M, mgo)
    @test get_initial_stepsize(mp, gst) == 1.0
    @test get_stepsize(mp, gst, 1) == 1.0
    @test get_last_stepsize(mp, gst, 1) == 1.0
    # Check Fallbacks of Problem
    @testset "Subgradient is Gradient" begin
        @test get_subgradient(M, mgo, p) == get_gradient(M, mgo, p)
        Y1 = zero_vector(M, p)
        Y2 = similar(Y1)
        @test get_subgradient!(M, Y2, mgo, p) == get_gradient!(M, Y1, mgo, p)
        @test Y1 == Y2
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
        Y1 = grad_f(M, p)
        @test isapprox(M, p, Y1, grad_f2(M, p))
        @test isapprox(M, p, Y1, get_gradient(M, mcgo, p))
        Y2 = zero_vector(M, p)
        get_gradient!(M, Y2, mcgo, p)
        @test isapprox(M, p, Y1, Y2)

        costgrad!(M, X, p) = (f(M, p), grad_f!(M, X, p))
        mcgo! = ManifoldCostGradientObjective(costgrad!; evaluation=InplaceEvaluation())
        @test isapprox(M, p, Y1, get_gradient(M, mcgo!, p))
        get_gradient!(M, Y2, mcgo!, p)
        @test isapprox(M, p, Y1, Y2)
        cmcgo = ManifoldCountObjective(M, mcgo, [:Cost, :Gradient])
        @test get_cost(M, cmcgo, p) == get_cost(M, mcgo, p)
        @test get_gradient(M, cmcgo, p) == get_gradient(M, mcgo, p)
        get_gradient!(M, Y2, cmcgo, p)
        get_gradient!(M, Y1, mcgo, p)
        # We called grad twice, cost 3x
        @test get_count(cmcgo, :Gradient) == 2
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
    @testset "FirstOrderObjective cases and Functions" begin
        c = f(M, p)
        G = grad_f(M, p)
        d = diff_f(M, p, X)
        fg(M, p) = (f(M, p), grad_f(M, p))
        fg!(M, X, p) = (f(M, p), grad_f!(M, X, p))
        gd(M, p, X) = (grad_f(M, p), diff_f(M, p, X))
        gd!(M, Y, p, X) = (grad_f!(M, Y, p), diff_f(M, p, X))
        fd(M, p, X) = (f(M, p), diff_f(M, p, X))
        fgd(M, p, X) = (f(M, p), grad_f(M, p), diff_f(M, p, X))
        fgd!(M, Y, p, X) = (f(M, p), grad_f!(M, Y, p), diff_f(M, p, X))

        # the number represents the case, a/i alloc/inplace
        # Use old names here
        mfo1a = ManifoldCostGradientObjective(fg)
        @test startswith(repr(mfo1a), "ManifoldFirstOrderObjective{AllocatingEvaluation, ")
        mfo1i = ManifoldCostGradientObjective(fg!; evaluation=InplaceEvaluation())
        mfo2a = ManifoldGradientObjective(f, grad_f)
        mfo2i = ManifoldGradientObjective(f, grad_f!; evaluation=InplaceEvaluation())
        mfo3a = ManifoldCostGradientObjective(fg; differential=diff_f)
        mfo3i = ManifoldCostGradientObjective(
            fg!; differential=diff_f, evaluation=InplaceEvaluation()
        )
        mfo4a = ManifoldGradientObjective(f, grad_f; differential=diff_f)
        mfo4i = ManifoldGradientObjective(
            f, grad_f!; differential=diff_f, evaluation=InplaceEvaluation()
        )
        mfo5a = ManifoldFirstOrderObjective(; cost=f, gradientdifferential=gd)
        mfo5i = ManifoldFirstOrderObjective(;
            cost=f, gradientdifferential=gd!, evaluation=InplaceEvaluation()
        )
        # an inplace does not make sense for 6
        mfo6 = ManifoldFirstOrderObjective(; costdifferential=fd)
        mfo7a = ManifoldFirstOrderObjective(; costdifferential=fd, gradient=grad_f)
        mfo7i = ManifoldFirstOrderObjective(;
            costdifferential=fd, gradient=grad_f!, evaluation=InplaceEvaluation()
        )
        mfo8a = ManifoldFirstOrderObjective(; costgradientdifferential=fgd)
        mfo8i = ManifoldFirstOrderObjective(;
            costgradientdifferential=fgd!, evaluation=InplaceEvaluation()
        )
        mfo9 = ManifoldFirstOrderObjective(; cost=f, differential=diff_f)

        # only cost
        @test_throws DomainError ManifoldFirstOrderObjective(; cost=f)
        # No cost
        @test_throws ErrorException ManifoldFirstOrderObjective(;)
        @test_throws ErrorException ManifoldFirstOrderObjective(; gradientdifferential=gd)
        @test_throws ErrorException ManifoldFirstOrderObjective(; gradient=grad_f)
        @test_throws ErrorException ManifoldFirstOrderObjective(; differential=diff_f)
        # test cost & diff for all
        # collect all allocs, inplace, and 6&9
        mfod1a = ManoptTestSuite.DummyDecoratedObjective(mfo1a)
        mfod1i = ManoptTestSuite.DummyDecoratedObjective(mfo1i)
        cda = [mfo1a, mfo2a, mfo3a, mfo4a, mfo5a, mfo7a, mfo8a, mfod1a]
        cdi = [mfo1i, mfo2i, mfo3i, mfo4i, mfo5i, mfo7i, mfo8i, mfod1i]
        cdr = [mfo6, mfo9]
        # For all: Test cost&diff
        Y = zero_vector(M, p)
        for obj in [cda..., cdi..., cdr...]
            @test get_cost(M, obj, p) == c
            @test Manopt.get_cost_function(obj)(M, p) == c
            # using gradient
            @test get_differential(M, obj, p, X) == d
            # using gradient!
            @test get_differential(M, obj, p, X; Y=Y) == d
            @test Manopt.get_differential_function(obj)(M, p, X) == d
        end
        Yi = zero_vector(M, p)
        # For all that have a gradient (all but 6&9) test their access
        for obj in [cda..., cdi...]
            @test get_gradient(M, obj, p) == G
            get_gradient!(M, Yi, obj, p)
            @test Yi == G
            ca, Ya = Manopt.get_cost_and_gradient(M, obj, p)
            @test ca == c
            @test Ya == G
            cb, _ = Manopt.get_cost_and_gradient!(M, Yi, obj, p)
            @test cb == c
            @test Yi == G
        end
        # For all allocs: test gradient function access.
        for obj in cda
            @test Manopt.get_gradient_function(obj)(M, p) == G
        end
        for obj in cdi
            Manopt.get_gradient_function(obj)(M, Yi, p)
            @test Yi == G
        end
    end
end
