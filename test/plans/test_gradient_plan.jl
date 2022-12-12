using Manopt, ManifoldsBase, Test

@testset "Gradient Plan" begin
    io = IOBuffer()
    M = ManifoldsBase.DefaultManifold(2)
    p = [4.0, 2.0]
    gst = GradientDescentState(
        M, zero(p); stopping_criterion=StopAfterIteration(20), stepsize=ConstantStepsize(M)
    )
    set_iterate!(gst, M, p)
    @test get_iterate(gst) == p
    gst.X = [1.0, 0.0]
    f(M, q) = distance(M, q, p) .^ 2
    grad_f(M, q) = -2 * log(M, q, p)
    mp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
    @test get_initial_stepsize(mp, gst) == 1.0
    @test get_stepsize(mp, gst, 1) == 1.0
    @test get_last_stepsize(mp, gst, 1) == 1.0
    # Check Fallbacks of Problen
    @test get_cost(mp, gst.p) == 0.0
    @test get_gradient(mp, gst.p) == zero_vector(M, p)
    @test_throws MethodError get_proximal_map(mp, 1.0, gst.p, 1)
    @test_throws MethodError get_subgradient(mp, gst.p)
    # Additional Specific Debugs
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
    # Additional Specific Records
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
