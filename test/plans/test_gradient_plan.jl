@testset "Gradient Plan" begin
    io = IOBuffer()
    M = ManifoldsBase.DefaultManifold(2)
    x = [4.0, 2.0]
    o = GradientDescentOptions(x, StopAfterIteration(20), ConstantStepsize(1.0))
    o.∇ = [1.0, 0.0]
    f = y -> distance(M, y, x) .^ 2
    ∇f = y -> -2 * log(M, y, x)
    p = GradientProblem(M, f, ∇f)
    @test get_initial_stepsize(p, o) == 1.0
    @test get_stepsize(p, o, 1) == 1.0
    @test get_last_stepsize(p, o, 1) == 1.0
    # Check Fallbacks of Problen
    @test get_cost(p, o.x) == 0.0
    @test get_gradient(p, o.x) == zero_tangent_vector(M, x)
    @test_throws ErrorException getProximalMap(p, 1.0, o.x, 1)
    @test_throws ErrorException get_subgradient(p, o.x)
    # Additional Specific Debugs
    a1 = DebugGradient(false, io)
    a1(p, o, 1)
    @test String(take!(io)) == "∇F(x):[1.0, 0.0]"
    a1a = DebugGradient("s:", io)
    a1a(p, o, 1)
    @test String(take!(io)) == "s:[1.0, 0.0]"
    a2 = DebugGradientNorm(false, io)
    a2(p, o, 1)
    @test String(take!(io)) == "|∇F(x)|:1.0"
    a2a = DebugGradientNorm("s:", io)
    a2a(p, o, 1)
    @test String(take!(io)) == "s:1.0"
    a3 = DebugStepsize(false, io)
    a3(p, o, 1)
    @test String(take!(io)) == "s:1.0"
    a3a = DebugStepsize("S:", io)
    a3a(p, o, 1)
    @test String(take!(io)) == "S:1.0"
    # Additional Specific Records
    b1 = RecordGradient(o.∇)
    b1(p, o, 1)
    @test b1.recorded_values == [o.∇]
    b2 = RecordGradientNorm()
    b2(p, o, 1)
    @test b2.recorded_values == [1.0]
    b3 = RecordStepsize()
    b3(p, o, 1)
    b3(p, o, 2)
    b3(p, o, 3)
    @test b3.recorded_values == [1.0, 1.0, 1.0]
end
