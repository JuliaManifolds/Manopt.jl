@testset "Gradient Plan" begin
    io = IOBuffer()
    M = Euclidean(2)
    x = RnPoint([4.,2.])
    o = GradientDescentOptions(x,stopAfterIteration(20), ConstantStepsize(1.))
    o.∇ = RnTVector([1., 0.])
    f = y -> distance(M,y,x).^2
    ∇f = y -> -2*log(M,y,x)
    p = GradientProblem(M,f,∇f)
    @test getInitialStepsize(p,o) == 1.
    @test getStepsize!(p,o,1) == 1.
    @test getLastStepsize(p,o,1) == 1.
    # Check Fallbacks of Problen
    @test getCost(p,o.x) == 0.
    @test getGradient(p,o.x) == zeroTVector(M,x)
    @test_throws ErrorException getProximalMap(p,1.,o.x,1)
    @test_throws ErrorException getSubGradient(p,o.x)
    # Additional Specific Debugs
    a1 = DebugGradient(false, x -> print(io,x))
    a1(p,o,1)
    @test String(take!(io)) == "∇F(x):RnT([1.0, 0.0])"
    a1a = DebugGradient("s:", x -> print(io,x))
    a1a(p,o,1)
    @test String(take!(io)) == "s:RnT([1.0, 0.0])"
    a2 = DebugGradientNorm(false, x -> print(io,x))
    a2(p,o,1)
    @test String(take!(io)) == "|∇F(x)|:1.0"
    a2a = DebugGradientNorm("s:", x -> print(io,x))
    a2a(p,o,1)
    @test String(take!(io)) == "s:1.0"
    a3 = DebugStepsize(false, x -> print(io,x))
    a3(p,o,1)
    @test String(take!(io)) == "s:1.0"
    a3a = DebugStepsize("S:", x -> print(io,x))
    a3a(p,o,1)
    @test String(take!(io)) == "S:1.0"
    # Additional Specific Records
    b1 = RecordGradient(o.∇)
    b1(p,o,1)
    @test b1.recordedValues == [o.∇]
    b2 = RecordGradientNorm()
    b2(p,o,1)
    @test b2.recordedValues == [1.]
    b3 = RecordStepsize()
    b3(p,o,1)
    b3(p,o,2)
    b3(p,o,3)
    @test b3.recordedValues == [1., 1., 1.]
end
