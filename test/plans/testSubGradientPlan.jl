@testset "Subgradient Plan" begin
    M = Euclidean(2)
    x = RnPoint([4., 2.])
    x0 = RnPoint([5.,2.])
    o = SubGradientMethodOptions(x0,stopAfterIteration(200), DecreasingStepsize(0.1))
    o.∂ = RnTVector([1., 0.])
    f = y -> distance(M,y,x)
    ∂f = y -> distance(M,x,y) == 0 ? zeroTVector(M,y) : -2*log(M,y,x)/distance(M,x,y)
    p = SubGradientProblem(M,f, ∂f)
    oR = solve(p,o)
    xHat = getSolverResult(p,oR)
    @test getInitialStepsize(p,o) == 0.1
    @test getStepsize!(p,o,1) == 0.1
    @test getLastStepsize(p,o,1) == 0.1
    # Check Fallbacks of Problen
    @test getCost(p,x) == 0.
    @test norm(M,x,getSubGradient(p,x)) == 0
    @test_throws ErrorException getGradient(p,o.x)
    @test_throws ErrorException getProximalMap(p,1.,o.x,1)
end
