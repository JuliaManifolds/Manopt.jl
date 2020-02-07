@testset "DouglasRachford" begin
    # Though this seems a strange way, it is a way to compute the mid point
    M = Sphere(2)
    p = [1., 0., 0.]
    q = [0., 1., 0.]
    r = [0., 0., 1.]
    start = [0.,0.,1.]
    result = mid_point(M,p,q)
    F(x) = distance(M,x,p)^2 + distance(M,x,q)^2
    prox1 = (η,x) -> proxDistance(M,η,p,x)
    prox2 = (η,x) -> proxDistance(M,η,q,x)
    xHat = DouglasRachford(M,F,[prox1,prox2],start)
    @test F(start) > F(xHat)
    @test distance(M,xHat,mid_point(M,p,q)) ≈ 0
    # but we can also compute the riemannian center of mass (locally) on Sn
    # though also this is not that usweful, but easy to test that DR works
    F2(x) = distance(M,x,p)^2 + distance(M,x,q)^2 + distance(M,x,r)^2
    prox3 = (η,x) -> proxDistance(M,η,r,x)
    o = DouglasRachford(M,F2,[prox1,prox2,prox3],start;
    debug = [DebugCost(), DebugIterate(), DebugProximalParameter(),100],
    record = [RecordCost(), RecordProximalParameter()],
    returnOptions=true
    )
    xHat2 = getSolverResult(o)
    drec2 = getRecord(o)
    xCmp = 1/sqrt(3)*ones(3)
    # since the default does not run that long -> rough estimate
    @test distance(M,xHat2,xCmp) ≈ 0 atol = 10^(-5)
end
