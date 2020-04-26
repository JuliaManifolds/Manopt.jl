@testset "Manopt Cyclic Proximal Point" begin
    using Dates
    n = 100
    N = PowerManifold(Circle(),n)
    f = artificial_S1_signal(n)
    F = x -> costL2TV(N,f,0.5,x)
    proxes = [ (λ,x) -> prox_distance(N,λ,f,x), (λ,x) -> prox_TV(N,0.5*λ,x) ]
    o = cyclic_proximal_point(
        N,
        F,
        proxes,
        f;
        λ = i -> π/(2*i),
        stopping_criterion = StopAfterIteration(100),
        debug = [DebugIterate()," ",DebugCost()," ",DebugProximalParameter(),"\n",10000],
        record = [RecordProximalParameter(), RecordIterate(f), RecordCost()],
        return_options=true
        )
    fR = get_solver_result(o)
    fR2 = cyclic_proximal_point(
        N,
        F,
        proxes,
        f;
        λ = i -> π/(2*i),
        stopping_criterion = StopAfterIteration(100),
    )
    @test fR == fR2
    rec = get_record(o)
    @test F(f) > F(fR)
    #
    o = CyclicProximalPointOptions(f, StopAfterIteration(1), i -> π/(2*i))
    p = ProximalProblem(N,F,proxes,[1,2])
    @test_throws ErrorException getProximalMap(p,1.,f,3)

    order1 = Manopt.update_cpp_order(3,0,1:3,RandomEvalOrder())
    @test all(isinteger.(order1))
    @test minimum(order1) == 1
    @test maximum(order1) == 3

    # i=0 introduce new fixed ranom order
    order2 = Manopt.update_cpp_order(3,0,1:3,FixedRandomEvalOrder())
    @test all(isinteger.(order2))
    @test minimum(order2) == 1
    @test maximum(order2) == 3
    order3 = Manopt.update_cpp_order(3,1,order2,FixedRandomEvalOrder())
    @test order3 == order2 # only update on i=0

    @test Manopt.update_cpp_order(3,0,1:3,LinearEvalOrder()) == 1:3
    @test Manopt.update_cpp_order(3,1,1:3,LinearEvalOrder()) == 1:3
end