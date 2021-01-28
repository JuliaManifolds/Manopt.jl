@testset "Manopt Cyclic Proximal Point" begin
    using Dates
    n = 100
    N = PowerManifold(Circle(), n)
    f = artificial_S1_signal(n)
    F(M, x) = costL2TV(M, f, 0.5, x)
    proxes = ((λ, x) -> prox_distance(N, λ, f, x), (λ, x) -> prox_TV(N, 0.5 * λ, x))
    o = cyclic_proximal_point(
        N,
        F,
        proxes,
        f;
        λ=i -> π / (2 * i),
        stopping_criterion=StopAfterIteration(100),
        debug=[
            DebugIterate(), " ", DebugCost(), " ", DebugProximalParameter(), "\n", 10000
        ],
        record=[RecordProximalParameter(), RecordIterate(f), RecordCost()],
        return_options=true,
    )
    fR = get_solver_result(o)
    fR2 = cyclic_proximal_point(
        N, F, proxes, f; λ=i -> π / (2 * i), stopping_criterion=StopAfterIteration(100)
    )
    @test fR == fR2
    rec = get_record(o)
    @test F(N, f) > F(N, fR)
    #
    o = CyclicProximalPointOptions(f, StopAfterIteration(1), i -> π / (2 * i))
    p = ProximalProblem(N, F, proxes, [1, 2])
    @test_throws ErrorException get_proximal_map(p, 1.0, f, 3)
    @test_throws ErrorException ProximalProblem(N, F, proxes, [1, 2, 2])
end
