using Manifolds, Manopt, Test, Dates

@testset "Cyclic Proximal Point" begin
    @testset "Allocating" begin
        n = 100
        N = PowerManifold(Circle(), n)
        f = artificial_S1_signal(n)
        F(M, x) = costL2TV(M, f, 0.5, x)
        proxes = (
            (N, λ, x) -> prox_distance(N, λ, f, x), (N, λ, x) -> prox_TV(N, 0.5 * λ, x)
        )
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
        o = CyclicProximalPointOptions(f, StopAfterIteration(1), i -> π / (2 * i))
        p = ProximalProblem(N, F, proxes, [1, 2])
        @test_throws ErrorException get_proximal_map(p, 1.0, f, 3)
        @test_throws ErrorException ProximalProblem(N, F, proxes, [1, 2, 2])
    end
    @testset "Mutating" begin
        n = 3
        M = Sphere(2)
        N = PowerManifold(M, NestedPowerRepresentation(), n)
        f = artificial_S2_lemniscate([0.0, 0.0, 1.0], n)
        F(N, x) = costL2TV(N, f, 0.5, x)
        proxes! = (
            (N, y, λ, x) -> prox_distance!(N, y, λ, f, x),
            (N, y, λ, x) -> prox_TV!(N, y, 0.5 * λ, x),
        )
        proxes = (
            (N, λ, x) -> prox_distance(N, λ, f, x), (N, λ, x) -> prox_TV(N, 0.5 * λ, x)
        )
        s1 = cyclic_proximal_point(
            N, F, proxes, f; λ=i -> π / (2 * i), stopping_criterion=StopAfterIteration(100)
        )
        s2 = cyclic_proximal_point(
            N,
            F,
            proxes!,
            f;
            λ=i -> π / (2 * i),
            stopping_criterion=StopAfterIteration(100),
            evaluation=MutatingEvaluation(),
        )
        @test isapprox(N, s1, s2)
    end
    @testset "Problem access functions" begin
        n = 3
        M = Sphere(2)
        N = PowerManifold(M, NestedPowerRepresentation(), n)
        f = artificial_S2_lemniscate([0.0, 0.0, 1.0], n)
        F(N, x) = costL2TV(N, f, 0.5, x)
        proxes! = (
            (N, y, λ, x) -> prox_distance!(N, y, λ, f, x),
            (N, y, λ, x) -> prox_TV!(N, y, 0.5 * λ, x),
        )
        proxes = (
            (N, λ, x) -> prox_distance(N, λ, f, x), (N, λ, x) -> prox_TV(N, 0.5 * λ, x)
        )
        for i in 1:2
            p1 = ProximalProblem(N, F, proxes)
            g = deepcopy(f)
            get_proximal_map!(p1, g, 1.0, g, i)
            @test isapprox(N, g, get_proximal_map(p1, 1.0, f, i))
            p2 = ProximalProblem(N, F, proxes!; evaluation=MutatingEvaluation())
            g = deepcopy(f)
            get_proximal_map!(p2, g, 1.0, g, i)
            @test isapprox(N, g, get_proximal_map(p2, 1.0, f, i))
        end
    end
end
