using LRUCache, Manifolds, ManifoldDiff, Manopt, Test
using ManifoldDiff: prox_distance

@testset "Proximal Plan" begin
    M = Euclidean(2)
    p = [1.0, 2.0]
    Q = [[2.0, 3.0], [3.0, 4.0]]
    f(M, p) = 0.5 * sum(distance(M, p, q)^2 for q in Q)
    f2(M, p) = 0.5 * distance(M, p, Q[1])
    proxes_f = Tuple((N, λ, p) -> prox_distance(N, λ, q, p) for q in Q)
    ppo = ManifoldProximalMapObjective(f, proxes_f)
    ppo2 = ManifoldProximalMapObjective(f2, proxes_f[1])
    @testset "Objective Decorator passthrough" begin
        dppo = Manopt.Test.DummyDecoratedObjective(ppo)
        for i in 1:2
            @test get_proximal_map(M, ppo, 0.1, p, i) ==
                get_proximal_map(M, dppo, 0.1, p, i)
            q = copy(M, p)
            r = copy(M, p)
            get_proximal_map!(M, q, ppo, 0.1, p, i)
            get_proximal_map!(M, r, dppo, 0.1, p, i)
            @test q == r
        end
    end
    @testset "Counts" begin
        cppo = ManifoldCountObjective(M, ppo, [:ProximalMap])
        q = get_proximal_map(M, cppo, 0.1, p, 1)
        @test q == get_proximal_map(M, ppo, 0.1, p, 1)
        q2 = copy(M, p)
        get_proximal_map!(M, q2, cppo, 0.1, p, 1)
        @test q2 == q
        @test get_count(cppo, :ProximalMap, 1) == 2
        # the single ones have to be tricked a bit
        cppo2 = ManifoldCountObjective(M, ppo, Dict([:ProximalMap => 0]))
        @test q == get_proximal_map(M, cppo2, 0.1, p, 1)
        get_proximal_map!(M, q2, cppo2, 0.1, p, 1)
        @test q2 == q
        @test get_count(cppo2, :ProximalMap) == 2
        # single function
        cppo3 = ManifoldCountObjective(M, ppo2, Dict([:ProximalMap => 0]))
        q = get_proximal_map(M, cppo3, 0.1, p)
        @test q == get_proximal_map(M, ppo2, 0.1, p)
        q2 = copy(M, p)
        get_proximal_map!(M, q2, cppo3, 0.1, p)
        @test q2 == q
        @test get_count(cppo3, :ProximalMap) == 2
    end
    @testset "Cache" begin
        cppo = ManifoldCountObjective(M, ppo, [:ProximalMap])
        ccppo = objective_cache_factory(M, cppo, (:LRU, [:ProximalMap]))
        for i in 1:2
            q = get_proximal_map(M, ppo, 0.1, p, i)
            @test q == get_proximal_map(M, ccppo, 0.1, p, i)
            @test q == get_proximal_map(M, ccppo, 0.1, p, i) # Cached
            q2 = copy(M, p)
            get_proximal_map!(M, q2, ccppo, 0.1, p, i) # Cached
            @test q2 == q
            @test get_count(ccppo, :ProximalMap, i) == 1

            q = get_proximal_map(M, ppo, 0.2, -p, i)
            get_proximal_map!(M, q2, ccppo, 0.2, -p, i)
            @test q2 == q
            get_proximal_map!(M, q2, ccppo, 0.2, -p, i) # Cached
            @test q2 == q
            @test q == get_proximal_map(M, ccppo, 0.2, -p, i) # Cached
            @test get_count(ccppo, :ProximalMap, i) == 2
        end
    end
end
