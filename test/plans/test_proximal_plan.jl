using Manopt, Manifolds, Test
include("../utils/dummy_types.jl")

@testset "Proximal Plan" begin
    M = Euclidean(2)
    p = [1.0, 2.0]
    Q = [[2.0, 3.0], [3.0, 4.0]]
    f(M, p) = sum(distance(M, p, q) for q in Q)
    proxes_f = Tuple((N, λ, p) -> prox_distance(N, λ, q, p) for q in Q)
    ppo = ManifoldProximalMapObjective(f, proxes_f)
    @testset "Objetive Decorator passthrough" begin
        dppo = DummyDecoratedObjective(ppo)
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
        # ToDo
        cppo = ManifoldCountObjective(M, ppo, [:ProximalMap])
        q = get_proximal_map(M, cppo, 0.1, p, 1)
        @test q == get_proximal_map(M, ppo, 0.1, p, 1)
        q2 = copy(M, p)
        get_proximal_map!(M, q2, cppo, 0.1, p, 1)
        @test q2 == q
        @test get_count(cppo, :ProximalMap, 1) == 2
        #the single ones currenlty are not implemented
        @test_throws MethodError get_proximal_map(M, cppo, 0.1, p)
        @test_throws MethodError get_proximal_map!(M, q, cppo, 0.1, p)
    end
end
