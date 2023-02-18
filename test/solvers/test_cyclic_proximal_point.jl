using Manifolds, Manopt, Test, Dates

@testset "Cyclic Proximal Point" begin
    @testset "Allocating" begin
        n = 100
        N = PowerManifold(Circle(), n)
        q = artificial_S1_signal(n)
        f(M, p) = costL2TV(M, q, 0.5, p)
        proxes = (
            (N, λ, p) -> prox_distance(N, λ, q, p), (N, λ, p) -> prox_TV(N, 0.5 * λ, p)
        )
        q2 = cyclic_proximal_point(
            N, f, proxes, q; λ=i -> π / (2 * i), stopping_criterion=StopAfterIteration(100)
        )
        @test f(N, q) > f(N, q2)
        o = CyclicProximalPointState(
            N, f; stopping_criterion=StopAfterIteration(1), λ=i -> π / (2 * i)
        )
        mpo = ManifoldProximalMapObjective(f, proxes, [1, 2])
        p = DefaultManoptProblem(N, mpo)
        @test_throws ErrorException get_proximal_map(p, 1.0, f, 3)
        @test_throws ErrorException ManifoldProximalMapObjective(f, proxes, [1, 2, 2])
    end
    @testset "Mutating" begin
        n = 3
        M = Sphere(2)
        N = PowerManifold(M, NestedPowerRepresentation(), n)
        q = artificial_S2_lemniscate([0.0, 0.0, 1.0], n)
        f(N, p) = costL2TV(N, q, 0.5, p)
        proxes! = (
            (N, qr, λ, p) -> prox_distance!(N, qr, λ, q, p),
            (N, q, λ, p) -> prox_TV!(N, q, 0.5 * λ, p),
        )
        proxes = (
            (N, λ, p) -> prox_distance(N, λ, q, p), (N, λ, p) -> prox_TV(N, 0.5 * λ, p)
        )
        s1 = cyclic_proximal_point(
            N, f, proxes, q; λ=i -> π / (2 * i), stopping_criterion=StopAfterIteration(100)
        )
        r = cyclic_proximal_point(
            N,
            f,
            proxes!,
            q;
            λ=i -> π / (2 * i),
            stopping_criterion=StopAfterIteration(100),
            evaluation=InplaceEvaluation(),
            return_state=true,
        )
        s2 = get_solver_result(r)
        @test isapprox(N, s1, s2)
        @test startswith(
            repr(r), "# Solver state for `Manopt.jl`s Cyclic Proximal Point Algorithm"
        )
    end
    @testset "Problem access functions" begin
        n = 3
        M = Sphere(2)
        N = PowerManifold(M, NestedPowerRepresentation(), n)
        q = artificial_S2_lemniscate([0.0, 0.0, 1.0], n)
        f(N, x) = costL2TV(N, q, 0.5, x)
        proxes! = (
            (N, qr, λ, p) -> prox_distance!(N, qr, λ, q, p),
            (N, q, λ, p) -> prox_TV!(N, q, 0.5 * λ, p),
        )
        proxes = (
            (N, λ, p) -> prox_distance(N, λ, q, p), (N, λ, p) -> prox_TV(N, 0.5 * λ, p)
        )
        for i in 1:2
            mpo1 = ManifoldProximalMapObjective(f, proxes)
            dmp1 = DefaultManoptProblem(N, mpo1)
            r = deepcopy(q)
            get_proximal_map!(dmp1, r, 1.0, r, i)
            @test isapprox(N, r, get_proximal_map(dmp1, 1.0, q, i))
            mpo2 = ManifoldProximalMapObjective(f, proxes!; evaluation=InplaceEvaluation())
            dmp2 = DefaultManoptProblem(N, mpo2)
            r = deepcopy(q)
            get_proximal_map!(dmp2, r, 1.0, r, i)
            @test isapprox(N, r, get_proximal_map(dmp2, 1.0, q, i))
        end
    end
    @testset "State accsess functions" begin
        M = Euclidean(3)
        p = ones(3)
        O = CyclicProximalPointState(M, zeros(3))
        set_iterate!(O, p)
        @test get_iterate(O) == p
    end
    @testset "Debug and Record prox parameter" begin
        io = IOBuffer()
        M = Euclidean(3)
        p = ones(3)
        O = CyclicProximalPointState(M, p)
        f(M, p) = costL2TV(M, q, 0.5, p)
        proxes = (
            (M, λ, p) -> prox_distance(M, λ, q, p), (M, λ, p) -> prox_TV(M, 0.5 * λ, p)
        )
        s = CyclicProximalPointState(
            M, f; stopping_criterion=StopAfterIteration(1), λ=i -> i
        )
        mpo = ManifoldProximalMapObjective(f, proxes, [1, 2])
        p = DefaultManoptProblem(M, mpo)
        ds = DebugSolverState(s, DebugProximalParameter(; io=io))
        step_solver!(p, ds, 1)
        debug = String(take!(io))
        @test startswith(debug, "λ:")
        rs = RecordSolverState(s, RecordProximalParameter())
        step_solver!(p, rs, 1)
        @test get_record(rs) == [1.0]
    end
end
