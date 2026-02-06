using Manifolds, Manopt, Test, Dates, LRUCache
using ManifoldDiff: prox_distance, prox_distance!

@testset "Cyclic Proximal Point" begin
    @testset "Allocating" begin
        n = 100
        N = PowerManifold(Circle(), n)
        q = [exp(Circle(), 0, X) for X in range(0, 3π; length = n)]
        f(M, p) = Manopt.Test.L2_Total_Variation(M, q, 0.5, p)
        proxes = (
            (N, λ, p) -> prox_distance(N, λ, q, p),
            (N, λ, p) -> Manopt.Test.prox_Total_Variation(N, 0.5 * λ, p),
        )
        q2 = cyclic_proximal_point(
            N, f, proxes, q; λ = i -> π / (2 * i), stopping_criterion = StopAfterIteration(100)
        )
        @test f(N, q) > f(N, q2)
        q3 = copy(N, q)
        cyclic_proximal_point!(
            N, f, proxes, q3; λ = i -> π / (2 * i), stopping_criterion = StopAfterIteration(100)
        )
        cpps = CyclicProximalPointState(
            N; p = q, stopping_criterion = StopAfterIteration(1), λ = i -> π / (2 * i)
        )
        mpo = ManifoldProximalMapObjective(f, proxes, [1, 2])
        p = DefaultManoptProblem(N, mpo)
        @test_throws ErrorException get_proximal_map(p, 1.0, f, 3)
        @test_throws ErrorException ManifoldProximalMapObjective(f, proxes, [1, 2, 2])
    end
    @testset "Number" begin
        M = Circle()
        data = [-π / 2, π / 4, 0.0, π / 4]
        q = sum(data) / length(data)
        f(M, p) = 1 / 10 * sum(distance.(Ref(M), dara, Ref(p)) .^ 2)
        proxes_f = [(N, λ, p) -> prox_distance(N, λ, q, p) for q in data]
        p1 = cyclic_proximal_point(M, f, proxes_f, data[1])
        @test isapprox(M, q, p1; atol = 1.0e-3)
        p2 = cyclic_proximal_point(M, f, proxes_f, data[1]; evaluation = InplaceEvaluation())
        @test p1 == p2
        s = cyclic_proximal_point(M, f, proxes_f, data[1]; return_state = true)
        p3 = get_solver_result(s)[]
        @test p2 == p3
    end
    @testset "Mutating" begin
        n = 3
        M = Sphere(2)
        N = PowerManifold(M, NestedPowerRepresentation(), n)
        q = [ #Adapted Lemniscate
            exp(
                    M,
                    [0.0, 0.0, 1.0],
                    π / 2.0 * (cos(t) / (sin(t)^2 + 1.0)) * [1.0, 0.0, 0.0] +
                    π / 2.0 * (cos(t) * sin(t) / (sin(t)^2 + 1.0)) * [0.0, 1.0, 0.0],
                ) for t in range(0, 2π; length = n)
        ]
        f(N, p) = Manopt.Test.L2_Total_Variation(N, q, 0.5, p)
        proxes! = (
            (N, qr, λ, p) -> prox_distance!(N, qr, λ, q, p),
            (N, q, λ, p) -> Manopt.Test.prox_Total_Variation!(N, q, 0.5 * λ, p),
        )
        proxes = (
            (N, λ, p) -> prox_distance(N, λ, q, p),
            (N, λ, p) -> Manopt.Test.prox_Total_Variation(N, 0.5 * λ, p),
        )
        s1 = cyclic_proximal_point(
            N, f, proxes, q; λ = i -> π / (2 * i), stopping_criterion = StopAfterIteration(100)
        )
        r = cyclic_proximal_point(
            N,
            f,
            proxes!,
            q;
            λ = i -> π / (2 * i),
            stopping_criterion = StopAfterIteration(100),
            evaluation = InplaceEvaluation(),
            return_state = true,
        )
        s2 = get_solver_result(r)
        @test isapprox(N, s1, s2)
        @test startswith(
            Manopt.status_sumary(r; inline = false),
            "# Solver state for `Manopt.jl`s Cyclic Proximal Point Algorithm"
        )
        @testset "Caching" begin
            r2 = cyclic_proximal_point(
                N,
                f,
                proxes!,
                q;
                λ = i -> π / (2 * i),
                cache = (:LRU, [:Cost, :ProximalMap], 50),
                stopping_criterion = StopAfterIteration(100),
                evaluation = InplaceEvaluation(),
                return_state = true,
                return_objective = true,
            )
        end
    end
    @testset "Problem access functions" begin
        n = 3
        M = Sphere(2)
        N = PowerManifold(M, NestedPowerRepresentation(), n)
        q = [ #Adapted Lemniscate
            exp(
                    M,
                    [0.0, 0.0, 1.0],
                    π / 2.0 * (cos(t) / (sin(t)^2 + 1.0)) * [1.0, 0.0, 0.0] +
                    π / 2.0 * (cos(t) * sin(t) / (sin(t)^2 + 1.0)) * [0.0, 1.0, 0.0],
                ) for t in range(0, 2π; length = n)
        ]
        f(N, x) = Manopt.Test.L2_Total_Variation(N, q, 0.5, x)
        proxes! = (
            (N, qr, λ, p) -> prox_distance!(N, qr, λ, q, p),
            (N, q, λ, p) -> Manopt.Test.prox_Total_Variation!(N, q, 0.5 * λ, p),
        )
        proxes = (
            (N, λ, p) -> prox_distance(N, λ, q, p),
            (N, λ, p) -> Manopt.Test.prox_Total_Variation(N, 0.5 * λ, p),
        )
        for i in 1:2
            mpo1 = ManifoldProximalMapObjective(f, proxes)
            dmp1 = DefaultManoptProblem(N, mpo1)
            r = deepcopy(q)
            get_proximal_map!(dmp1, r, 1.0, r, i)
            @test isapprox(N, r, get_proximal_map(dmp1, 1.0, q, i))
            mpo2 = ManifoldProximalMapObjective(f, proxes!; evaluation = InplaceEvaluation())
            dmp2 = DefaultManoptProblem(N, mpo2)
            r = deepcopy(q)
            get_proximal_map!(dmp2, r, 1.0, r, i)
            @test isapprox(N, r, get_proximal_map(dmp2, 1.0, q, i))
        end
    end
    @testset "State access functions" begin
        M = Euclidean(3)
        p = ones(3)
        O = CyclicProximalPointState(M; p = zeros(3))
        set_iterate!(O, p)
        @test get_iterate(O) == p
    end
    @testset "Debug and Record prox parameter" begin
        io = IOBuffer()
        M = Euclidean(3)
        p = ones(3)
        O = CyclicProximalPointState(M; p = p)
        f(M, p) = Manopt.Test.L2_Total_Variation(M, q, 0.5, p)
        proxes = (
            (M, λ, p) -> prox_distance(M, λ, q, p),
            (M, λ, p) -> Manopt.Test.prox_Total_Variation(M, 0.5 * λ, p),
        )
        s = CyclicProximalPointState(
            M; p = p, stopping_criterion = StopAfterIteration(1), λ = i -> i
        )
        mpo = ManifoldProximalMapObjective(f, proxes, [1, 2])
        dmp = DefaultManoptProblem(M, mpo)
        ds = DebugSolverState(s, DebugProximalParameter(; io = io))
        step_solver!(dmp, ds, 1)
        debug = String(take!(io))
        @test startswith(debug, "λ:")
        rs = RecordSolverState(s, RecordProximalParameter())
        step_solver!(dmp, rs, 1)
        @test get_record(rs) == [1.0]
    end
end
