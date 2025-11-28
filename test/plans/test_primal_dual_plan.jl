using Manifolds, ManifoldDiff, ManifoldsBase, Manopt, Test
using RecursiveArrayTools

@testset "Test primal dual plan" begin
    #
    # Perform an really easy test, just compute a mid point
    #
    pixelM = Sphere(2)
    M = PowerManifold(pixelM, NestedPowerRepresentation(), 2)
    data = [[1.0, 0.0, 0.0], 1 / sqrt(2) .* [1.0, 1.0, 0.0]]
    α = 1
    # known minimizer
    δ = min(α / distance(pixelM, data[1], data[2]), 0.5)
    x_hat = shortest_geodesic(M, data, reverse(data), δ)
    N = TangentBundle(M)
    fidelity(M, x) = 1 / 2 * distance(M, x, f)^2
    Λ(M, x) = ArrayPartition(x, Manopt.Test.forward_logs(M, x))
    function Λ!(M, Y, x)
        N = TangentBundle(M)
        copyto!(M, Y[N, :point], x)
        zero_vector!(N.manifold, Y[N, :vector], Y[N, :point])
        Manopt.Test.forward_logs!(M, Y[N, :vector], x)
        return Y
    end
    prior(M, x) = norm(norm.(Ref(pixelM), x, (Λ(M, x))[N, :vector]), 1)
    f(M, x) = (1 / α) * fidelity(M, x) + prior(M, x)
    prox_f(M, λ, x) = ManifoldDiff.prox_distance(M, λ / α, data, x, 2)
    prox_f!(M, y, λ, x) = ManifoldDiff.prox_distance!(M, y, λ / α, data, x, 2)
    function prox_g_dual(N, n, λ, ξ)
        return ArrayPartition(
            ξ[N, :point],
            Manopt.Test.project_collaborative_TV(
                base_manifold(N), λ, n[N, :point], ξ[N, :vector], Inf, Inf, 1.0
            ),
        )
    end
    function prox_g_dual!(N, η, n, λ, ξ)
        η[N, :point] .= ξ[N, :point]
        Manopt.Test.project_collaborative_TV!(
            base_manifold(N), η[N, :vector], λ, n[N, :point], ξ[N, :vector], Inf, Inf, 1.0
        )
        return η
    end
    DΛ(M, m, X) = ArrayPartition(
        zero_vector(M, m), Manopt.Test.differential_forward_logs(M, m, X)
    )
    function DΛ!(M, Y, m, X)
        N = TangentBundle(M)
        zero_vector!(M, Y[N, :point], m)
        Manopt.Test.differential_forward_logs!(M, Y[N, :vector], m, X)
        return Y
    end
    function adjoint_DΛ(N, m, n, Y)
        return Manopt.Test.adjoint_differential_forward_logs(
            N.manifold, m, Y[N, :vector]
        )
    end
    function adjoint_DΛ!(N, X, m, n, Y)
        return Manopt.Test.adjoint_differential_forward_logs!(
            N.manifold, X, m, Y[N, :vector]
        )
    end

    m = fill(mid_point(pixelM, data[1], data[2]), 2)
    n = Λ(M, m)
    p0 = deepcopy(data)
    X0 = ArrayPartition(zero_vector(M, m), zero_vector(M, m))

    pdmoe = PrimalDualManifoldObjective(f, prox_f, prox_g_dual, adjoint_DΛ; Λ = Λ)
    p_exact = TwoManifoldProblem(M, N, pdmoe)
    pdmol = PrimalDualManifoldObjective(
        f, prox_f, prox_g_dual, adjoint_DΛ; linearized_forward_operator = DΛ
    )
    p_linearized = TwoManifoldProblem(M, N, pdmol)
    s_exact = ChambollePockState(M; m = m, n = n, p = zero.(p0), X = X0, variant = :exact)
    s_linearized = ChambollePockState(M; m = m, n = n, p = p0, X = X0, variant = :linearized)
    n_old = ArrayPartition(n[N, :point], n[N, :vector])
    p_old = copy(p0)
    ξ_old = ArrayPartition(X0[N, :point], X0[N, :vector])

    set_iterate!(s_exact, p0)
    @test all(get_iterate(s_exact) .== p0)

    osm = PrimalDualSemismoothNewtonState(
        M;
        m = m,
        n = n,
        p = zero.(p0),
        X = X0,
        primal_stepsize = 0.0,
        dual_stepsize = 0.0,
        regularization_parameter = 0.0,
    )
    set_iterate!(osm, p0)
    @test all(get_iterate(osm) .== p0)

    @testset "test Mutating/Allocation Problem Variants" begin
        pdmoa = PrimalDualManifoldObjective(
            f, prox_f, prox_g_dual, adjoint_DΛ; linearized_forward_operator = DΛ, Λ = Λ
        )
        p1 = TwoManifoldProblem(M, N, pdmoa)
        pdmoi = PrimalDualManifoldObjective(
            f,
            prox_f!,
            prox_g_dual!,
            adjoint_DΛ!;
            linearized_forward_operator = (DΛ!),
            Λ = (Λ!),
            evaluation = InplaceEvaluation(),
        )
        p2 = TwoManifoldProblem(M, N, pdmoi)
        x1 = get_primal_prox(p1, 1.0, p0)
        x2 = get_primal_prox(p2, 1.0, p0)
        @test x1 == x2
        get_primal_prox!(p1, x1, 0.8, p0)
        get_primal_prox!(p2, x2, 0.8, p0)
        @test x1 == x2

        ξ1 = get_dual_prox(p1, n, 1.0, X0)
        ξ2 = get_dual_prox(p2, n, 1.0, X0)
        @test ξ1[N, :point] == ξ2[N, :point]
        @test ξ1[N, :vector] == ξ2[N, :vector]
        get_dual_prox!(p1, ξ1, n, 1.0, X0)
        get_dual_prox!(p2, ξ2, n, 1.0, X0)
        @test ξ1[N, :point] == ξ2[N, :point]
        @test ξ1[N, :vector] == ξ2[N, :vector]

        y1 = forward_operator(p1, p0)
        y2 = forward_operator(p2, p0)
        @test y1[N, :point][1] == y2[N, :point][1]
        @test y1[N, :point][2] == y2[N, :point][2]
        @test y1[N, :vector][1] == y2[N, :vector][1]
        @test y1[N, :vector][2] == y2[N, :vector][2]
        forward_operator!(p1, y1, p0)
        forward_operator!(p2, y2, p0)
        @test y1[N, :point][1] == y2[N, :point][1]
        @test y1[N, :point][2] == y2[N, :point][2]
        @test y1[N, :vector][1] == y2[N, :vector][1]
        @test y1[N, :vector][2] == y2[N, :vector][2]

        X = log(M, m, p0)
        Y1 = linearized_forward_operator(p1, m, X, n)
        Y2 = linearized_forward_operator(p2, m, X, n)
        @test Y1[N, :point] == Y2[N, :point]
        @test Y1[N, :vector] == Y2[N, :vector]
        linearized_forward_operator!(p1, Y1, m, X, n)
        linearized_forward_operator!(p2, Y2, m, X, n)
        @test Y1[N, :point] == Y2[N, :point]
        @test Y1[N, :vector] == Y2[N, :vector]

        Z1 = adjoint_linearized_operator(p1, m, n, X0)
        Z2 = adjoint_linearized_operator(p2, m, n, X0)
        @test Z1 == Z2
        adjoint_linearized_operator!(p1, Z1, m, n, X0)
        adjoint_linearized_operator!(p2, Z2, m, n, X0)
        @test Z1 == Z2
    end
    @testset "Primal/Dual residual" begin
        pmdoe = PrimalDualManifoldObjective(f, prox_f, prox_g_dual, adjoint_DΛ; Λ = Λ)
        p_exact = TwoManifoldProblem(M, N, pdmoe)
        pmdol = PrimalDualManifoldObjective(
            f, prox_f, prox_g_dual, adjoint_DΛ; linearized_forward_operator = DΛ
        )
        p_linearized = TwoManifoldProblem(M, N, pmdol)
        s_exact = ChambollePockState(M; m = m, n = n, p = p0, X = X0, variant = :exact)
        s_linearized = ChambollePockState(M; m = m, n = n, p = p0, X = X0, variant = :linearized)
        @test primal_residual(p_exact, s_exact, p_old, ξ_old, n_old) ≈ 0 atol = 1.0e-16
        @test primal_residual(p_linearized, s_linearized, p_old, ξ_old, n_old) ≈ 0 atol =
            1.0e-16
        @test dual_residual(p_exact, s_exact, p_old, ξ_old, n_old) ≈ 4.0 atol = 1.0e-16
        @test dual_residual(p_linearized, s_linearized, p_old, ξ_old, n_old) ≈ 0.0 atol =
            1.0e-16

        step_solver!(p_exact, s_exact, 1)
        step_solver!(p_linearized, s_linearized, 1)
        @test primal_residual(p_exact, s_exact, p_old, ξ_old, n_old) > 0
        @test primal_residual(p_linearized, s_linearized, p_old, ξ_old, n_old) > 0
        @test dual_residual(p_exact, s_exact, p_old, ξ_old, n_old) > 4.0
        @test dual_residual(p_linearized, s_linearized, p_old, ξ_old, n_old) > 0

        o_err = ChambollePockState(M; m = m, n = n, p = p0, X = X0, variant = :err)
        @test_throws DomainError dual_residual(p_exact, o_err, p_old, ξ_old, n_old)
    end
    @testset "Debug prints" begin
        a = StoreStateAction([:Iterate, :X, :n, :m])
        update_storage!(a, Dict(:Iterate => p_old, :X => ξ_old, :n => n_old, :m => copy(m)))
        io = IOBuffer()

        d1 = DebugDualResidual(; storage = a, io = io)
        d1(p_exact, s_exact, 1)
        s = String(take!(io))
        @test startswith(s, "Dual Residual:")

        d2 = DebugPrimalResidual(; storage = a, io = io)
        d2(p_exact, s_exact, 1)
        s = String(take!(io))
        @test startswith(s, "Primal Residual: ")

        d3 = DebugPrimalDualResidual(; storage = a, io = io)
        d3(p_exact, s_exact, 1)
        s = String(take!(io))
        @test startswith(s, "PD Residual: ")

        d4 = DebugPrimalChange(; storage = a, prefix = "Primal Change: ", io = io)
        d4(p_exact, s_exact, 1)
        s = String(take!(io))
        @test startswith(s, "Primal Change: ")

        d5 = DebugPrimalIterate(; io = io)
        d5(p_exact, s_exact, 1)
        s = String(take!(io))
        @test startswith(s, "p:")

        d6 = DebugDualIterate(; io = io)
        d6(p_exact, s_exact, 1)
        s = String(take!(io))
        @test startswith(s, "X:")

        d7 = DebugDualChange(; storage = a, io = io)
        d7(p_exact, s_exact, 1)
        s = String(take!(io))
        @test startswith(s, "Dual Change:")

        d7a = DebugDualChange((X0, n); storage = a, io = io)
        d7a(p_exact, s_exact, 1)
        s = String(take!(io))
        @test startswith(s, "Dual Change:")

        d8 = DebugDualBaseIterate(; io = io)
        d8(p_exact, s_exact, 1)
        s = String(take!(io))
        @test startswith(s, "n:")

        d9 = DebugDualBaseChange(; storage = a, io = io)
        d9(p_exact, s_exact, 1)
        s = String(take!(io))
        @test startswith(s, "Dual Base Change:")

        d10 = DebugPrimalBaseIterate(; io = io)
        d10(p_exact, s_exact, 1)
        s = String(take!(io))
        @test startswith(s, "m:")

        d11 = DebugPrimalBaseChange(; storage = a, io = io)
        d11(p_exact, s_exact, 1)
        s = String(take!(io))
        @test startswith(s, "Primal Base Change:")

        d12 = DebugDualResidual((p0, X0, n); storage = a, io = io)
        d12(p_exact, s_exact, 1)
        s = String(take!(io))
        @test startswith(s, "Dual Residual:")

        d13 = DebugPrimalDualResidual((p0, X0, n); storage = a, io = io)
        d13(p_exact, s_exact, 1)
        s = String(take!(io))
        @test startswith(s, "PD Residual:")

        d14 = DebugPrimalResidual((p0, X0, n); storage = a, io = io)
        d14(p_exact, s_exact, 1)
        s = String(take!(io))
        @test startswith(s, "Primal Residual:")
    end
    @testset "Records" begin
        a = StoreStateAction([:Iterate, :X, :n, :m])
        update_storage!(a, Dict(:Iterate => p_old, :X => ξ_old, :n => n_old, :m => copy(m)))
        io = IOBuffer()

        for r in [
                RecordPrimalChange(),
                RecordPrimalIterate(p0),
                RecordDualIterate(X0),
                RecordDualChange(),
                RecordDualBaseIterate(n),
                RecordDualBaseChange(),
                RecordPrimalBaseIterate(p0),
                RecordPrimalBaseChange(),
            ]
            r(p_exact, s_exact, 1)
            @test length(get_record(r)) == 1
        end
    end
    @testset "Objective Decorator passthrough" begin
        # PD
        pdmo = PrimalDualManifoldObjective(
            f, prox_f, prox_g_dual, adjoint_DΛ; Λ = Λ, linearized_forward_operator = DΛ
        )
        ro = Manopt.Test.DummyDecoratedObjective(pdmo)
        q1 = get_primal_prox(M, ro, 0.1, p0)
        q2 = get_primal_prox(M, pdmo, 0.1, p0)
        @test q1 == q2
        get_primal_prox!(M, q1, ro, 0.1, p0)
        get_primal_prox!(M, q2, pdmo, 0.1, p0)
        @test q1 == q2
        Y1 = get_dual_prox(N, ro, n, 0.1, X0)
        Y2 = get_dual_prox(N, pdmo, n, 0.1, X0)
        @test Y1 == Y2
        get_dual_prox!(N, Y1, ro, n, 0.1, X0)
        get_dual_prox!(N, Y2, pdmo, n, 0.1, X0)
        @test Y1 == Y2
        Y1 = linearized_forward_operator(M, N, ro, m, p0, n)
        Y2 = linearized_forward_operator(M, N, pdmol, m, p0, n)
        @test Y1 == Y2
        linearized_forward_operator!(M, N, Y1, ro, m, p0, n)
        linearized_forward_operator!(M, N, Y2, pdmol, m, p0, n)
        @test Y1 == Y2
        Z1 = adjoint_linearized_operator(M, N, ro, m, n, X0)
        Z2 = adjoint_linearized_operator(M, N, pdmol, m, n, X0)
        @test Z1 == Z2
        adjoint_linearized_operator!(M, N, Z1, ro, m, n, X0)
        adjoint_linearized_operator!(M, N, Z2, pdmol, m, n, X0)
        @test Z1 == Z2
        s = forward_operator(M, N, ro, p0)
        t = forward_operator(M, N, pdmo, p0)
        @test s == t
        forward_operator!(M, N, s, ro, p0)
        forward_operator!(M, N, t, pdmo, p0)
        @test Y1 == Y2
    end
end
