using Manopt, Manifolds, ManifoldsBase, Test

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
    Λ(M, x) = ProductRepr(x, forward_logs(M, x))
    function Λ!(M, Y, x)
        N = TangentBundle(M)
        copyto!(M, Y[N, :point], x)
        zero_vector!(N.manifold, Y[N, :vector], Y[N, :point])
        forward_logs!(M, Y[N, :vector], x)
        return Y
    end
    prior(M, x) = norm(norm.(Ref(pixelM), x, (Λ(M, x))[N, :vector]), 1)
    cost(M, x) = (1 / α) * fidelity(M, x) + prior(M, x)
    prox_F(M, λ, x) = prox_distance(M, λ / α, data, x, 2)
    prox_F!(M, y, λ, x) = prox_distance!(M, y, λ / α, data, x, 2)
    function prox_G_dual(N, n, λ, ξ)
        return ProductRepr(
            ξ[N, :point],
            project_collaborative_TV(
                base_manifold(N), λ, n[N, :point], ξ[N, :vector], Inf, Inf, 1.0
            ),
        )
    end
    function prox_G_dual!(N, η, n, λ, ξ)
        η[N, :point] .= ξ[N, :point]
        project_collaborative_TV!(
            base_manifold(N), η[N, :vector], λ, n[N, :point], ξ[N, :vector], Inf, Inf, 1.0
        )
        return η
    end
    DΛ(M, m, X) = ProductRepr(zero_vector(M, m), differential_forward_logs(M, m, X))
    function DΛ!(M, Y, m, X)
        N = TangentBundle(M)
        zero_vector!(M, Y[N, :point], m)
        differential_forward_logs!(M, Y[N, :vector], m, X)
        return Y
    end
    function adjoint_DΛ(N, m, n, Y)
        return adjoint_differential_forward_logs(N.manifold, m, Y[N, :vector])
    end
    function adjoint_DΛ!(N, X, m, n, Y)
        return adjoint_differential_forward_logs!(N.manifold, X, m, Y[N, :vector])
    end

    m = fill(mid_point(pixelM, data[1], data[2]), 2)
    n = Λ(M, m)
    x0 = deepcopy(data)
    ξ0 = ProductRepr(zero_vector(M, m), zero_vector(M, m))

    p_exact = PrimalDualProblem(M, N, cost, prox_F, prox_G_dual, adjoint_DΛ; Λ=Λ)
    p_linearized = PrimalDualProblem(
        M, N, cost, prox_F, prox_G_dual, adjoint_DΛ; linearized_forward_operator=DΛ
    )
    o_exact = ChambollePockOptions(m, n, x0, ξ0; variant=:exact)
    o_linearized = ChambollePockOptions(m, n, x0, ξ0; variant=:linearized)
    n_old = ProductRepr(n[N, :point], n[N, :vector])
    x_old = copy(x0)
    ξ_old = ProductRepr(ξ0[N, :point], ξ0[N, :vector])

    @testset "test Mutating/Allocation Problem Variants" begin
        p1 = PrimalDualProblem(
            M, N, cost, prox_F, prox_G_dual, adjoint_DΛ; linearized_forward_operator=DΛ, Λ=Λ
        )
        p2 = PrimalDualProblem(
            M,
            N,
            cost,
            prox_F!,
            prox_G_dual!,
            adjoint_DΛ!;
            linearized_forward_operator=DΛ!,
            Λ=Λ!,
            evaluation=MutatingEvaluation(),
        )
        x1 = get_primal_prox(p1, 1.0, x0)
        x2 = get_primal_prox(p2, 1.0, x0)
        @test x1 == x2
        get_primal_prox!(p1, x1, 0.8, x0)
        get_primal_prox!(p2, x2, 0.8, x0)
        @test x1 == x2

        ξ1 = get_dual_prox(p1, n, 1.0, ξ0)
        ξ2 = get_dual_prox(p2, n, 1.0, ξ0)
        @test ξ1[N, :point] == ξ2[N, :point]
        @test ξ1[N, :vector] == ξ2[N, :vector]
        get_dual_prox!(p1, ξ1, n, 1.0, ξ0)
        get_dual_prox!(p2, ξ2, n, 1.0, ξ0)
        @test ξ1[N, :point] == ξ2[N, :point]
        @test ξ1[N, :vector] == ξ2[N, :vector]

        y1 = forward_operator(p1, x0)
        y2 = forward_operator(p2, x0)
        @test y1[N, :point][1] == y2[N, :point][1]
        @test y1[N, :point][2] == y2[N, :point][2]
        @test y1[N, :vector][1] == y2[N, :vector][1]
        @test y1[N, :vector][2] == y2[N, :vector][2]
        forward_operator!(p1, y1, x0)
        forward_operator!(p2, y2, x0)
        @test y1[N, :point][1] == y2[N, :point][1]
        @test y1[N, :point][2] == y2[N, :point][2]
        @test y1[N, :vector][1] == y2[N, :vector][1]
        @test y1[N, :vector][2] == y2[N, :vector][2]

        X = log(M, m, x0)
        Y1 = linearized_forward_operator(p1, m, X, n)
        Y2 = linearized_forward_operator(p2, m, X, n)
        @test Y1[N, :point] == Y2[N, :point]
        @test Y1[N, :vector] == Y2[N, :vector]
        linearized_forward_operator!(p1, Y1, m, X, n)
        linearized_forward_operator!(p2, Y2, m, X, n)
        @test Y1[N, :point] == Y2[N, :point]
        @test Y1[N, :vector] == Y2[N, :vector]

        Z1 = adjoint_linearized_operator(p1, m, n, ξ0)
        Z2 = adjoint_linearized_operator(p2, m, n, ξ0)
        @test Z1 == Z2
        adjoint_linearized_operator!(p1, Z1, m, n, ξ0)
        adjoint_linearized_operator!(p2, Z2, m, n, ξ0)
        @test Z1 == Z2
    end
    @testset "Primal/Dual residual" begin
        p_exact = PrimalDualProblem(M, N, cost, prox_F, prox_G_dual, adjoint_DΛ; Λ=Λ)
        p_linearized = PrimalDualProblem(
            M, N, cost, prox_F, prox_G_dual, adjoint_DΛ; linearized_forward_operator=DΛ
        )
        o_exact = ChambollePockOptions(m, n, x0, ξ0; variant=:exact)
        o_linearized = ChambollePockOptions(m, n, x0, ξ0; variant=:linearized)
        @test primal_residual(p_exact, o_exact, x_old, ξ_old, n_old) ≈ 0 atol = 1e-16
        @test primal_residual(p_linearized, o_linearized, x_old, ξ_old, n_old) ≈ 0 atol =
            1e-16
        @test dual_residual(p_exact, o_exact, x_old, ξ_old, n_old) ≈ 4.0 atol = 1e-16
        @test dual_residual(p_linearized, o_linearized, x_old, ξ_old, n_old) ≈ 0.0 atol =
            1e-16

        step_solver!(p_exact, o_exact, 1)
        step_solver!(p_linearized, o_linearized, 1)
        @test primal_residual(p_exact, o_exact, x_old, ξ_old, n_old) > 0
        @test primal_residual(p_linearized, o_linearized, x_old, ξ_old, n_old) > 0
        @test dual_residual(p_exact, o_exact, x_old, ξ_old, n_old) > 4.0
        @test dual_residual(p_linearized, o_linearized, x_old, ξ_old, n_old) > 0

        o_err = ChambollePockOptions(m, n, x0, ξ0; variant=:err)
        @test_throws DomainError dual_residual(p_exact, o_err, x_old, ξ_old, n_old)
    end
    @testset "Debug prints" begin
        a = StoreOptionsAction((:x, :ξ, :n, :m))
        update_storage!(a, Dict(:x => x_old, :ξ => ξ_old, :n => n_old, :m => copy(m)))
        io = IOBuffer()

        d1 = DebugDualResidual(a, io)
        d1(p_exact, o_exact, 1)
        s = String(take!(io))
        @test startswith(s, "Dual Residual:")

        d2 = DebugPrimalResidual(a, io)
        d2(p_exact, o_exact, 1)
        s = String(take!(io))
        @test startswith(s, "Primal Residual: ")

        d3 = DebugPrimalDualResidual(a, io)
        d3(p_exact, o_exact, 1)
        s = String(take!(io))
        @test startswith(s, "PD Residual: ")

        d4 = DebugPrimalChange(a, "Primal Change: ", io)
        d4(p_exact, o_exact, 1)
        s = String(take!(io))
        @test startswith(s, "Primal Change: ")

        d5 = DebugPrimalIterate(io)
        d5(p_exact, o_exact, 1)
        s = String(take!(io))
        @test startswith(s, "x:")

        d6 = DebugDualIterate(io)
        d6(p_exact, o_exact, 1)
        s = String(take!(io))
        @test startswith(s, "ξ:")

        d7 = DebugDualChange(a, io)
        d7(p_exact, o_exact, 1)
        s = String(take!(io))
        @test startswith(s, "Dual Change:")

        d7a = DebugDualChange((ξ0, n), a, io)
        d7a(p_exact, o_exact, 1)
        s = String(take!(io))
        @test startswith(s, "Dual Change:")

        d8 = DebugDualBaseIterate(io)
        d8(p_exact, o_exact, 1)
        s = String(take!(io))
        @test startswith(s, "n:")

        d9 = DebugDualBaseChange(a, io)
        d9(p_exact, o_exact, 1)
        s = String(take!(io))
        @test startswith(s, "Dual Base Change:")

        d10 = DebugPrimalBaseIterate(io)
        d10(p_exact, o_exact, 1)
        s = String(take!(io))
        @test startswith(s, "m:")

        d11 = DebugPrimalBaseChange(a, io)
        d11(p_exact, o_exact, 1)
        s = String(take!(io))
        @test startswith(s, "Primal Base Change:")

        d12 = DebugDualResidual((x0, ξ0, n), a, io)
        d12(p_exact, o_exact, 1)
        s = String(take!(io))
        @test startswith(s, "Dual Residual:")

        d13 = DebugPrimalDualResidual((x0, ξ0, n), a, io)
        d13(p_exact, o_exact, 1)
        s = String(take!(io))
        @test startswith(s, "PD Residual:")

        d14 = DebugPrimalResidual((x0, ξ0, n), a, io)
        d14(p_exact, o_exact, 1)
        s = String(take!(io))
        @test startswith(s, "Primal Residual:")
    end
    @testset "Records" begin
        a = StoreOptionsAction((:x, :ξ, :n, :m))
        update_storage!(a, Dict(:x => x_old, :ξ => ξ_old, :n => n_old, :m => copy(m)))
        io = IOBuffer()

        for r in [
            RecordPrimalChange(),
            RecordPrimalIterate(x0),
            RecordDualIterate(ξ0),
            RecordDualChange(),
            RecordDualBaseIterate(n),
            RecordDualBaseChange(),
            RecordPrimalBaseIterate(x0),
            RecordPrimalBaseChange(),
        ]
            r(p_exact, o_exact, 1)
            @test length(get_record(r)) == 1
        end
    end
end
