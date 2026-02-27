using Manopt, Manifolds, ManifoldsBase, ManifoldDiff, Test, RecursiveArrayTools
using ManifoldDiff: differential_shortest_geodesic_startpoint, prox_distance

@testset "PD-RSSN" begin
    #
    # Perform an really easy test, just compute a mid point
    #
    pixelM = Sphere(2)
    signal_section_size = 1
    M = PowerManifold(pixelM, NestedPowerRepresentation(), 2 * signal_section_size)
    p1 = [1.0, 0.0, 0.0]
    p2 = 1 / sqrt(2) .* [1.0, 1.0, 0.0]
    data = vcat(fill(p1, signal_section_size), fill(p2, signal_section_size))
    α = 1.0
    σ = 0.5
    τ = 0.5
    # known minimizer
    δ = min(α / distance(pixelM, data[1], data[end]), 0.5)
    x_hat = shortest_geodesic(M, data, reverse(data), δ)
    N = M
    fidelity(M, x) = 1 / 2 * distance(M, x, f)^2
    Λ(M, x) = ArrayPartition(x, forward_logs(M, x))
    prior(M, x) = norm(norm.(Ref(M.manifold), x, submanifold_component(N, Λ(x), 2)), 1)
    f(M, x) = (1 / α) * fidelity(M, x) + prior(M, x)
    prox_f(M, λ, x) = prox_distance(M, λ / α, data, x, 2)

    prox_g_dual(N, n, λ, ξ) =
        Manopt.Test.project_collaborative_TV(N, λ, n, ξ, Inf, Inf, 1.0) # non-isotropic
    DΛ(M, m, X) = Manopt.Test.differential_forward_logs(M, m, X)
    adjoint_DΛ(N, m, n, ξ) = Manopt.Test.adjoint_differential_forward_logs(M, m, ξ)

    function Dprox_F(M, λ, x, η)
        return ManifoldDiff.differential_shortest_geodesic_startpoint(
            M, x, data, λ / (α + λ), η
        )
    end
    function Dprox_G_dual(N, n, λ, ξ, η)
        return Manopt.Test.differential_project_collaborative_TV(
            N, λ, n, ξ, η, Inf, Inf, 0.0
        )
    end

    m = fill(mid_point(pixelM, p1, p2), 2 * signal_section_size)
    n = m
    x0 = deepcopy(data)
    ξ0 = zero_vector(M, m)

    s = primal_dual_semismooth_Newton(
        M, N, f, x0, ξ0, m, n, prox_f, Dprox_F, prox_g_dual, Dprox_G_dual, DΛ, adjoint_DΛ;
        primal_stepsize = σ, dual_stepsize = τ, return_state = true,
    )
    @test startswith(
        Manopt.status_summary(s; context = :default),
        "# Solver state for `Manopt.jl`s primal dual semismooth Newton"
    )
    y = get_solver_result(s)
    @test x_hat ≈ y atol = 2 * 1.0e-7

    update_dual_base(p, o, i) = o.n
    o2 = primal_dual_semismooth_Newton(
        M, N, f, x0, ξ0, m, n, prox_f, Dprox_F, prox_g_dual, Dprox_G_dual, DΛ, adjoint_DΛ;
        primal_stepsize = σ, dual_stepsize = τ, update_dual_base = update_dual_base,
        return_state = false,
    )
    y2 = o2
    @test x_hat ≈ y2 atol = 2 * 1.0e-7
    @testset "Objective Decorator passthrough" begin
        # PDNSSN additional tests
        pdmsno = PrimalDualManifoldSemismoothNewtonObjective(
            f, prox_f, Dprox_F, prox_g_dual, Dprox_G_dual, DΛ, adjoint_DΛ
        )
        ro = Manopt.Test.DummyDecoratedObjective(pdmsno)
        X = zero_vector(M, x0)
        Y = get_differential_primal_prox(M, pdmsno, 0.1, x0, X)
        Y2 = get_differential_primal_prox(M, ro, 0.1, x0, X)
        @test Y == Y2
        get_differential_primal_prox!(M, Y, pdmsno, 0.1, x0, X)
        get_differential_primal_prox!(M, Y2, ro, 0.1, x0, X)
        @test Y == Y2

        X = zero_vector(N, ξ0)
        Y = get_differential_dual_prox(N, pdmsno, n, 0.1, ξ0, X)
        Y2 = get_differential_dual_prox(N, ro, n, 0.1, ξ0, X)
        @test Y == Y2
        get_differential_dual_prox!(N, Y, pdmsno, n, 0.1, ξ0, X)
        get_differential_dual_prox!(N, Y2, ro, n, 0.1, ξ0, X)
        @test Y == Y2
    end
end
