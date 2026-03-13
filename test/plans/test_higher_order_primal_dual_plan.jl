using Manopt, Manifolds, ManifoldsBase, Test, RecursiveArrayTools
using Manopt.Test: forward_logs, adjoint_differential_forward_logs
using ManifoldDiff:
    differential_shortest_geodesic_startpoint,
    differential_shortest_geodesic_startpoint!,
    prox_distance!
@testset "Test higher order primal dual plan" begin
    # Perform an really easy test, just compute a mid point
    #
    pixelM = Sphere(2)
    signal_section_size = 1
    M = PowerManifold(pixelM, NestedPowerRepresentation(), 2 * signal_section_size)
    data = [[1.0, 0.0, 0.0], 1 / sqrt(2) .* [1.0, 1.0, 0.0]]
    α = 1
    # known minimizer
    δ = min(α / distance(pixelM, data[1], data[2]), 0.5)
    p_hat = shortest_geodesic(M, data, reverse(data), δ)
    N = M
    fidelity(M, p) = 1 / 2 * distance(M, p, f)^2
    Λ(M, p) = ArrayPartition(p, forward_logs(M, p))
    prior(M, p) = norm(norm.(Ref(M.manifold), p, submanifold_component(N, Λ(p), 2)), 1)
    f(M, p) = (1 / α) * fidelity(M, p) + prior(M, p)
    prox_f(M, λ, p) = prox_distance(M, λ / α, data, p, 2)
    prox_f!(M, q, λ, p) = prox_distance!(M, q, λ / α, data, p, 2)
    prox_g_dual(N, n, λ, X) = project_collaborative_TV(N, λ, n, X, Inf, Inf, 1.0)
    prox_g_dual!(N, η, n, λ, X) = project_collaborative_TV(N, η, λ, n, X, Inf, Inf, 1.0)
    DΛ(M, m, X) = differential_forward_logs(M, m, X)
    DΛ!(M, Y, m, X) = differential_forward_logs!(M, Y, m, X)
    adjoint_DΛ(N, m, n, X) = adjoint_differential_forward_logs(M, m, X)
    adjoint_DΛ!(N, Y, m, n, X) = adjoint_differential_forward_logs!(M, Y, m, X)

    function Dprox_F(M, λ, p, X)
        return differential_shortest_geodesic_startpoint(M, p, data, λ / (α + λ), X)
    end
    function Dprox_F!(M, Y, λ, p, X)
        differential_shortest_geodesic_startpoint!(M, Y, p, data, λ / (α + λ), X)
        return Y
    end
    function Dprox_G_dual(N, n, λ, X, Y)
        return Manopt.Test.differential_project_collaborative_TV(N, n, X, Y, Inf, Inf)
    end
    function Dprox_G_dual!(N, Z, n, λ, X, Y)
        return Manopt.Test.differential_project_collaborative_TV!(
            N, Z, n, X, Y, Inf, Inf
        )
    end

    m = fill(mid_point(pixelM, data[1], data[2]), 2)
    n = m
    p0 = deepcopy(data)
    ξ0 = zero_vector(M, m)
    X = log(M, p0, m)
    Ξ = X

    obj1 = PrimalDualManifoldSemismoothNewtonObjective(
        f, prox_f, Dprox_F, prox_g_dual, Dprox_G_dual, DΛ, adjoint_DΛ
    )
    obj2 = PrimalDualManifoldSemismoothNewtonObjective(
        f, prox_f!, Dprox_F!, prox_g_dual!, Dprox_G_dual!, DΛ!, adjoint_DΛ!;
        evaluation = InplaceEvaluation(),
    )
    obj3 = PrimalDualManifoldSemismoothNewtonObjective(
        f, prox_f, Dprox_F, prox_g_dual, Dprox_G_dual, DΛ, adjoint_DΛ, Λ = Λ
    )
    @testset "test Mutating/Allocation Problem Variants" begin
        p1 = TwoManifoldProblem(M, N, obj1)
        p2 = TwoManifoldProblem(M, N, obj2)
        x1 = get_differential_primal_prox(p1, 1.0, p0, X)
        x2 = get_differential_primal_prox(p2, 1.0, p0, X)
        @test x1 == x2
        get_differential_primal_prox!(p1, x1, 1.0, p0, X)
        get_differential_primal_prox!(p2, x2, 1.0, p0, X)
        @test x1 == x2

        ξ1 = get_differential_dual_prox(p1, n, 1.0, ξ0, Ξ)
        ξ2 = get_differential_dual_prox(p2, n, 1.0, ξ0, Ξ)
        @test ξ1 ≈ ξ2 atol = 2 * 1.0e-16
        get_differential_dual_prox!(p1, ξ1, n, 1.0, ξ0, Ξ)
        get_differential_dual_prox!(p2, ξ2, n, 1.0, ξ0, Ξ)
        @test ξ1 ≈ ξ2 atol = 2 * 1.0e-16
    end
    @testset "show/repr and status_summary" begin
        s1 = repr(obj1)
        @test startswith(s1, "PrimalDualManifoldSemismoothNewtonObjective(f,")
        s2 = repr(obj2)
        @test startswith(s2, "PrimalDualManifoldSemismoothNewtonObjective(f,")
        @test contains(s2, "InplaceEvaluation")
        s3 = repr(obj3)
        @test startswith(s3, "PrimalDualManifoldSemismoothNewtonObjective(f,")
        @test contains(s3, "Λ = ")
        s4 = Manopt.status_summary(obj1)
        @test startswith(s4, "A primal dual semismooth Newton objective")
        s5 = Manopt.status_summary(obj3)
        @test startswith(s5, "A primal dual semismooth Newton objective")
        @test contains(s5, "Λ")
    end
end
