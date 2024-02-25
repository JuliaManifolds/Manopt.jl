using Manopt, Manifolds, Test, QuadraticModels, RipQP, ManifoldDiff

@testset "The Proximal Bundle Method" begin
    @testset "Basic Constructor tests" begin end
    @testset "A simple median run" begin
        M = Sphere(2)
        p1 = [1.0, 0.0, 0.0]
        p2 = 1 / sqrt(2) .* [1.0, 1.0, 0.0]
        p3 = 1 / sqrt(2) .* [1.0, 0.0, 1.0]
        data = [p1, p2, p3]
        f(M, p) = sum(1 / length(data) * distance.(Ref(M), Ref(p), data))
        function ∂f(M, p)
            return sum(
                1 / length(data) *
                ManifoldDiff.subgrad_distance.(Ref(M), data, Ref(p), 1; atol=1e-8),
            )
        end
        p0 = p1
        pbm_s = proximal_bundle_method(M, f, ∂f, p0; return_state=true)
        @test startswith(
            repr(pbm_s), "# Solver state for `Manopt.jl`s Proximal Bundle Method\n"
        )
        q = get_solver_result(pbm_s)
        # with default parameters for both median and proximal bundle, this is not very precise
        m = median(M, data)
        @test distance(M, q, m) < 2 * 1e-3
        # test accessors
        @test get_iterate(pbm_s) == q
        @test norm(M, q, get_subgradient(pbm_s)) < 1e-4
        # twst the other stopping criterion mode
        q2 = proximal_bundle_method(
            M,
            f,
            ∂f,
            p0;
            stopping_criterion=StopWhenLagrangeMultiplierLess([1e-8, 1e-8]; mode=:both),
        )
        @test distance(M, q2, m) < 2 * 1e-3
    end
end
