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
        p = get_solver_result(pbm_s)
        @test distance(M, p, median(M, data)) < 2 * 1e-3 #with default params this is not very precise
    end
end
