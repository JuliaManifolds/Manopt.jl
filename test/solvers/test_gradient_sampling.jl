using Manifolds, Manopt, QuadraticModels, Random, RipQP, Test
using ManifoldDiff: grad_distance, prox_distance

_debug_gradient_sampling = false

@testset "Gradient Sampling Algorithm" begin

    # Adapted from Ole Gunnar for now
    d = 100
    σ = π / 8
    M = Manifolds.Sphere(2)
    p = 1 / sqrt(2) * [1.0, 0.0, 1.0]
    # Generate random points "around" p:
    Random.seed!(42)
    data = [exp(M, p, σ * rand(M; vector_at = p)) for i in 1:d]
    p0 = data[1]

    # Define riemannian center of mass:
    f(M, p) = sum(1 / (2 * d) * distance.(Ref(M), Ref(p), data) .^ 2)
    grad_f(M, p) = sum(1 / d * grad_distance.(Ref(M), data, Ref(p)))

    # For comparison
    m1 = gradient_descent(
        M, f, grad_f, p0;
        return_state = true,
        record = [:Iteration, :Cost, RecordGradientNorm()]
    )

    Random.seed!(23)
    m2 = gradient_sampling(
        M, f, grad_f, p0;
        return_state = true,
        debug = _debug_gradient_sampling ? [:Iteration, :Cost, " ", :subgradient_norm_tolerance, " ", :sampling_radius, " | ", :GradientNorm, " ", :Change, "\n", :Stop, 10] : [],
        record = [:Iteration, :Cost, RecordGradientNorm()]
    )

    s2 = get_state(m2, true)
    @test startswith(repr(s2), "GradientSamplingState(; ")
    @test startswith(Manopt.status_summary(s2), "# Solver state for `Manopt.jl`s Gradient Sampling Algorithm")

    p2 = get_solver_result(s2)
    @test f(M, p2) < f(M, p0)

    p3 = copy(M, p0)
    Random.seed!(23)
    gradient_sampling!(
        M, f, grad_f, p3;
        sampling_radius = 0.1,
        subgradient_norm_tolerance = 0.02,
        sub_problem = gradient_sampling_subsolver,
        sub_state = AllocatingEvaluation(),
    )
    # The parameters of this run are chosen so that reduction is necessary,
    # they hence to not work that well and we end up a bit further away.
    @test isapprox(M, p2, p3; atol = 2.0e-3)

    if _debug_gradient_sampling
        p1 = get_solver_result(m1)
        p2 = get_solver_result(m2)
        @info "p1 " p1 "with cost " f(M, p1)
        @info "p2 " p2 "with cost " f(M, p2)
        using CairoMakie
        fig, ax, plt = lines(get_record(m2, :Iteration, 1), get_record(m2, :Iteration, 2))
        lines!(ax, get_record(m1, :Iteration, 1), get_record(m1, :Iteration, 2))
        fig
    end
end
