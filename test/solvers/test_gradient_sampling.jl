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

    Random.seed!(23)
    m2 = gradient_sampling(
        M, f, grad_f, p0;
        return_state = true,
        debug = _debug_gradient_sampling ? [:Iteration, :Cost, " ", :subgradient_norm_tolerance, " ", :sampling_radius, " | ", :GradientNorm, " ", :Change, "\n", :Stop, 10] : [],
        record = [:Iteration, :Cost, RecordGradientNorm()]
    )

    @testset "documented constructor" begin
        # the `convex_hull_coeffs` default must not mention the static parameter `R`
        sd = GradientSamplingState(M)
        @test length(sd.convex_hull_coeffs) == 6
        @test eltype(sd.convex_hull_coeffs) === Float64
        sd7 = GradientSamplingState(M; sample_size = 7)
        @test length(sd7.convex_hull_coeffs) == 8
        # keywords of different number types are promoted instead of erroring
        si = GradientSamplingState(M; sampling_radius = 1)
        @test si.sampling_radius isa Float64
        @test eltype(si.convex_hull_coeffs) === Float64
        sf = GradientSamplingState(M; sampling_radius = 0.5f0)
        @test sf.sampling_radius isa Float64 # promoted against the other defaults
    end

    @testset "the step size is initialized" begin
        awn = AdaptiveWNGradient()(M)
        gss = GradientSamplingState(M; p = copy(M, p0), stepsize = awn)
        awn.weight = 5.0
        awn.count = 3
        initialize_solver!(DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f)), gss)
        @test awn.weight == awn.initial_bound
        @test awn.count == 0
    end

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
    @test isapprox(M, p2, p3; atol = 3.0e-3)
    @testset "a manifold whose points are numbers" begin
        Mc = Circle()
        datac = [-0.2, 0.0, 0.3]
        fc(N, q) = sum(distance.(Ref(N), Ref(q), datac) .^ 2) / (2 * length(datac))
        grad_fc(N, q) = sum(grad_distance.(Ref(N), datac, Ref(q))) / length(datac)
        Random.seed!(42)
        qc = gradient_sampling(
            Mc, fc, grad_fc, 0.5; stopping_criterion = StopAfterIteration(20)
        )
        @test qc isa Float64
        @test fc(Mc, qc) < fc(Mc, 0.5)
    end

    # the in-place gradient variant produces the same iterates
    grad_f!(M, X, p) = copyto!(M, X, p, grad_f(M, p))
    Random.seed!(23)
    p4 = gradient_sampling(M, f, grad_f!, p0; evaluation = InplaceEvaluation())
    @test isapprox(M, p2, p4)

    if _debug_gradient_sampling
        # For comparison
        m1 = gradient_descent(
            M, f, grad_f, p0;
            return_state = true,
            record = [:Iteration, :Cost, RecordGradientNorm()]
        )
        p1 = get_solver_result(m1)
        p2 = get_solver_result(m2)
        @info "p1 " p1 "with cost " f(M, p1)
        @info "p2 " p2 "with cost " f(M, p2)
        using CairoMakie
        fig, ax, plt = lines(get_record(m2, :Iteration, 1), get_record(m2, :Iteration, 2))
        lines!(ax, get_record(m1, :Iteration, 1), get_record(m1, :Iteration, 2))
        display(fig)
    end
end
