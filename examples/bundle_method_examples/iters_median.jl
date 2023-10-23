using Manopt, Manifolds, Random, QuadraticModels, RipQP, Plots

function test_dist_function(N)
    d = []
    # l = Int(1e2)
    M = N
    Random.seed!(manifold_dimension(N))
    data = [rand(M) for i in 1:100]

    F2(M, y) = sum(1 / (2 * length(data)) * distance.(Ref(M), Ref(y), data))
    gradF2(M, y) = sum(1 / (2 * length(data)) * grad_distance.(Ref(M), data, Ref(y), 1))

    m = rand(data)

    bundle_min = convex_bundle_method(
        M,
        F2,
        gradF2,
        m;
        record=[:Iteration, :Cost],
        return_state=true,
        # stopping_criterion=StopWhenBundleLess(1e-12),
        stopping_criterion=StopAfterIteration(100),
        # debug=[:Iteration, :Cost, "\n"],
    )

    subgradient_min = subgradient_method(
        M,
        F2,
        gradF2,
        m;
        stopping_criterion=StopAfterIteration(100),
        # stopping_criterion=StopWhenSubgradientNormLess(1e-12),
        record=[:Iteration, :Cost],
        return_state=true,
        # debug=[:Iteration, :Cost, "\n"],
    )
    return (
        get_record_action(bundle_min)[:Cost][end] -
        get_record_action(subgradient_min)[:Cost][end]
    ),
    get_record_action(bundle_min)[:Iteration],
    get_record_action(bundle_min)[:Cost],
    get_record_action(subgradient_min)[:Iteration],
    get_record_action(subgradient_min)[:Cost]
end

function plot_graphs_median(m::Int)
    a, x1b, y1b, x1s, y1s = test_max_function(Hyperbolic(m))
    b, x2b, y2b, x2s, y2s = test_max_function(SymmetricPositiveDefinite(m))

    p1b = plot(
        x1b,
        y1b;
        xaxis=:log,
        label="H^$(m), Bundle Method",
        xlabel="Iterations",
        ylabel="Cost",
        legend=:outertop,
    )

    p1s = plot(
        x1s,
        y1s;
        xaxis=:log,
        label="H^$(m), Subgradient Method",
        xlabel="Iterations",
        ylabel="Cost",
        legend=:outertop,
    )

    p2b = plot(
        x2b,
        y2b;
        xaxis=:log,
        label="SPD($(m^2)), Bundle Method",
        xlabel="Iterations",
        ylabel="Cost",
        legend=:outertop,
    )

    p2s = plot(
        x2s,
        y2s;
        xaxis=:log,
        label="SPD($(m^2)), Subgradient Method",
        xlabel="Iterations",
        ylabel="Cost",
        legend=:outertop,
    )

    p = plot(
        p1b,
        p1s,
        p2b,
        p2s;
        plot_title="Riemannian Distance",
        window_title="Numerical Example",
    )
    display(p)

    return a, b
end
