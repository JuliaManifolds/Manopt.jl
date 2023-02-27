using Manopt, Manifolds, Random, QuadraticModels, RipQP, Plots, LaTeXStrings

function test_max_function(N,m_par)
    d = []
    # l = Int(1e2)
    M = N
    Random.seed!(manifold_dimension(N))
    data = [rand(M) for i in 1:100]

    F(M, y) = sum(1 / (2 * length(data)) * distance.(Ref(M), data, Ref(y)) .^ 2)
    gradF(M, y) = sum(1 / length(data) * grad_distance.(Ref(M), data, Ref(y)))
    F2(M, y) = sum(1 / (2 * length(data)) * distance.(Ref(M), Ref(y), data))
    gradF2(M, y) = sum(1 / (2 * length(data)) * grad_distance.(Ref(M), data, Ref(y), 1))

    F3(M, y) = max(F(M, y), F2(M, y))
    function subgradF3(M, y)
        if isapprox(F3(M, y), F(M, y)) && !isapprox(F3(M, y), F2(M, y))
            return gradF(M, y)
        elseif isapprox(F3(M, y), F2(M, y)) && !isapprox(F3(M, y), F(M, y))
            return gradF2(M, y)
        else
            r = rand()
            return r * gradF(M, y) + (1 - r) * gradF2(M, y)
        end
    end

    p = rand(data)

    bundle_min = bundle_method(
        M,
        F3,
        subgradF3,
        p;
        m=m_par,
        record=[:Iteration, :Cost],
        return_state=true,
        stopping_criterion=StopAfterIteration(100),
        # debug=[:Iteration, :Cost, "\n"],
    )

    subgradient_min = subgradient_method(
        M,
        F3,
        subgradF3,
        p;
        stopping_criterion=StopAfterIteration(100),
        # stopping_criterion=StopWhenChangeLess(1e-8),
        record=[:Iteration, :Cost],
        return_state=true,
        # debug=[:Iteration, :Cost, "\n"],
    )
    return get_record_action(bundle_min)[:Cost][end], get_record_action(subgradient_min)[:Cost][end], get_record_action(bundle_min)[:Iteration], get_record_action(bundle_min)[:Cost], get_record_action(subgradient_min)[:Iteration], get_record_action(subgradient_min)[:Cost]
end

function plot_graphs_max(n::Int)

    fig = plot() # produces an empty plot

    # Plot bundle methods with variable m-parameter
    # for m_par in 0.3:0.1:0.4
    #     a, x1b, y1b, x1s, y1s = test_max_function(Hyperbolic(n), m_par)
    #     # b, x2b, y2b, x2s, y2s = test_max_function(SymmetricPositiveDefinite(m))
    #     plot!(fig, x1b, y1b; xlabel="Iterations", ylabel="Cost", label=L"m = %$m_par") # the loop fills in the plot with this
    # end

    # Plot bundle method and subgradient
    mb, ms, x1b, y1b, x1s, y1s = test_max_function(Hyperbolic(n), 0.0125)
    plot!(fig, x1b, y1b; xlabel="Iterations", ylabel="Cost", label="Bundle Method")
    hline!([mb]; linestyle=:dash, label="$mb")
    plot!(fig, x1s, y1s; xlabel="Iterations", ylabel="Cost", label="Subgradient Method")
    hline!([ms]; linestyle=:dash, label="$ms")

    # p1 = plot(
    #     [x1b x1s],
    #     [y1b y1s];
    #     label=["Bundle Method" "Subgradient Method"],
    #     xlabel="Iterations",
    #     ylabel="Cost",
    #     legend=:outertop,
    # )

    # p1s = plot(
    #     x1s,
    #     y1s;
    #     label="H^$(m), Subgradient Method",
    #     xlabel="Iterations",
    #     ylabel="Cost",
    #     legend=:outertop,
    # )

    # p2b = plot(
    #     x2b,
    #     y2b;
    #     label="SPD($(m^2)), Bundle Method",
    #     xlabel="Iterations",
    #     ylabel="Cost",
    #     legend=:outertop,
    # )

    # p2s = plot(
    #     x2s,
    #     y2s;
    #     label="SPD($(m^2)), Subgradient Method",
    #     xlabel="Iterations",
    #     ylabel="Cost",
    #     legend=:outertop,
    # )

    # p = plot(p1b, p1s)#; plot_title = "max{d,d^2}", window_title="Numerical Example")
    display(fig)
    savefig(fig,"bundle_min_sub_min.png")
    return mb, ms, mb-ms
end
