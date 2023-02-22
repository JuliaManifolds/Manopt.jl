using Manopt, Manifolds, Random, QuadraticModels, RipQP, Plots

function test_max_function(N)
    d = []
    # l = Int(1e2)
    M = N
    Random.seed!(manifold_dimension(N))
    data = [rand(M; σ=0.4) for i in 1:100]

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

    m = rand(data)

    bundle_min = bundle_method(
        M,
        F3,
        subgradF3,
        m;
        record=[:Iteration, :Cost],
        return_state=true,
        # stopping_criterion=StopAfterIteration(10),
        # debug=[:Iteration, :Cost, "\n"],
    )

    subgradient_min = subgradient_method(
        M,
        F3,
        subgradF3,
        m;
        stopping_criterion=StopWhenChangeLess(1e-8),
        record=[:Iteration, :Cost],
        return_state=true,
        # debug=[:Iteration, :Cost, "\n"],
    )
    return get_record_action(bundle_min)[:Iteration], get_record_action(bundle_min)[:Cost], get_record_action(subgradient_min)[:Iteration], get_record_action(subgradient_min)[:Cost]
end

function plot_graphs(m::Int)
    x1b, y1b, x1s, y1s = test_max_function(Hyperbolic(m))
    x2b, y2b, x2s, y2s = test_max_function(SymmetricPositiveDefinite(m))

    p1b = plot(
        x1b,
        y1b;
        title="f = max{d,d^2} bundle and subgradient methods",
        label="Hyperbolic Space, Bundle",
        xlabel="Iteration",
        ylabel="Cost",
        legend=:outertop,
    )

    p1s = plot(
        x1s,
        y1s;
        title="f = max{d,d^2} bundle and subgradient methods",
        label="Hyperbolic Space, Bundle",
        xlabel="Iteration",
        ylabel="Cost",
        legend=:outertop,
    )

    p2b = plot(
        x2b,
        y2b;
        title="f = max{d,d^2} bundle and subgradient methods",
        label="Sym Pos Def Matrices, Subgradient",
        xlabel="Iteration",
        ylabel="Cost",
        legend=:outertop,
    )

    p2s = plot(
        x2s,
        y2s;
        title="",
        label="Sym Pos Def Matrices, Subgradient",
        xlabel="Iteration",
        ylabel="Cost",
        legend=:outertop,
    )

    display(plot(p1b, p1s, p2b, p2s))
    return nothing
end
