using Manopt, Manifolds, Random, QuadraticModels, RipQP, Plots

function test_max_function(N)
    d = []
    # l = Int(1e2)
    for j in 1:10
        M = N
        Random.seed!(manifold_dimension(N) * j)
        data = [rand(M; σ=0.4) for i in 1:100]

        F(M, y) = sum(1 / (2 * length(data)) * distance.(Ref(M), data, Ref(y)) .^ 2)
        gradF(M, y) = sum(1 / length(data) * grad_distance.(Ref(M), data, Ref(y)))
        F2(M, y) = sum(1 / (2 * length(data)) * distance.(Ref(M), Ref(y), data))
        gradF2(M, y) = sum(1 / (2 * length(data)) * grad_distance.(Ref(M), data, Ref(y), 1))

        F3(M, y) = max(F(M, y), F2(M, y))
        function gradF3(M, y)
            if F3(M, y) == F(M, y) && F3(M, y) != F2(M, y)
                return gradF(M, y)
            elseif F3(M, y) == F2(M, y) && F3(M, y) != F(M, y)
                return gradF2(M, y)
            else
                r = rand()
                return r * gradF(M, y) + (1 - r) * gradF2(M, y)
            end
        end

        # Find intersection point between F and F2
        G(M, y) = (F(M, y) - F2(M, y))^2
        m = NelderMead(M, G)

        bundle_min = bundle_method(
            M,
            F3,
            gradF3,
            m;
            stopping_criterion=StopAfterIteration(10),
            # debug=[:Iteration, :Cost, "\n"],
        )

        subgradient_min = subgradient_method(
            M,
            F3,
            gradF3,
            m;
            stopping_criterion=StopAfterIteration(10),
            # debug=[:Iteration, :Cost, "\n"],
        )

        push!(d, distance(M, bundle_min, subgradient_min))
    end
    return mean(d)
end

function plot_graphs(m::Int)
    t1 = [test_max_function(Hyperbolic(i)) for i in 3:m]
    t2 = [test_max_function(SymmetricPositiveDefinite(i)) for i in 3:m]

    p1 = plot(
        [3:m],
        [t1 t2];
        title="Mean distance between bundle and subgradient methods",
        label=["Hyperbolic Space" "Sym Pos Def Matirces"],
        xlabel="Dimension",
        ylabel="Mean distance (10 loops)",
        legend=:outertop,
    )

    display(plot(p1))
    return nothing
end
