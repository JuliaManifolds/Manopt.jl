using Manopt, Manifolds, Random, QuadraticModels, RipQP, Plots

function compare_times(manifold_name, n::Int)
    t = Float64[]
    s = Float64[]
    for j in 3:n
        M = manifold_name(j)
        Random.seed!(42)
        data = [rand(M; σ=0.4) for i in 1:100]

        F(M, y) = sum(1 / (2 * length(data)) * distance.(Ref(M), data, Ref(y)) .^ 2)
        gradF(M, y) = sum(1 / length(data) * grad_distance.(Ref(M), data, Ref(y)))

        F2(M, y) = sum(1 / (2 * length(data)) * distance.(Ref(M), Ref(y), data))
        gradF2(M, y) = sum(1 / (2 * length(data)) * grad_distance.(Ref(M), data, Ref(y), 1))

        F3(M, y) = max(F(M, y), F2(M, y))
        function gradF3(M, y)
            if F3(M, y) == F(M, y)
                return gradF(M, y)
            else
                return gradF2(M, y)
            end
        end

        push!(t, @timed(bundle_method(
            M,
            F3,
            gradF3,
            data[1];
            # m = 0.0125,
            # tol = 1e-8,
            # stopping_criterion=StopWhenAny(
            #     StopAfterIteration(20), StopWhenChangeLess(1e-8)
            # ),
            # debug=[:Iteration, :Cost, "\n", 10],
        )).time)

        push!(s, @timed(subgradient_method(
            M,
            F3,
            gradF3,
            data[1];
            # stopping_criterion=StopWhenAny(
            #     StopAfterIteration(20), StopWhenChangeLess(1e-8)
            # ),
            # debug=[:Iteration, :Cost, "\n", 10],
        )).time)
    end
    return t, s
end

function plot_time_graphs(m::Int)
    t1, s1 = compare_times(Hyperbolic, m)
    t2, s2 = compare_times(SymmetricPositiveDefinite, m)

    p1 = plot(
        [3:m],
        [t1 t2];
        title="Bundle Method",
        label=["Hyperbolic Space" "Symmetric Pos Def Matirces"],
        xlabel="Dimension",
        ylabel="Time",
    )
    p2 = plot(
        [3:m],
        [s1 s2];
        title="Subgradient Method",
        label=["Hyperbolic Space" "Symmetric Pos Def Matirces"],
        xlabel="Dimension",
        ylabel="Time",
    )
    #display(plot(p1, p2))
    return nothing
end
