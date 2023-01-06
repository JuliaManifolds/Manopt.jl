using Manopt, Manifolds, Random, QuadraticModels, RipQP, Plots

function time_bundle(man_name)
    t = []
    for n in 3:100
        M = man_name(n)
        Random.seed!(n)
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

        push!(
            t,
            @timed(
                bundle_method(
                    M,
                    F3,
                    gradF3,
                    data[1];
                    m = 0.125,
                    tol = 1e-8,
                    stopping_criterion=StopWhenAny(
                        StopAfterIteration(20), StopWhenChangeLess(1e-8)
                    ),
                    #debug=[:Iteration, :Cost, "\n", 10],
                )
            ).time,
        )
    end
    return t
end

t1 = time_bundle(Hyperbolic)
t2 = time_bundle(SymmetricPositiveDefinite)

display(plot([3:100], t1, t2; title="", labels = ["Hyperbolic Space", "Symmetric Pos Def Matirces"], xlabel = "Dimension", ylabel = "Time"))
