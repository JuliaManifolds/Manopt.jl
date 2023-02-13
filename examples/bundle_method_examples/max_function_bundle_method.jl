using Manopt, Manifolds, Random, QuadraticModels, RipQP

function check_maxfunc(M)
    println(manifold_dimension(M))
    for i in 1:100
        l = Int(1e2)
        # Random.seed!(42)
        data = [rand(M; σ=0.4) for i in 1:l]

        F(M, y) = sum(1 / (2 * length(data)) * distance.(Ref(M), data, Ref(y)) .^ 2)
        gradF(M, y) = sum(1 / length(data) * grad_distance.(Ref(M), data, Ref(y)))
        F2(M, y) = sum(1 / (2 * length(data)) * distance.(Ref(M), Ref(y), data) .^ 4)
        gradF2(M, y) = sum(1 / (2 * length(data)) * grad_distance.(Ref(M), data, Ref(y), 4))

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

        # Find intersection point between F and F2
        G(M, y) = (F(M, y) - F2(M, y))^2
        m = NelderMead(M, G)
        #m = subgradient_method(M, G, gradG, rand(M))

        bundle_min = bundle_method(
            M,
            F3,
            subgradF3,
            m;
            stopping_criterion=StopAfterIteration(1000),
            # debug=[:Iteration, :Cost, "\n"],
        )
    end
end
