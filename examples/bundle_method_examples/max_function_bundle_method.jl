using Manopt, Manifolds, Random, QuadraticModels, RipQP

function check_maxfunc(M)
    println(manifold_dimension(M))
    for i in 1:10
        println("Test $i")
        l = Int(1e2)
        Random.seed!(i+30)
        data = [rand(M) for j in 1:l]

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
        # G(M, y) = (F(M, y) - F2(M, y))^2
        # m = NelderMead(M, G)
        #m = subgradient_method(M, G, gradG, rand(M))

        bundle_min = bundle_method(
            M,
            F3,
            subgradF3,
            data[i];
            diam = 10.0,
            #stopping_criterion=StopAfterIteration(10),
            debug=["    ", :Iteration, :Cost, "\n", 1],
        )

        subgrad_min = subgradient_method(
            M,
            F3,
            subgradF3,
            data[i];
            # stopping_criterion=StopWhenSubgradientNormLess(1e-8),
            debug=["    ", :Iteration, :Cost, "\n", 1000],
        )

        println("Distance between minima: $(distance(M, bundle_min, subgrad_min))")
        println(
            "$(F3(M, bundle_min) < F3(M, subgrad_min) ? "F3(bundle_min) < F3(subgrad_min)" : "F3(bundle_min) ≥ F3(subgrad_min)")",
        )
        println(
            "    |F3(bundle_min) - F3(subgrad_min)| = $(abs(F3(M, bundle_min) - F3(M, subgrad_min)))",
        )
    end
    # return bundle_min
end
