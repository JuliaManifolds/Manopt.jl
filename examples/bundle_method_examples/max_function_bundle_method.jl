using Manopt, Manifolds, Random, QuadraticModels, RipQP
include("level_set_diameter.jl")
function check_maxfunc(M)
    println(manifold_dimension(M))
    # for i in 1:10
        # println("Test $i")
        l = Int(1e2)
        Random.seed!(22)
        data = [rand(M; σ = 1.) for j in 1:l]

        F(M, y) = sum(1 / (2 * length(data)) * distance.(Ref(M), Ref(y), data))
        gradF(M, y) = sum(1 / (2 * length(data)) * grad_distance.(Ref(M), data, Ref(y), 1))
        F2(M, y) = sum(1 / (2 * length(data)) * distance.(Ref(M), data, Ref(y)) .^ 2)
        gradF2(M, y) = sum(1 / length(data) * grad_distance.(Ref(M), data, Ref(y)))

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
        # m = subgradient_method(M, G, gradG, rand(M))

        p0 = data[1]
        
        diam = 0.1 #level_set_diameter(M, F3, subgradF3, p0)
        println("Level set diameter = $diam")

        bundle_min = bundle_method(
            M,
            F3,
            subgradF3,
            p0;
            diam = diam,
            stopping_criterion=StopWhenBundleLess(1e-10)|StopAfterIteration(5000),
            debug=["    ", :Iteration, (:Cost,"F(p): %1.13e"),"\n", :Stop, 100],
        )

        subgrad_min = subgradient_method(
            M,
            F3,
            subgradF3,
            p0;
            stopping_criterion=StopWhenSubgradientNormLess(1e-8)|StopAfterIteration(5000),
            debug=["    ", :Iteration, (:Cost,"F(p): %1.9e"), "\n", :Stop, 1000],
        )

        println("Distance between minima: $(distance(M, bundle_min, subgrad_min))")
        println(
            "$(F3(M, bundle_min) < F3(M, subgrad_min) ? "F3(bundle_min) < F3(subgrad_min)" : "F3(bundle_min) ≥ F3(subgrad_min)")",
        )
        println(
            "    |F3(bundle_min) - F3(subgrad_min)| = $(abs(F3(M, bundle_min) - F3(M, subgrad_min)))",
        )
    # end
end

check_maxfunc(SymmetricPositiveDefinite(2))