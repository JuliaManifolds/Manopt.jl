using FiniteDifferences, ManifoldDiff, Manopt, Manifolds, Random, QuadraticModels, RipQP

function check_maxfunc(M)
    println(manifold_dimension(M))
    # for i in 1:10
        # println("Test $i")
        l = Int(1e4)
        Random.seed!(41)
        data = [rand(M; σ = 2.) for j in 1:l]

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
        r_backend = ManifoldDiff.TangentDiffBackend(ManifoldDiff.FiniteDifferencesBackend())
        #=
        # Estimation of the diameter of the level set of F3 at p0 
        N = ProductManifold(M, M) #M^2 
        prod_data = [rand(N) for j in 1:10000]
        G(N, q) = -sum(1 / (2 * length(prod_data)) * distance.(Ref(N), Ref(q), prod_data).^2)
        G(q) = -sum(1 / (2 * length(prod_data)) * distance.(Ref(N), Ref(q), prod_data).^2)
        # gradG(N, q) = -sum(1 / length(prod_data) * grad_distance.(Ref(N), prod_data, Ref(q)))
        gradG(N, q) = ManifoldDiff.gradient(N, G, q, r_backend)
        # q[N, 1]
        H1(q) = F3(M, get_component(N, q, 1)) - F3(M, p0)
        gradH1(N,q) = ManifoldDiff.gradient(N, H1, q, r_backend)
        H2(q) = F3(M, get_component(N, q, 2)) - F3(M, p0)
        gradH2(N,q) = ManifoldDiff.gradient(N, H2, q, r_backend)
        H(N, q) = [H1(q), H2(q)]
        gradH(N, q) = [gradH1(N, q), gradH2(N, q)]

        initial_product_point = prod_data[1]
        pts = augmented_Lagrangian_method(
            N, G, gradG, initial_product_point; 
            g=H, grad_g=gradH, 
            record=[:Iterate, :Cost],
            return_state=true,
            debug=[:Iteration, :Cost, :Stop, "\n"],
            stopping_criterion=StopWhenAny(StopWhenCostNan(), StopAfterIteration(300) | (
                StopWhenSmallerOrEqual(:ϵ, 1e-6) & StopWhenChangeLess(1e-6)
            )),
        )
        # Get the iterate corresponding to the last non-NaN value of the cost function
        if length(get_record_action(pts)[:Iterate]) > 1 
            p_diam = get_record_action(pts)[:Iterate][end-1]
        elseif !isnan(get_record_action(pts)[:Cost][1])
            p_diam = get_record_action(pts)[:Iterate][1]
        else
            p_diam = initial_product_point
        end
        =#
        diam = 1e5# -G(N, p_diam)  
        println("Level set diameter = $diam")

        bundle_min = bundle_method(
            M,
            F3,
            subgradF3,
            p0;
            diam = diam,
            stopping_criterion=StopWhenSubgradientNormLess(1e-11),
            debug=["    ", :Iteration, (:Cost,"F(p): %1.9e"),"\n", :Stop, 1],
        )

        subgrad_min = subgradient_method(
            M,
            F3,
            subgradF3,
            p0;
            stopping_criterion=StopWhenSubgradientNormLess(1e-11),
            debug=["    ", :Iteration, (:Cost,"F(p): %1.9e"), "\n", :Stop, 1],
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
