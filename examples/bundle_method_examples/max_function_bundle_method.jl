using Manopt, Manifolds, Random, QuadraticModels, RipQP
include("level_set_diameter.jl")
function check_maxfunc(M, tol=1e-8)
    println(M)
    # for i in 1:10
    # println("Test $i")
    l = Int(1e2)
    Random.seed!(22)
    data = [rand(M; σ=1.0) for j in 1:l]

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
    G(M, y) = (F(M, y) - F2(M, y))^2
    m = NelderMead(M, G)
    # m = subgradient_method(M, G, gradG, rand(M))

    println(check_gradient(M, F3, subgradF3, m))


    p0 = m

    # diam = level_set_diameter(M, F3, subgradF3, p0)
    # println("Level set diameter = $diam")

    bundle_min = bundle_method(
        M,
        F3,
        subgradF3,
        p0;
        δ=sqrt(2),
        diam=2.,#.8,
        stopping_criterion=StopWhenBundleLess(tol) | StopAfterIteration(5000),
        debug=[
            :Iteration,
            :Stop,
            (:Cost, "F(p): %1.16f "),
            (:ξ, "ξ: %1.16f "),
            (:ε, "ε: %1.16f "),
            (:diam, "diam: %1.16f "),
            :Stop,
            1000,
            "\n",
        ],
    )

    subgrad_min = subgradient_method(
        M,
        F3,
        subgradF3,
        p0;
        stopping_criterion=StopWhenSubgradientNormLess(sqrt(tol)) | StopAfterIteration(5000),
        debug=[
            :Iteration,
            (:Cost, "F(p): %1.16f"),
            :Stop,
            1000,
            "\n",
        ],
    )
    # prox_min = p0
    try
        global prox_min = prox_bundle_method(
            M,
            F3,
            subgradF3,
            p0;
            δ=.0,
            μ=1.,
            stopping_criterion=StopWhenProxBundleLess(tol) | StopAfterIteration(5000),
            debug=[
                :Iteration,
                :Stop,
                (:Cost, "F(p): %1.16f "),
                (:ν, "ν: %1.16f "),
                (:c, "c: %1.16f "),
                (:μ, "μ: %1.16f "),
                (:η, "η: %1.16f "),
                :Stop,
                1000,
                "\n",
            ],
        )
    catch y
        println("The prox_bundle_method got stuck at the subsolver level.")
    end
    println("Distance between p0 and bundle_min: $(distance(M, bundle_min, p0))")
    println("Distance between bundle_min and subgrad_min: $(distance(M, bundle_min, subgrad_min))")
    println(
        "$(F3(M, bundle_min) < F3(M, subgrad_min) ? "F3(bundle_min) < F3(subgrad_min)" : "F3(bundle_min) ≥ F3(subgrad_min)")",
        )
    println(
        "    |F3(bundle_min) - F3(subgrad_min)| = $(abs(F3(M, bundle_min) - F3(M, subgrad_min)))",
        )
    println("Distance between bundle_min and prox_min: $(distance(M, bundle_min, subgrad_min))")
    try
        println(
            "$(F3(M, prox_min) < F3(M, subgrad_min) ? "F3(prox_min) < F3(subgrad_min)" : "F3(prox_min) ≥ F3(subgrad_min)")",
        )
        println(
            "    |F3(prox_min) - F3(subgrad_min)| = $(abs(F3(M, prox_min) - F3(M, subgrad_min)))",
        )
        println(
            "$(F3(M, prox_min) < F3(M, bundle_min) ? "F3(prox_min) < F3(bundle_min)" : "F3(prox_min) ≥ F3(bundle_min)")",
        )
        println(
        "    |F3(prox_min) - F3(bundle_min)| = $(abs(F3(M, prox_min) - F3(M, bundle_min))) \n\n",
        )
    catch y
        println("prox_min is not defined")
    end
end

check_maxfunc(SymmetricPositiveDefinite(3))
# check_maxfunc(SymmetricPositiveDefinite(7)) # prox_bundle_method yields a lower minimum (by 1e-11), but takes 50 more iterations
# check_maxfunc(Hyperbolic(2)) # bundle_method is by far better, whereas for bigger dimensions the prox_bundle_method errors at the subsolver level: AssertionError: all(pt0.x .> fd.lvar) && all(pt0.x .< fd.uvar)

# check_maxfunc(SymmetricPositiveDefinite(60)) # bundle_method is much faster: I'm still waiting for the prox_bundle_method to stop
# check_maxfunc(Hyperbolic(37))
