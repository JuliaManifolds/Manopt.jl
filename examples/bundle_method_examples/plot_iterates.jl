using Plots, Manopt, Manifolds, Random, QuadraticModels, RipQP

function plot_maxfunc_iterates(M)
    l = Int(1e2)
    Random.seed!(2)
    data = [rand(M) for j in 1:l]

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

    bundle_min = bundle_method(
        M,
        F3,
        subgradF3,
        data[1];
        diam=10.0,
        stopping_criterion=StopWhenSubgradientNormLess(1e-12),
        record=:Iterate,
        return_state=true,
    )
    bi_pts = get_record(bundle_min)
    # bi_pts = convert.(
    #     Ref(PoincareHalfSpacePoint),
    #     Manifolds._hyperbolize.(Ref(M), bi_pts)
    # )
    subgrad_min = subgradient_method(
        M,
        F3,
        subgradF3,
        data[1];
        stopping_criterion=StopWhenSubgradientNormLess(1e-1),
        record=:Iterate,
        return_state=true,
    )
    si_pts = get_record(subgrad_min)
    # si_pts = convert.(
    #     Ref(PoincareHalfSpacePoint),
    #     Manifolds._hyperbolize.(Ref(M), si_pts)
    # )
    fig = plot()
    plot!(fig, M, bi_pts; geodesic_interpolation=100)
    plot!(fig, M, bi_pts; label="Serious iterates", wireframe=false)
    plot!(fig, M, si_pts; geodesic_interpolation=100, wireframe=false)
    plot!(fig, M, si_pts; label="Subgrad iterates", wireframe=false)
    plot!(fig, M, data; label="Data points")
    return display(fig)
end
