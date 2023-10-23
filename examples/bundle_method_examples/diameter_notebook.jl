### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ aa77ff42-5a4c-41f1-b432-954cd428e1e4
using Pkg;
Pkg.activate();

# ╔═╡ 196d1c38-3f90-41ea-90d5-292b87957eeb
using Random, QuadraticModels, RipQP, Manifolds, Manopt

# ╔═╡ 0910a6fb-0dbc-4322-bc9b-4880aca7574e
begin
    M = SymmetricPositiveDefinite(2)
    Random.seed!(42)
    l = Int(1e2)
    data = [rand(M; σ=1.0) for j in 1:l]
end

# ╔═╡ f65b0da1-ee29-47f1-ad55-beed2415f438
p0 = data[1]

# ╔═╡ 8eaf0064-c19d-46cf-8fa4-83b9d070ac0d
begin
    F(M, q) = sum(1 / (2 * length(data)) * distance.(Ref(M), Ref(q), data))
    F2(M, q) = sum(1 / (2 * length(data)) * distance.(Ref(M), data, Ref(q)) .^ 2)
    F3(M, q) = max(F(M, q), F2(M, q))
end

# ╔═╡ 06ab01c9-da04-453e-b669-6cfadaeb7337
begin
    gradF(M, q) = sum(1 / (2 * length(data)) * grad_distance.(Ref(M), data, Ref(q), 1))
    gradF2(M, q) = sum(1 / length(data) * grad_distance.(Ref(M), data, Ref(q)))
    function subgradF3(M, q)
        if isapprox(F3(M, q), F(M, q)) && !isapprox(F3(M, q), F2(M, q))
            return gradF(M, q)
        elseif isapprox(F3(M, q), F2(M, q)) && !isapprox(F3(M, q), F(M, q))
            return gradF2(M, q)
        else
            r = rand()
            return r * gradF(M, q) + (1 - r) * gradF2(M, q)
        end
    end
end

# ╔═╡ 6d74fa80-c53f-411a-93c0-0c4418a56e26
function level_set_diameter(
    M,
    f,
    ∂f,
    p0;
    sub_solver::Function=augmented_Lagrangian_method,
    iter_cap::Int=60,
    random_seed::Int=42,
    distr_var::Real=1.0,
    show_err::Bool=false,
)
    N = M^2
    Random.seed!(random_seed)
    prod_data = [rand(N; σ=distr_var) for j in 1:10000]
    G(N, q) = -sum(1 / (2 * length(prod_data)) * distance.(Ref(N), Ref(q), prod_data) .^ 2)
    gradG(N, q) = -sum(1 / length(prod_data) * grad_distance.(Ref(N), prod_data, Ref(q)))
    H1(N, q) = f(M, q[N, 1]) - f(M, p0)
    function gradH1(N, q)
        r = rand(N)
        set_component!(N, r, ∂f(M, q[N, 1]), 1)
        set_component!(N, r, zeros(representation_size(M)), 2)
        return r
    end
    #gradH1(N, q) = hcat(∂f(M, q[N, 1]), zeros(representation_size(M)))
    H2(N, q) = f(M, q[N, 2]) - f(M, p0)
    function gradH2(N, q)
        r = rand(N)
        set_component!(N, r, zeros(representation_size(M)), 1)
        set_component!(N, r, ∂f(M, q[N, 2]), 2)
        return r
    end
    #gradH2(N, q) = hcat(zeros(representation_size(M)), ∂f(M, q[N, 2]))
    H(N, q) = [H1(N, q), H2(N, q)]
    gradH(N, q) = [gradH1(N, q), gradH2(N, q)]
    initial_product_point = prod_data[1]
    pts = 0.0
    err = ""
    try
        pts = sub_solver(
            N,
            G,
            gradG,
            initial_product_point;
            g=H,
            grad_g=gradH,
            record=[:Iterate, :Cost],
            return_state=true,
            # debug=[:Iteration, :Cost, :Stop, "\n"],
            stopping_criterion=StopWhenAny(
                StopWhenCostNan(),
                StopWhenIterNan(),
                StopAfterIteration(300) |
                (StopWhenSmallerOrEqual(:ϵ, 1e-6) & StopWhenChangeLess(1e-6)),
            ),
        )
    catch e
        err = "$e. Ran ALM with a $iter_cap iterations cap."
    finally
        pts = sub_solver(
            N,
            G,
            gradG,
            initial_product_point;
            g=H,
            grad_g=gradH,
            record=[:Iterate, :Cost],
            return_state=true,
            # debug=[:Iteration, :Cost, :Stop, "\n"],
            stopping_criterion=StopWhenAny(
                StopWhenCostNan(),
                StopWhenIterNan(),
                StopAfterIteration(iter_cap) |
                (StopWhenSmallerOrEqual(:ϵ, 1e-6) & StopWhenChangeLess(1e-6)),
            ),
        )
    end
    if show_err
        println("\n $err")
    end
    # Get the iterate corresponding to the last non-NaN value of the cost function
    if length(get_record_action(pts)[:Iterate]) > 1
        p_diam = get_record_action(pts)[:Iterate][end - 1]
    elseif !isnan(get_record_action(pts)[:Cost][1])
        p_diam = get_record_action(pts)[:Iterate][1]
    else
        p_diam = initial_product_point
    end
    return -G(N, p_diam)
end

# ╔═╡ 90d0ea98-1cd9-4d90-8381-8d1a9243bd56
diam = level_set_diameter(M, F3, subgradF3, p0)

# ╔═╡ 5f92e023-7360-479d-8841-7ea1ebfda6e3
begin
    println("Bundle method:")
    bundle_min = convex_bundle_method(
        M,
        F3,
        subgradF3,
        p0;
        diam=diam,
        stopping_criterion=StopWhenBundleLess(1e-6) | StopAfterIteration(5000),
        debug=["    ", :Iteration, (:Cost, "F(p): %1.9e"), "\n", :Stop, 5],
    )
end

# ╔═╡ 89f0d2b2-9015-4627-8dfe-d7f92ea9f105
begin
    println("Subgradient method:")
    subgrad_min = subgradient_method(
        M,
        F3,
        subgradF3,
        p0;
        stopping_criterion=StopWhenSubgradientNormLess(1e-6) | StopAfterIteration(5000),
        debug=["    ", :Iteration, (:Cost, "F(p): %1.9e"), "\n", :Stop, 5],
    )
end

# ╔═╡ 008c4811-4980-4dd9-b8eb-6b16cd995d04
begin
    println("Distance between minima: $(distance(M, bundle_min, subgrad_min))")
    println(
        "$(F3(M, bundle_min) < F3(M, subgrad_min) ? "F3(bundle_min) < F3(subgrad_min)" : "F3(bundle_min) ≥ F3(subgrad_min)")",
    )
    println(
        "    |F3(bundle_min) - F3(subgrad_min)| = $(abs(F3(M, bundle_min) - F3(M, subgrad_min)))",
    )
end

# ╔═╡ Cell order:
# ╠═aa77ff42-5a4c-41f1-b432-954cd428e1e4
# ╠═196d1c38-3f90-41ea-90d5-292b87957eeb
# ╠═0910a6fb-0dbc-4322-bc9b-4880aca7574e
# ╠═f65b0da1-ee29-47f1-ad55-beed2415f438
# ╠═8eaf0064-c19d-46cf-8fa4-83b9d070ac0d
# ╠═06ab01c9-da04-453e-b669-6cfadaeb7337
# ╠═6d74fa80-c53f-411a-93c0-0c4418a56e26
# ╠═90d0ea98-1cd9-4d90-8381-8d1a9243bd56
# ╟─5f92e023-7360-479d-8841-7ea1ebfda6e3
# ╟─89f0d2b2-9015-4627-8dfe-d7f92ea9f105
# ╠═008c4811-4980-4dd9-b8eb-6b16cd995d04
