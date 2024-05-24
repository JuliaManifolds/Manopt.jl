### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# ╔═╡ 0005bf5f-f7d2-4cbd-ad2f-0e3b2b958f2f
using Pkg; Pkg.activate() # since we need the extend-DR branch

# ╔═╡ d6ca8ff8-b46e-4dea-8d43-e8a4c7a0f677
using Manifolds, Manopt, ManoptExamples

# ╔═╡ 3653045b-2081-484b-8fc4-d7397b6fc28d
using NamedColors, Plots, BenchmarkTools

# ╔═╡ c9d59b34-c7e1-49b0-9396-44ffbd824db2
begin
paul_tol = load_paul_tol()
indigo = paul_tol["mutedindigo"]
green = paul_tol["mutedgreen"]
sand = paul_tol["mutedsand"]
olive = paul_tol["mutedolive"]
teal = paul_tol["mutedteal"]
wine = paul_tol["mutedwine"]
grey = paul_tol["mutedgrey"]
end;

# ╔═╡ 5e98f2a5-c369-427c-98c2-7ffcbc716622
@doc raw"""
	prox_distance_C(M, λ, q, p)

Compute the proximal map of the distance function to a convex set ``C``,
that is with

```
\operatorname{proj_C}(p) = \argmin_{q \in C} d_{mathcal M}(p,q)
\quad\text{ define }
d_C(p) = d(\operatorname{proj}_C(p))
```

Then the proximal map of ``d_C`` is given by

```math
\operatorname{prox}_{λd_C(p)} = \begin{cases}
	γ(\frac{λ}{d_C(p)}; p, \operatorname{proj}_C(p)) & \text{ if } λ < d_C(p)\\
	\operatorname{proj}_C(p) & \text{ if } d_C(p)  ≤ d_C(p)
\end{cases}
```

# Input
* `M` a manifold
* `λ` the proximal parameter
* `q` the projection of `p` onto ``C``
* `p` the argument of the proximal map
"""
function prox_C(M, λ, q, p)
	d = distance(M, p, q)
	(λ < d) && (return shortest_geodesic(M, p, q, λ/d))
	return copy(M,q)
end	

# ╔═╡ f42053fb-b168-4f0d-9847-9a0bff850e72
function prox_C!(M, s, λ, proj_c, p)
	d = distance(M, p, q)
	(λ < d) && (return shortest_geodesic!(M, s, p, q, λ/d))
	return copyto!(M, s, q)	
end

# ╔═╡ ee853d97-49e0-48d5-8ba4-56488bf60200
struct GeodesicBall{P}
	center::P
	radius::Float64
end

# ╔═╡ 841f7d4c-35e7-41aa-9d0e-1398b8401e12
function project(M::AbstractManifold, B::GeodesicBall, p)
	d = distance(M, p, B.center)
	X = log(M, B.center, p)
	return exp(M, B.center, B.radius/d .* X)
end

# ╔═╡ dc73a669-f12a-4944-bbd0-b87ded61a933
M = PositiveNumbers()^2

# ╔═╡ 9d6359fc-2e65-4ff8-b0d0-d72641ece757
C = GeodesicBall([35.0, 35.0], 0.4)

# ╔═╡ 1f323db3-7698-4c20-a66b-ad2e56ca7b0a
Ck = [GeodesicBall(ci, ri) for (ci, ri) in [
	([15.0,15.0], 0.4),
	([65.0,65.0], 0.4),
	([10.0,60.0], 0.4),
	([60.0,10.0], 0.4),
]]

# ╔═╡ 21560a33-f639-493e-8ccb-393d5543cf75
function cost(N, p)
	sum = 0.0
	for i = 1:length(Ck)
		sum += distance(N.manifold, p[N,i], project(N.manifold, Ck[i], p[N,i]))
	end
	#if distance(N.manifold, p[N,1], project(N.manifold, C, p[N,1])) > 1e-14
	#	return Inf
	#end
	return sum
end

# ╔═╡ 038914db-3463-43a8-abc3-4ccd7d499f18
proxes = [ [ (M, λ, p) -> prox_C(M, λ, project(M, Ci, p), p) for Ci in Ck ]..., (M, λ, p) -> project(M, C, p)]

# ╔═╡ bef76406-0aae-41a8-882b-11d72fe00222
n_p = length(proxes)

# ╔═╡ 8f69a7b2-c1e9-4f4e-a0d1-4bd0d30fb01e
prox1 = (N, λ, p) -> [ proxes[i](N.manifold, λ, p[N,i]) for i=1:n_p ]

# ╔═╡ 96cef965-15fb-46db-bbb8-422e9706c0de
prox2 = (N, λ, p) -> fill(mean(N.manifold, p, GradientDescentEstimation(); stop_iter=10), length(Ck)+1)

# ╔═╡ fad104f0-31eb-46c5-acd9-be0db3a7593f
N = PowerManifold(M, NestedPowerRepresentation(), n_p)

# ╔═╡ b6491d40-a489-45ce-a05c-a16206d7b921
q0 = [4.0, 4.0]

# ╔═╡ a2d0f610-dfb9-4628-b122-c206d4488dc0
p0 = [copy(M, q0) for _ in 1:n_p]

# ╔═╡ eb5f68d7-2549-4de6-a69b-32a2459a449b
x_star = [35.0*exp(0.4*cos(5*π/4)), 35.0*exp(0.4*cos(5*π/4))]

# ╔═╡ 480ec918-c928-46d9-b207-c093e05e4f4b
λ = 0.55 # prox parameter

# ╔═╡ 528f8342-f002-4c7f-adf2-881e36657cd6
α = 0.69 # relaxation

# ╔═╡ 3fc613d0-be2c-44d7-abc5-f6910f93b008
n=2

# ╔═╡ d67632e2-6ad9-4ec8-972e-af9d095516af
θ = 0.1

# ╔═╡ 24ff4538-319d-4578-98fc-0c2fdc1571b3
sc = StopWhenChangeLess(1e-11) | StopAfterIteration(1500);

# ╔═╡ 5f7c3331-6589-41ad-a9ef-cb6643d5d736
@time s = DouglasRachford(
    N,
    cost,
    [prox1, prox2],
    p0;
    λ=i -> λ,
    α=i -> α,
	debug =  [:Iteration, " | ", :Cost, " ", :Change, "\n", 10, :Stop],
    record = [:Iteration, :Cost, :Iterate],
    stopping_criterion=sc,
    return_state=true,
)

# ╔═╡ e11d8c79-4add-4185-8791-074bb608cc02
t = @benchmark DouglasRachford(
    $N,
    $cost,
    $([prox1, prox2]),
    $p0;
    λ=$(i -> λ),
    α=$(i -> α),
    stopping_criterion=$sc,
)

# ╔═╡ 92f12dc0-45c8-4cbf-8444-0415380b7fbc
distance(M, C.center, project(M, C, get_solver_result(s)[1])) - C.radius

# ╔═╡ cf6fdac2-0fe7-4ae2-b36d-ed93944bfba5
# acceleration
@time s_acc = DouglasRachford(
    N,
    cost,
    [prox1, prox2],
    p0;
	n=n,
    λ=i -> λ,
    α=i -> α,
	debug =  [:Iteration, " | ", :Cost, " ", :Change, "\n", 10, :Stop],
    record = [:Iteration, :Cost, :Iterate],
    stopping_criterion=sc,
    return_state=true,
)

# ╔═╡ bfa1c26a-70ba-49b2-8edf-c0cf36c875ee
t_acc = @benchmark DouglasRachford(
    $N,
    $cost,
    $([prox1, prox2]),
    $p0;
	n=$n,
    λ=$(i -> λ),
    α=$(i -> α),
    stopping_criterion=$sc,
)

# ╔═╡ 6744e6fa-8002-41a4-9e37-4de5d0413693
# inertia
@time s_inertia = DouglasRachford(
    N,
    cost,
    [prox1, prox2],
    p0;
	θ=i -> θ,
    λ=i -> λ,
    α=i -> α,
	debug =  [:Iteration, " | ", :Cost, " ", :Change, "\n", 10, :Stop],
    record = [:Iteration, :Cost, :Iterate],
    stopping_criterion=sc,
    return_state=true,
)

# ╔═╡ d7091803-fdd2-4926-a0b7-c827cb8ff03e
# inertia
t_inertia = @benchmark DouglasRachford(
    $N,
    $cost,
    $([prox1, prox2]),
    $p0;
	θ=$(i -> θ),
    λ=$(i -> λ),
    α=$(i -> α),
    stopping_criterion=$sc,
)

# ╔═╡ 8cfed9db-9ad5-4cd8-a79c-3467c56d17ba
# inertia & acc
@time s_acc_inertia = DouglasRachford(
    N,
    cost,
    [prox1, prox2],
    p0;
	θ=i -> θ,
	n=n,
    λ=i -> λ,
    α=i -> α,
	debug =  [:Iteration, " | ", :Cost, " ", :Change, "\n", 10, :Stop],
    record = [:Iteration, :Cost, :Iterate],
    stopping_criterion=sc,
    return_state=true,
)

# ╔═╡ 42f8a30c-64f5-4366-8571-91d0be43bbd4
# inertia
t_acc_inertia = @benchmark DouglasRachford(
    $N,
    $cost,
    $([prox1, prox2]),
    $p0;
	θ=$(i -> θ),
	n=$n,
    λ=$(i -> λ),
    α=$(i -> α),
    stopping_criterion=$sc,
)

# ╔═╡ d37c6bf2-20a7-4c04-acb1-80f47212fc2e
iterates = [ [e[1] for e in get_record(s, :Iteration)] for s in [s, s_acc, s_inertia, s_acc_inertia]];

# ╔═╡ 1cbee04b-17c2-4774-9d7e-4da9499e338e
costs = [ [e[2] for e in get_record(s, :Iteration)] for s in [s, s_acc, s_inertia, s_acc_inertia]];

# ╔═╡ 3d5c8235-89d4-40d6-a427-941c777bfa30
errors = [ [distance(M, e[3][N,1], x_star) for e in get_record(s, :Iteration)] for s in [s, s_acc, s_inertia, s_acc_inertia]];

# ╔═╡ ca100387-20db-4fbe-8c2a-1c20bc076429
begin
    fig = plot(
	    xlabel=raw"Iterations $k$", ylabel=raw"Cost $f(x)$",
        xaxis=:log,
		#ylim = (1e-26,1e6),
		#xlim = (0,34),
    );
	plot!(fig, iterates[1], costs[1], color=indigo, label="Douglas-Rachford");
	plot!(fig, iterates[2], costs[2], color=sand, label="DR with $n-acceleration");
	plot!(fig, iterates[3], costs[3], color=teal, label=raw"DR with intertia");
	plot!(fig, iterates[4], costs[4], color=olive, label=raw"DR with both");
	fig
end

# ╔═╡ 22a0f88c-c775-4bbf-9743-48566b6e05fd
begin
    fig2 = plot(
	    xlabel=raw"Iterations $k$", ylabel=raw"Error",
        xaxis=:log,
		yaxis=:log,
		#ylim = (1e-26,1e6),
		#xlim = (0,34),
    );
	plot!(fig2, iterates[1], errors[1], color=indigo, label="Douglas-Rachford");
	plot!(fig2, iterates[2], errors[2], color=sand, label="DR with $n-acceleration");
	plot!(fig2, iterates[3], errors[3], color=teal, label=raw"DR with intertia");
	plot!(fig2, iterates[4], errors[4], color=olive, label=raw"DR with both");
	fig2
end

# ╔═╡ 487c0c98-175c-425c-816b-84f3b75f3915
# Iterations
[length(iters) for iters in iterates]

# ╔═╡ Cell order:
# ╠═0005bf5f-f7d2-4cbd-ad2f-0e3b2b958f2f
# ╠═d6ca8ff8-b46e-4dea-8d43-e8a4c7a0f677
# ╠═3653045b-2081-484b-8fc4-d7397b6fc28d
# ╠═c9d59b34-c7e1-49b0-9396-44ffbd824db2
# ╠═5e98f2a5-c369-427c-98c2-7ffcbc716622
# ╠═f42053fb-b168-4f0d-9847-9a0bff850e72
# ╠═ee853d97-49e0-48d5-8ba4-56488bf60200
# ╠═841f7d4c-35e7-41aa-9d0e-1398b8401e12
# ╠═dc73a669-f12a-4944-bbd0-b87ded61a933
# ╠═9d6359fc-2e65-4ff8-b0d0-d72641ece757
# ╠═1f323db3-7698-4c20-a66b-ad2e56ca7b0a
# ╠═21560a33-f639-493e-8ccb-393d5543cf75
# ╠═038914db-3463-43a8-abc3-4ccd7d499f18
# ╠═bef76406-0aae-41a8-882b-11d72fe00222
# ╠═8f69a7b2-c1e9-4f4e-a0d1-4bd0d30fb01e
# ╠═96cef965-15fb-46db-bbb8-422e9706c0de
# ╠═fad104f0-31eb-46c5-acd9-be0db3a7593f
# ╠═b6491d40-a489-45ce-a05c-a16206d7b921
# ╠═a2d0f610-dfb9-4628-b122-c206d4488dc0
# ╠═eb5f68d7-2549-4de6-a69b-32a2459a449b
# ╠═480ec918-c928-46d9-b207-c093e05e4f4b
# ╠═528f8342-f002-4c7f-adf2-881e36657cd6
# ╠═3fc613d0-be2c-44d7-abc5-f6910f93b008
# ╠═d67632e2-6ad9-4ec8-972e-af9d095516af
# ╠═24ff4538-319d-4578-98fc-0c2fdc1571b3
# ╠═5f7c3331-6589-41ad-a9ef-cb6643d5d736
# ╠═e11d8c79-4add-4185-8791-074bb608cc02
# ╠═92f12dc0-45c8-4cbf-8444-0415380b7fbc
# ╠═cf6fdac2-0fe7-4ae2-b36d-ed93944bfba5
# ╠═bfa1c26a-70ba-49b2-8edf-c0cf36c875ee
# ╠═6744e6fa-8002-41a4-9e37-4de5d0413693
# ╠═d7091803-fdd2-4926-a0b7-c827cb8ff03e
# ╠═8cfed9db-9ad5-4cd8-a79c-3467c56d17ba
# ╠═42f8a30c-64f5-4366-8571-91d0be43bbd4
# ╠═d37c6bf2-20a7-4c04-acb1-80f47212fc2e
# ╠═1cbee04b-17c2-4774-9d7e-4da9499e338e
# ╠═3d5c8235-89d4-40d6-a427-941c777bfa30
# ╠═ca100387-20db-4fbe-8c2a-1c20bc076429
# ╠═22a0f88c-c775-4bbf-9743-48566b6e05fd
# ╠═487c0c98-175c-425c-816b-84f3b75f3915
