### A Pluto.jl notebook ###
# v0.19.37

using Markdown
using InteractiveUtils

# ╔═╡ d9fee95e-bcf0-11ee-3d3f-0dbdbf97340a
using Pkg; Pkg.activate() # since we need the extend-DR branch

# ╔═╡ ec036e1e-5c25-48e3-8f03-6a093e26cf0e
using Manifolds, Manopt, ManoptExamples

# ╔═╡ 0e6cf896-d2b8-4341-b14f-2af669f6e65a
using NamedColors, Plots, BenchmarkTools

# ╔═╡ 0701d788-7b5d-488f-ae8b-49be56c73635
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

# ╔═╡ a19eb28a-dcc3-4f41-bbe2-8db669400916
md"""
#### Numerical Example for the acceleration and inertia on DR

##### The Rosenbrock problem
"""

# ╔═╡ 40fd3a50-cabb-4a16-99c8-6999905f269a
E = Euclidean(2)

# ╔═╡ fea048b5-1960-4ee8-92b5-9b757d90ab4e
M = MetricManifold(E, ManoptExamples.RosenbrockMetric())

# ╔═╡ cb6dbb63-a25f-4416-9490-42064b84d887
a = 2*10^5; b = 1; p0 = [0.1, 0.2]; p_star = [b,b^2];

# ╔═╡ 9234eb8c-aa3e-4141-9fc6-eeb567a2291b
f = ManoptExamples.RosenbrockCost(a,b)

# ╔═╡ 83998811-520a-4d88-a939-e00fe9df8839
∇f!! = ManoptExamples.RosenbrockGradient!!(a=a, b=b)

# ╔═╡ 47e5a316-4bf1-4626-a60b-b2dd85dddc09
begin
# These are described as “being defined” in the paper – I would prefer if that wold be derived.
	# f1, f2 hgere refers to the two summands of Rosenbrock, i.e. f = f1 + f2
	# (a) in-place variants
	function prox_f1!(M, q, λ, p)
		q .= [p[1], (p[2] + 2*a*λ*p[1]^2 ) / (1+2*a*λ) ]
		return q
	end
	function prox_f2!(M, q, λ, p)
		q .= [
			(p[1] + 2*λ*b) / (1+2*λ),
			p[2] - ( 4*λ*(p[1] + 2*λ*b) * (p[1] - b)  + 4 * λ^2 * (p[1]-b)^2 ) / (1+2 * λ)^2
		]
		return q
	end
	prox_f1(M, λ, p) = prox_f1!(M, copy(M, p), λ, p)
	prox_f2(M, λ, p) = prox_f2!(M, copy(M, p), λ, p)
end

# ╔═╡ 319b9c41-d06a-40b5-8a63-849d87df3cb4
sc = StopWhenChangeLess(1e-14)

# ╔═╡ a6c4c243-1662-4a89-b608-e357e5156546
md"""
### Classical Douglas Rachford
"""

# ╔═╡ 7164236b-2419-4fcb-b338-60d8df0da6d6
# A first simple run
s1 = DouglasRachford(M, f, [prox_f1!, prox_f2!], p0;
	α = i -> 0.5,
	λ = i -> 1.0,
	debug = [:Iteration, :Cost, "\n" , 25, :Stop],
	record = [ :Iteration, :Cost],
	stopping_criterion=sc,
	evaluation = InplaceEvaluation(),
	reflection_evaluation = InplaceEvaluation(),
	return_state=true
)

# ╔═╡ 7e8cb360-09e8-47ee-a371-05ca48389998
# Benchmark
@benchmark DouglasRachford($M, $f, [$prox_f1!, $prox_f2!], $p0;
	α = $(i -> 0.5),
	λ = $(i -> 1.0),
	stopping_criterion = $(sc),
	evaluation = $(InplaceEvaluation()),
	reflection_evaluation = $(InplaceEvaluation()),
)

# ╔═╡ eb7d8c91-bbdc-4147-85c2-0e2bc1b23836
q1 = get_solver_result(s1)

# ╔═╡ e301d997-1a48-4151-9ad5-aefe4302ab59
f(M, q1)

# ╔═╡ c49ab158-d5cf-4d2c-8871-1e2738f23b12
md"""
## Inertia
"""

# ╔═╡ c1f8552d-888e-4b9a-8a2f-2e6c0d13554e
s2 = DouglasRachford(M, f, [prox_f1!, prox_f2!], p0;
	α = i -> 0.5,
	λ = i -> 1.0,
	θ = i -> 0.12, # Inertia
	debug = [:Iteration, :Cost, "\n" , 10, :Stop],
	record = [ :Iteration, :Cost],
	stopping_criterion=sc,
	evaluation = InplaceEvaluation(),
	reflection_evaluation = InplaceEvaluation(),
	return_state=true
)

# ╔═╡ d05953ef-a401-43e1-9abe-f007857fad82
q2 = get_solver_result(s2)

# ╔═╡ 48dc2b53-d552-4b7b-9d9f-56d0f1526312
f(M, q2)

# ╔═╡ 3cf0a4d1-3f9f-4341-8e05-69715d61ee38
@benchmark DouglasRachford($M, $f, [$prox_f1!, $prox_f2!], $p0;
	α = $(i -> 0.5),
	λ = $(i -> 1.0),
	θ = $(i -> 0.12),
	stopping_criterion=$sc,
	evaluation = $(InplaceEvaluation()),
	reflection_evaluation = $(InplaceEvaluation()),
)

# ╔═╡ 579801a1-68a2-45c8-9a79-55819100211c
md"""
## Acceleration
"""

# ╔═╡ 8cc4e351-f48f-4817-bf14-9ef7be45bd94
s3 = DouglasRachford(M, f, [prox_f1!, prox_f2!], p0;
	α = i -> 0.5,
	λ = i -> 1.0,
	n = 3,
	debug = [:Iteration, :Cost, "\n" , 10, :Stop],
	record = [ :Iteration, :Cost],
	stopping_criterion=sc,
	evaluation = InplaceEvaluation(),
	reflection_evaluation = InplaceEvaluation(),
	return_state=true
)

# ╔═╡ a163f7ac-7b18-4d72-9f6e-23595624f4c9
q3 = get_solver_result(s3)

# ╔═╡ 972e4e37-0e51-461d-b380-ac60bb867a21
f(M, q3)

# ╔═╡ 8da066b8-5629-4c8e-bb67-fb99599b9c0d
@benchmark DouglasRachford($M, $f, [$prox_f1!, $prox_f2!], $p0;
	α = $(i -> 0.5),
	λ = $(i -> 1.0),
	n = 2,
	stopping_criterion=$sc,
	evaluation = $(InplaceEvaluation()),
	reflection_evaluation = $(InplaceEvaluation()),
)

# ╔═╡ 776f3a34-ec4f-4903-9743-8d36a9c70f8a
md"""
## Acceleration _and_ Interatia
"""

# ╔═╡ 7ba2428e-e974-4daf-8201-ef19d83445be
s4 = DouglasRachford(M, f, [prox_f1!, prox_f2!], p0;
	α = i -> 0.5,
	λ = i -> 1.0,
	θ = i -> 0.12, # Inertia
	n = 3,
	debug = [:Iteration, :Cost, "\n" , 10, :Stop],
	record = [ :Iteration, :Cost],
	stopping_criterion=sc,
	evaluation = InplaceEvaluation(),
	reflection_evaluation = InplaceEvaluation(),
	return_state=true
)

# ╔═╡ fcff553e-b71c-4f69-8743-57a02c6361ba
@benchmark DouglasRachford($M, $f, [$prox_f1!, $prox_f2!], $p0;
	α = $(i -> 0.5),
	λ = $(i -> 1.0),
	θ = i -> 0.12, # Inertia
	n = 3,
	stopping_criterion=$sc,
	evaluation = $(InplaceEvaluation()),
	reflection_evaluation = $(InplaceEvaluation()),
)

# ╔═╡ 542333c5-c804-4108-ae47-30fb592250f5
md"""
## Quasi Newton

A sanity check – quasi Newton should be able to do this reasonably quick as well.
"""

# ╔═╡ 236c5a90-7dac-4500-865e-411a38f76ba5
s5 = quasi_Newton(E, f, ∇f!!, p0;
	debug = [:Iteration, :Cost, "\n" , 100, :Stop],
	record = [ :Iteration, :Cost],
	memory_size=5,
	stopping_criterion=sc,
	evaluation = InplaceEvaluation(),
	return_state=true
)

# ╔═╡ 287d23df-ac93-4507-853d-f58d9628aa5f
q5 = get_solver_result(s5)

# ╔═╡ cae4c5b5-5249-4d2a-8e5e-b6d0a5c36408
f(E, q5)

# ╔═╡ 79ef3024-573a-44e1-93ba-511448e09b53
@benchmark s5 = quasi_Newton($E, $f, $∇f!!, $p0;
	memory_size=5,
	stopping_criterion=$sc,
	evaluation = $(InplaceEvaluation()),
)

# ╔═╡ 4219d122-d759-48df-aa03-749b07f026ff
md"""
## Summary
"""

# ╔═╡ 7c833a34-7577-43a4-9673-e63a611408cb
iterates = [ [e[1] for e in get_record(s, :Iteration)] for s in [s1,s2, s3, s4, s5]];

# ╔═╡ c84cf5c0-4d3d-458c-a355-4d3a136af852
costs = [ [e[2] for e in get_record(s, :Iteration)] for s in [s1,s2, s3, s4, s5]];

# ╔═╡ bea48c86-b38f-4cbc-99a0-2ac2c1eb1ca7
begin
    fig = plot(
	    xlabel=raw"Iterations $k$", ylabel=raw"Cost $f(x)$ (log. scale)",
        yaxis=:log,
		ylim = (1e-26,1e6),
		xlim = (0,34),
    );
	plot!(fig, iterates[1], costs[1], color=indigo, label="Douglas-Rachford");
	plot!(fig, iterates[2], costs[2], color=green, label=raw"DR with intertia of $0.12$");
	plot!(fig, iterates[3], costs[3], color=sand, label=raw"DR with 3-acceleration");
	plot!(fig, iterates[4], costs[4], color=teal, label=raw"DR both previous ones");
	fig
	plot!(fig, iterates[5], costs[5], color=wine, label=raw"Euclidean L-BFGS");
	fig
end

# ╔═╡ cd38ee25-058b-4351-9352-328e7387ce63
md"""
When we look at Quasi Newton, it reaches even numerical zero, but only relatively late:
"""

# ╔═╡ 4a6538a3-8698-4084-a54e-527b374f2ed6
begin
	fig2 = deepcopy(fig)
	plot!(fig2, xlim=(0,214))
end

# ╔═╡ Cell order:
# ╠═d9fee95e-bcf0-11ee-3d3f-0dbdbf97340a
# ╠═ec036e1e-5c25-48e3-8f03-6a093e26cf0e
# ╠═0e6cf896-d2b8-4341-b14f-2af669f6e65a
# ╠═0701d788-7b5d-488f-ae8b-49be56c73635
# ╟─a19eb28a-dcc3-4f41-bbe2-8db669400916
# ╠═40fd3a50-cabb-4a16-99c8-6999905f269a
# ╠═fea048b5-1960-4ee8-92b5-9b757d90ab4e
# ╠═cb6dbb63-a25f-4416-9490-42064b84d887
# ╠═9234eb8c-aa3e-4141-9fc6-eeb567a2291b
# ╠═83998811-520a-4d88-a939-e00fe9df8839
# ╠═47e5a316-4bf1-4626-a60b-b2dd85dddc09
# ╠═319b9c41-d06a-40b5-8a63-849d87df3cb4
# ╟─a6c4c243-1662-4a89-b608-e357e5156546
# ╠═7164236b-2419-4fcb-b338-60d8df0da6d6
# ╠═7e8cb360-09e8-47ee-a371-05ca48389998
# ╠═eb7d8c91-bbdc-4147-85c2-0e2bc1b23836
# ╠═e301d997-1a48-4151-9ad5-aefe4302ab59
# ╟─c49ab158-d5cf-4d2c-8871-1e2738f23b12
# ╠═c1f8552d-888e-4b9a-8a2f-2e6c0d13554e
# ╠═d05953ef-a401-43e1-9abe-f007857fad82
# ╠═48dc2b53-d552-4b7b-9d9f-56d0f1526312
# ╠═3cf0a4d1-3f9f-4341-8e05-69715d61ee38
# ╟─579801a1-68a2-45c8-9a79-55819100211c
# ╠═8cc4e351-f48f-4817-bf14-9ef7be45bd94
# ╠═a163f7ac-7b18-4d72-9f6e-23595624f4c9
# ╠═972e4e37-0e51-461d-b380-ac60bb867a21
# ╠═8da066b8-5629-4c8e-bb67-fb99599b9c0d
# ╟─776f3a34-ec4f-4903-9743-8d36a9c70f8a
# ╠═7ba2428e-e974-4daf-8201-ef19d83445be
# ╠═fcff553e-b71c-4f69-8743-57a02c6361ba
# ╟─542333c5-c804-4108-ae47-30fb592250f5
# ╠═236c5a90-7dac-4500-865e-411a38f76ba5
# ╠═287d23df-ac93-4507-853d-f58d9628aa5f
# ╠═cae4c5b5-5249-4d2a-8e5e-b6d0a5c36408
# ╠═79ef3024-573a-44e1-93ba-511448e09b53
# ╟─4219d122-d759-48df-aa03-749b07f026ff
# ╠═7c833a34-7577-43a4-9673-e63a611408cb
# ╠═c84cf5c0-4d3d-458c-a355-4d3a136af852
# ╠═bea48c86-b38f-4cbc-99a0-2ac2c1eb1ca7
# ╟─cd38ee25-058b-4351-9352-328e7387ce63
# ╠═4a6538a3-8698-4084-a54e-527b374f2ed6
