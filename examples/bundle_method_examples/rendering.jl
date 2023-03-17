### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ 1c1f69c8-c330-11ed-2117-2ba0918ec216
using Pkg; Pkg.activate()

# ╔═╡ e41fee0e-ad95-44c0-96ee-9457453f534e
using Random, LinearAlgebra, QuadraticModels, RipQP, Manifolds, Manopt, ColorSchemes

# ╔═╡ f8a8374a-b088-4739-babc-43f1867eb85f
begin
	Random.seed!(42)
	img = artificial_SPD_image(25, 1.5)
	M = SymmetricPositiveDefinite(3)
	N = PowerManifold(M, NestedPowerRepresentation(), size(img)[1], size(img)[2])
	f(N, q) = costTV2(N, q, 2)
	gradf(N, q) = grad_TV2(N, q)
end

# ╔═╡ bede3a2d-67c5-4925-81d3-ff306c9b2095
#s = subgradient_method(N, f, gradf, rand(N); stopping_criterion=StopAfterIteration(90), debug=[:Iteration, :Cost, "\n"])

# ╔═╡ 23668e76-8b34-43f6-af65-64eaaeba7dd6
#b = bundle_method(N, f, gradf, rand(N); stopping_criterion=StopAfterIteration(90), debug=[:Iteration, :Cost, "\n"])

# ╔═╡ fc28e3af-33ca-4f1b-9dbd-a90a0717348a
asymptote_export_SPD("export1.fig"; data=img, scale_axes=(4.,4.,4.), color_scheme=ColorSchemes.hsv)

# ╔═╡ d5218834-ab32-46b4-bcd8-9789df00a787
render_asymptote("export1.fig")

# ╔═╡ b8311ec8-342c-42eb-b7b3-d8716a1af3af
#asymptote_export_SPD("export2.fig"; data=b, scale_axes=(6.,6.,6.))

# ╔═╡ 6dc21c0d-e2c1-41e9-8358-cf9b41b19102
#render_asymptote("export2.fig")

# ╔═╡ 2aa71666-c59f-416b-b11a-01931a08f695
img2 = map(p -> exp(M, p, rand(M; vector_at=p, tangent_distr=:Rician, σ=0.03)), img) #add (exp) noise to image

# ╔═╡ 997a9286-1788-46e6-8a5f-ae8cee7d43a2
asymptote_export_SPD("export2.fig"; data=img2, scale_axes=(12.,12.,12.), color_scheme=ColorSchemes.hsv)

# ╔═╡ 7c03e397-1dd8-4f58-9bb9-c412284b8690
render_asymptote("export2.fig")

# ╔═╡ eee8994b-7167-4e06-9f61-a4bd54c399f3
#s = subgradient_method(N, f, gradf, img2; stopping_criterion=StopAfterIteration(90), debug=[:Iteration, :Cost, "\n"])

# ╔═╡ d1cd19f4-1412-4831-bed3-8b1abf25bfa0
begin
	img3 = artificial_SPD_image2(16)
	asymptote_export_SPD("export3.fig"; data=img3, scale_axes=(8.,8.,8.), color_scheme=ColorSchemes.hsv)
	render_asymptote("export3.fig")
end

# ╔═╡ ec0de758-b00b-4bdd-a0d3-b2361b03c74a
begin
	data=img3#map(p -> exp(M, p, rand(M; vector_at=p, tangent_distr=:Rician, σ=0.03)), img3) 
	α = 6.0
	L = PowerManifold(M, NestedPowerRepresentation(), size(img3)[1], size(img3)[2])
	g(L, q) = 1 / α * costL2TV(L, data, α, q)
	gradg(L, q) = 1 / α * grad_distance(L, data, q) + grad_TV(L, q)
end

# ╔═╡ 6749f982-99cb-4dd2-ad56-9aec03a3663d
s=subgradient_method(L, g, gradg, data; 
stepsize = ConstantStepsize(1e-2),
debug=[:Iteration, (:Cost,"F(p): %1.15e"), "\n"],
stopping_criterion=StopAfterIteration(5000))

# ╔═╡ d9416139-6170-421a-b181-641d50c98695
b=bundle_method(L, g, gradg, data; 
m=0.9, diam=100., debug=[:Iteration, (:Cost,"F(p): %1.16e"), "\n"], stopping_criterion=StopAfterIteration(20))

# ╔═╡ ed110367-2d6c-4eba-adb1-14ae9a4b1175
#g(L, s) ≈ g(L, b)

# ╔═╡ a9d0b9a7-1ed3-4f1e-aefe-8e518f1899bf
begin
	asymptote_export_SPD("b.fig"; data=s, scale_axes=(8.,8.,8.), color_scheme=ColorSchemes.hsv)
	render_asymptote("b.fig")
end

# ╔═╡ Cell order:
# ╠═1c1f69c8-c330-11ed-2117-2ba0918ec216
# ╠═e41fee0e-ad95-44c0-96ee-9457453f534e
# ╠═f8a8374a-b088-4739-babc-43f1867eb85f
# ╠═bede3a2d-67c5-4925-81d3-ff306c9b2095
# ╠═23668e76-8b34-43f6-af65-64eaaeba7dd6
# ╠═fc28e3af-33ca-4f1b-9dbd-a90a0717348a
# ╠═d5218834-ab32-46b4-bcd8-9789df00a787
# ╠═b8311ec8-342c-42eb-b7b3-d8716a1af3af
# ╠═6dc21c0d-e2c1-41e9-8358-cf9b41b19102
# ╠═2aa71666-c59f-416b-b11a-01931a08f695
# ╠═997a9286-1788-46e6-8a5f-ae8cee7d43a2
# ╠═7c03e397-1dd8-4f58-9bb9-c412284b8690
# ╠═eee8994b-7167-4e06-9f61-a4bd54c399f3
# ╠═d1cd19f4-1412-4831-bed3-8b1abf25bfa0
# ╠═ec0de758-b00b-4bdd-a0d3-b2361b03c74a
# ╠═6749f982-99cb-4dd2-ad56-9aec03a3663d
# ╠═d9416139-6170-421a-b181-641d50c98695
# ╠═ed110367-2d6c-4eba-adb1-14ae9a4b1175
# ╠═a9d0b9a7-1ed3-4f1e-aefe-8e518f1899bf
