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

# ╔═╡ 5582ddd4-396a-4609-892d-84060303e461


# ╔═╡ eee8994b-7167-4e06-9f61-a4bd54c399f3
s = subgradient_method(N, f, gradf, img2; stopping_criterion=StopAfterIteration(90), debug=[:Iteration, :Cost, "\n"])

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
# ╠═5582ddd4-396a-4609-892d-84060303e461
# ╠═eee8994b-7167-4e06-9f61-a4bd54c399f3
