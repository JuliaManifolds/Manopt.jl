### A Pluto.jl notebook ###
# v0.19.0

using Markdown
using InteractiveUtils

# ╔═╡ d18893e3-89ec-46fc-9265-0daac744caea
using Manopt, Manifolds, Colors, PlutoUI

# ╔═╡ 3b55c91e-bda9-11ec-0399-b77442d8ff45
md"""
# Illustration of the Gradient of a Second Order Difference

This example explains how to compute the gradient of the second order
difference mid point model using `adjoint_Jacobi_field`

This example also illustrates the `PowerManifold` manifold as well as `ArmijoLinesearch`.
"""

# ╔═╡ b2a115cc-5d4e-4052-afc1-2972ed493958
md"We define some colors from [Paul Tol](https://personal.sron.nl/~pault/)"

# ╔═╡ c19c8ede-ea81-4b10-85c2-f73051d8090a
begin
    black = RGBA{Float64}(colorant"#000000")
    TolVibrantBlue = RGBA{Float64}(colorant"#0077BB")
    TolVibrantOrange = RGBA{Float64}(colorant"#EE7733")
    TolVibrantMagenta = RGBA{Float64}(colorant"#EE3377")
    TolVibrantCyan = RGBA{Float64}(colorant"#33BBEE")
    TolVibrantTeal = RGBA{Float64}(colorant"#009988")
end;

# ╔═╡ b5ba2630-e247-46b9-b306-faf8a531f22a
begin
    T = [0:0.1:1.0...]
    #render asy yes/no. If not, images included w/ markdown are assumed to be prerendered
    render_asy = false
    localpath = join(splitpath(@__FILE__)[1:(end - 1)], "/") # remove file to get files folder
    image_prefix = localpath * "/second_order_difference"
    @info image_prefix
end;

# ╔═╡ 2a51bd75-c47e-452c-bfea-f6d7cfcd3e45
md"""
Assume we have two points ``p,q`` on the equator of the
[Sphere](https://juliamanifolds.github.io/Manifolds.jl/stable/manifolds/sphere.html) ``\mathcal M = \mathbb S^2``
and a point ``r`` near the north pole
"""

# ╔═╡ 3c8a6104-d380-4a32-907c-aaf205ea0bdd
begin
    M = Sphere(2)
    p = [1.0, 0.0, 0.0]
    q = [0.0, 1.0, 0.0]
    c = mid_point(M, p, q)
    r = shortest_geodesic(M, [0.0, 0.0, 1.0], c, 0.1)
end;

# ╔═╡ 9cfecd72-0c1a-4432-9628-54b71db9c22d
md"""
Now the second order absolute difference can be stated as [^BacakBergmannSteidlWeinmann2016]

```math
d_2(p_1,p:2,p_3) := \min_{c ∈ \mathcal C_{p_1,p_3}} d_{\mathcal M}(c,p_2),\qquad p_1,p_2,p_3∈\mathcal M,
```

where ``\mathcal C_{p,q}`` is the set of all mid points ``g(\frac{1}{2};p,q)``, between `p`and `q`, i.e. where ``g``
is a (not necessarily minimizing) geodesic connecting both.

For illustration we further define the point opposite of the mid point `c` defined above
"""

# ╔═╡ a9596c67-1cfb-4d94-a7c8-71bf36fe06bb
c2 = -c;

# ╔═╡ 2dfaebe0-e706-4eb6-b661-aed9d8ae19c1
md"To illustrate the second order difference let‘s look at the geodesic connecting ``r`` and the mid point ``c``"

# ╔═╡ 1c484736-bb94-4f64-bcfd-ad7ed52ade39
geoPts_rc = shortest_geodesic(M, r, c, T);

# ╔═╡ 80aa3913-7639-4752-a762-58c17a7e1c47
render_asy && begin
    asymptote_export_S2_signals(
        image_prefix * "/SecondOrderData.asy";
        curves=[geoPts_rc],
        points=[[p, r, q], [c, c2]],
        colors=Dict(:curves => [TolVibrantTeal], :points => [black, TolVibrantBlue]),
        dot_size=3.5,
        line_width=0.75,
        camera_position=(1.2, 1.0, 0.5), #src
    )
    render_asymptote(image_prefix * "/SecondOrderData.asy"; render=2)
end;

# ╔═╡ 717ba868-7da5-4ce1-9163-2b20f8cdc253
PlutoUI.LocalResource(image_prefix * "/SecondOrderData.png")

# ╔═╡ dcf6bbeb-5b09-4689-8bb9-e35f089cff45
md" **Figure.** _The three poins ``p, q``, and ``r`` (neart north pole, all black), which is connected by a geodesic (teal) to the mid point of the first two, ``c``, (blue). The length of this geodesic is the cost of the second order total variation._"

# ╔═╡ 07a44b4c-876c-4fff-ad72-2baedc06d090
md"""
Since we moved ``r`` 10% along the geodesic from the north pole to ``c``, the distance
to ``c`` is ``\frac{9\pi}{20}\approx 1.4137``, and this is also what the second order total variation cost, see [`costTV2`](https://manoptjl.org/stable/functions/costs.html#Manopt.costTV2), yields.
"""

# ╔═╡ ebab946c-ba27-422d-a191-56294e57177b
costTV2(M, (p, r, q))

# ╔═╡ db26c554-31ac-4d62-93a1-9440361fa821
md"""
But also its gradient can be
easily computed since it is just a distance with respect to ``r`` and a
concatenation of a geodesic, where the start or end point is the argument,
respectively, with a distance.
Hence the adjoint differentials
[`adjoint_differential_geodesic_startpoint`](https://manoptjl.org/stable/functions/adjointdifferentials.html#Manopt.adjoint_differential_geodesic_startpoint-Tuple{AbstractManifold,%20Any,%20Any,%20Any,%20Any}) and [`adjoint_differential_geodesic_endpoint`](https://manoptjl.org/stable/functions/adjointdifferentials.html#Manopt.adjoint_differential_geodesic_endpoint-Tuple{AbstractManifold,%20Any,%20Any,%20Any,%20Any}) can be employed.
The gradient is also directly implemented, see [`grad_TV2`](https://manoptjl.org/stable/functions/gradients.html#Manopt.grad_TV2).
we obtain
"""

# ╔═╡ b685cd13-8b06-4d6d-a001-576f68951bd9
(Xp, Xr, Xq) = grad_TV2(M, (p, r, q))

# ╔═╡ dddd7edf-47b7-4b6d-8b20-e49b4db211a9
render_asy && begin
    asymptote_export_S2_signals(
        image_prefix * "/SecondOrderGradient.asy";
        points=[[p, r, q], [c, c2]],
        tangent_vectors=[Tuple.([[p, -Xp], [r, -Xr], [q, -Xq]])],
        colors=Dict(:tvectors => [TolVibrantCyan], :points => [black, TolVibrantBlue]),
        dot_size=3.5,
        line_width=0.75,
        camera_position=(1.2, 1.0, 0.5),
    )
    render_asymptote(image_prefix * "/SecondOrderGradient.asy"; render=2)
end;

# ╔═╡ 18bc72b8-fef9-4890-bdb8-dc477e0109db
PlutoUI.LocalResource(image_prefix * "/SecondOrderGradient.png")

# ╔═╡ cdf26e32-2383-4ac3-b814-ef7c98860b55
md"""
**Figure.** _The negative gradient of the second order difference cost indicates the movement of the three points in order to reduce their cost._
"""

# ╔═╡ 77149fb5-f3cb-438f-843c-9221ae5c08a9
md"If we now perform a gradient step with constant step size 1, we obtain the three points"

# ╔═╡ e53622ca-3e08-4e16-a1a3-6536a46c8a9f
pn, rn, qn = exp.(Ref(M), [p, r, q], [-Xp, -Xr, -Xq])

# ╔═╡ 6b50fb13-45db-4cbb-a1e0-d62187c0b2d3
md"as well we the new mid point"

# ╔═╡ 0bbd56ce-9041-4974-8525-a0b5bd03e9fe
cn = mid_point(M, pn, qn)

# ╔═╡ 2eee7e24-20aa-4ae4-a336-b2eec90ce24c
md"Let‘s also again consider the geodesic connecting the new point ``r_n`` and the new mid point ``c_n`` as well as the gradient"

# ╔═╡ ce758b03-0177-45c2-b895-e64d133ca4af
begin
    geoPts_rncn = shortest_geodesic(M, rn, cn, T)
    (Xpn, Xrn, Xqn) = grad_TV2(M, (pn, rn, qn))
end;

# ╔═╡ 77e2409a-0168-4715-8770-23272e469f87
md"The new configuration of the three points looks as follows"

# ╔═╡ d90730e1-7111-42d9-8688-b4ff9cca9c6e
render_asy && begin
    asymptote_export_S2_signals(
        image_prefix * "/SecondOrderMin1.asy";
        points=[[p, r, q], [c, c2, cn], [pn, rn, qn]],
        curves=[geoPts_rncn],
        tangent_vectors=[
            Tuple.([[p, -Xp], [r, -Xr], [q, -Xq]]),
            Tuple.([[pn, -Xpn], [rn, -Xrn], [qn, -Xqn]]),
        ],
        colors=Dict(
            :tvectors => [TolVibrantCyan, TolVibrantMagenta],
            :points => [black, TolVibrantBlue, TolVibrantOrange],
            :curves => [TolVibrantTeal],
        ),
        dot_size=3.5,
        line_width=0.75,
        camera_position=(1.2, 1.0, 0.5),
    )
    render_asymptote(image_prefix * "/SecondOrderMin1.asy"; render=2)
end;

# ╔═╡ 533f324b-40ca-4b74-968e-e9f1ce4262eb
PlutoUI.LocalResource(image_prefix * "/SecondOrderMin1.png")

# ╔═╡ a959be34-c0fb-4dc3-b46d-1702f616a8bb
md"""
**Figure.** *The new situation of ``p_n, q_n``, and ``r_n`` (orange) and the miod point of the first two, ``c_n`` (blue), which is again connected by a geodesic to ``r_n``. Note that this geodesic is shorter, but also that ``c`` and ``r`` switched places*. The new gradient (magenta) is only slightly reduced in magnitude.* 
"""

# ╔═╡ 5b4aa1ae-7ab1-4efc-a796-1fa4e771e2b2
md"""
One can see, that this step slightly “overshoots”, i.e. ``r`` is now even below ``c``.
and the cost function is still at
"""

# ╔═╡ ea3ceaad-d032-42fd-84ca-a08b2fdd6db0
costTV2(M, (pn, rn, qn))

# ╔═╡ 96600890-af72-4842-973e-438cb930f57b
md"""
But we can also search for the best step size using [`linesearch_backtrack`](https://manoptjl.org/stable/plans/index.html#Manopt.linesearch_backtrack-Union{Tuple{T},%20Tuple{TF},%20Tuple{AbstractManifold,%20TF,%20Any,%20T,%20Any,%20Any,%20Any},%20Tuple{AbstractManifold,%20TF,%20Any,%20T,%20Any,%20Any,%20Any,%20AbstractRetractionMethod},%20Tuple{AbstractManifold,%20TF,%20Any,%20T,%20Any,%20Any,%20Any,%20AbstractRetractionMethod,%20T},%20Tuple{AbstractManifold,%20TF,%20Any,%20T,%20Any,%20Any,%20Any,%20AbstractRetractionMethod,%20T,%20Any}}%20where%20{TF,%20T})
on the `PowerManifold` manifold ``\mathcal N = \mathcal M^3 = (\mathbb S^2)^3``
"""

# ╔═╡ 4b662610-0812-4ae2-809c-52f7a8979398
begin
    x = [p, r, q]
    N = PowerManifold(M, NestedPowerRepresentation(), 3)
    s = linesearch_backtrack(N, x -> costTV2(N, x), x, grad_TV2(N, x), 1.0, 0.957, 0.999)
end

# ╔═╡ 17ccdc00-35ad-4ec5-a480-6f6527e38834
md"and we obtain the new points"

# ╔═╡ db8916e6-0ee7-4e37-a879-5f35aa11d38d
begin
    pm, rm, qm = exp.(Ref(M), [p, r, q], s * [-Xp, -Xr, -Xq])
    cm = mid_point(M, pm, qm)
    geoPts_pmqm = shortest_geodesic(M, pm, qm, T)
end;

# ╔═╡ 78d44ce1-9d7b-4cb1-b9cb-2d402b1dfabb
md"we obtain"

# ╔═╡ fdf9aa5e-a25a-42f3-a74c-cef5877f2a1d
render_asy && begin
    asymptote_export_S2_signals(
        image_prefix * "/SecondOrderMin2.asy";
        points=[[p, r, q], [c, c2, cm], [pm, rm, qm]],
        curves=[geoPts_pmqm],
        tangent_vectors=[Tuple.([[p, -Xp], [r, -Xr], [q, -Xq]])],
        colors=Dict(
            :tvectors => [TolVibrantCyan],
            :points => [black, TolVibrantBlue, TolVibrantOrange],
            :curves => [TolVibrantTeal],
        ),
        dot_size=3.5,
        line_width=0.75,
        camera_position=(1.2, 1.0, 0.5),
    )
    render_asymptote(image_prefix * "/SecondOrderMin2.asy"; render=2)
end;

# ╔═╡ b0a75ff9-9ff7-4db6-ad82-76f183663dc0
PlutoUI.LocalResource(image_prefix * "/SecondOrderMin2.png")

# ╔═╡ 9c37f43b-dc3e-45fb-a99f-756c83f2f658
md"""
**Figure.** *For the best step size found by line search, ``r_m`` and ``c_m`` nearly agree, i.e. ``r_m`` lies on the geodesic between ``p_m`` and ``q_m`` as the geodesic drawn here indicates.*
"""

# ╔═╡ 428d1422-7521-42fd-993d-e617115edf14
md"Here, the cost function yields"

# ╔═╡ 25804825-c9f6-4cfa-b209-918ad19d2ae1
costTV2(M, (pm, rm, qm))

# ╔═╡ 718f744e-e6af-4158-9418-7d2f4b757903
md"which is much closer to zero, as one can also see, since the new center ``c_m`` and ``r_m`` are quite close."

# ╔═╡ 962936da-c154-444f-8d48-ca1176dee1a1
md"""
## Literature

[^BacakBergmannSteidlWeinmann2016]:
    > Bačák, M., Bergmann, R., Steidl, G. and Weinmann, A.: _A second order nonsmooth
    > variational model for restoring manifold-valued images_,
    > SIAM Journal on Scientific Computations, Volume 38, Number 1, pp. A567–597,
    > doi: [10.1137/15M101988X](https://doi.org/10.1137/15M101988X),
    > arXiv: [1506.02409](https://arxiv.org/abs/1506.02409)
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
Manifolds = "1cead3c2-87b3-11e9-0ccd-23c62b72b94e"
Manopt = "0fc0a36d-df90-57f3-8f93-d78a9fc72bb5"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
Colors = "~0.12.8"
Manifolds = "~0.7.8"
Manopt = "~0.3.23"
PlutoUI = "~0.7.38"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.2"
manifest_format = "2.0"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "6f1d9bc1c08f9f4a8fa92e3ea3cb50153a1b40d4"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.1.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "62e51b39331de8911e4a7ff6f5aaf38a5f4cc0ae"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.2.0"

[[deps.ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "c933ce606f6535a7c7b98e1d86d5d1014f730596"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "5.0.7"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9950387274246d08af38f6eef8cb5480862a435f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.14.0"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "12fc73e5e0af68ad3137b886e3f7c1eacfca2640"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.17.1"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "b153278a25dd42c65abbf4e62344f9d22e59191b"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.43.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.CovarianceEstimation]]
deps = ["LinearAlgebra", "Statistics", "StatsBase"]
git-tree-sha1 = "a3e070133acab996660d31dcf479ea42849e368f"
uuid = "587fd27a-f159-11e8-2dae-1979310e6154"
version = "0.2.7"

[[deps.DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "5a4168170ede913a2cd679e53c2123cb4b889795"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.53"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.Einsum]]
deps = ["Compat"]
git-tree-sha1 = "4a6b3eee0161c89700b6c1949feae8b851da5494"
uuid = "b7d42ee7-0b51-5a75-98ca-779d3107e4c0"
version = "0.4.1"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "246621d23d1f43e3b9c368bf3b72b2331a27c286"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.2"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "57c021de207e234108a6f1454003120a1bf350c4"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.6.0"

[[deps.HybridArrays]]
deps = ["LinearAlgebra", "Requires", "StaticArrays"]
git-tree-sha1 = "eb6b23460f5544c5d09efae0818b86736cefcd3d"
uuid = "1baab800-613f-4b0a-84e4-9cd3431bfbb9"
version = "0.4.10"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "91b5dcf362c5add98049e6c29ee756910b03051d"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.3"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.Kronecker]]
deps = ["LinearAlgebra", "NamedDims", "SparseArrays", "StatsBase"]
git-tree-sha1 = "a51f46415c844dee694cb8b20a3fcbe6dba342c2"
uuid = "2c470bb0-bcc8-11e8-3dad-c9649493f05e"
version = "0.5.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "a970d55c2ad8084ca317a4658ba6ce99b7523571"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.12"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Manifolds]]
deps = ["Colors", "Distributions", "Einsum", "Graphs", "HybridArrays", "Kronecker", "LinearAlgebra", "ManifoldsBase", "Markdown", "Random", "RecipesBase", "RecursiveArrayTools", "Requires", "SimpleWeightedGraphs", "SpecialFunctions", "StaticArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "cc5e34ecd061bf9bedd62a8b8a28abe96c3b7c6e"
uuid = "1cead3c2-87b3-11e9-0ccd-23c62b72b94e"
version = "0.7.8"

[[deps.ManifoldsBase]]
deps = ["LinearAlgebra", "Markdown"]
git-tree-sha1 = "b8518cbf0f06f1375c95479759b266bd75d89ad9"
uuid = "3362f125-f0bb-47a3-aa74-596ffd7ef2fb"
version = "0.12.13"

[[deps.Manopt]]
deps = ["ColorSchemes", "ColorTypes", "Colors", "DataStructures", "Dates", "LinearAlgebra", "ManifoldsBase", "Markdown", "Printf", "Random", "Requires", "SparseArrays", "StaticArrays", "Statistics", "Test"]
git-tree-sha1 = "a8a292a422b0a6924f46d4e022fc8263721120c5"
uuid = "0fc0a36d-df90-57f3-8f93-d78a9fc72bb5"
version = "0.3.23"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.NamedDims]]
deps = ["AbstractFFTs", "ChainRulesCore", "CovarianceEstimation", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "0856b62716585eb90cc1dada226ac9eab5f69aa5"
uuid = "356022a1-0364-5f58-8944-0da4b18d706f"
version = "0.2.47"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "e8185b83b9fc56eb6456200e873ce598ebc7f262"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.7"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "621f4f3b4977325b9128d5fae7a8b4829a0c2222"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.4"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "670e559e5c8e191ded66fa9ea89c97f10376bb4c"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.38"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "d3538e7f8a790dc8903519090857ef8e1283eecd"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.5"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "ChainRulesCore", "DocStringExtensions", "FillArrays", "LinearAlgebra", "RecipesBase", "Requires", "StaticArrays", "Statistics", "ZygoteRules"]
git-tree-sha1 = "bfe14f127f3e7def02a6c2b1940b39d0dabaa3ef"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "2.26.3"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.SimpleWeightedGraphs]]
deps = ["Graphs", "LinearAlgebra", "Markdown", "SparseArrays", "Test"]
git-tree-sha1 = "a6f404cc44d3d3b28c793ec0eb59af709d827e4e"
uuid = "47aef6b3-ad0c-573a-a1e2-d07658019622"
version = "1.2.1"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "5ba658aeecaaf96923dce0da9e703bd1fe7666f9"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.4"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "87e9954dfa33fd145694e42337bdd3d5b07021a6"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.6.0"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "4f6ec5d99a28e1a749559ef7dd518663c5eca3d5"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "8d7530a38dbd2c397be7ddd01a424e4f411dcc41"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.2"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8977b17906b0a1cc74ab2e3a05faa16cf08a8291"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.16"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5950925ff997ed6fb3e985dcce8eb1ba42a0bbe7"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.18"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─3b55c91e-bda9-11ec-0399-b77442d8ff45
# ╠═d18893e3-89ec-46fc-9265-0daac744caea
# ╟─b2a115cc-5d4e-4052-afc1-2972ed493958
# ╠═c19c8ede-ea81-4b10-85c2-f73051d8090a
# ╠═b5ba2630-e247-46b9-b306-faf8a531f22a
# ╟─2a51bd75-c47e-452c-bfea-f6d7cfcd3e45
# ╠═3c8a6104-d380-4a32-907c-aaf205ea0bdd
# ╟─9cfecd72-0c1a-4432-9628-54b71db9c22d
# ╠═a9596c67-1cfb-4d94-a7c8-71bf36fe06bb
# ╟─2dfaebe0-e706-4eb6-b661-aed9d8ae19c1
# ╠═1c484736-bb94-4f64-bcfd-ad7ed52ade39
# ╠═80aa3913-7639-4752-a762-58c17a7e1c47
# ╟─717ba868-7da5-4ce1-9163-2b20f8cdc253
# ╟─dcf6bbeb-5b09-4689-8bb9-e35f089cff45
# ╟─07a44b4c-876c-4fff-ad72-2baedc06d090
# ╠═ebab946c-ba27-422d-a191-56294e57177b
# ╟─db26c554-31ac-4d62-93a1-9440361fa821
# ╠═b685cd13-8b06-4d6d-a001-576f68951bd9
# ╠═dddd7edf-47b7-4b6d-8b20-e49b4db211a9
# ╟─18bc72b8-fef9-4890-bdb8-dc477e0109db
# ╟─cdf26e32-2383-4ac3-b814-ef7c98860b55
# ╟─77149fb5-f3cb-438f-843c-9221ae5c08a9
# ╠═e53622ca-3e08-4e16-a1a3-6536a46c8a9f
# ╟─6b50fb13-45db-4cbb-a1e0-d62187c0b2d3
# ╟─0bbd56ce-9041-4974-8525-a0b5bd03e9fe
# ╠═2eee7e24-20aa-4ae4-a336-b2eec90ce24c
# ╠═ce758b03-0177-45c2-b895-e64d133ca4af
# ╟─77e2409a-0168-4715-8770-23272e469f87
# ╠═d90730e1-7111-42d9-8688-b4ff9cca9c6e
# ╠═533f324b-40ca-4b74-968e-e9f1ce4262eb
# ╟─a959be34-c0fb-4dc3-b46d-1702f616a8bb
# ╟─5b4aa1ae-7ab1-4efc-a796-1fa4e771e2b2
# ╠═ea3ceaad-d032-42fd-84ca-a08b2fdd6db0
# ╟─96600890-af72-4842-973e-438cb930f57b
# ╠═4b662610-0812-4ae2-809c-52f7a8979398
# ╟─17ccdc00-35ad-4ec5-a480-6f6527e38834
# ╠═db8916e6-0ee7-4e37-a879-5f35aa11d38d
# ╟─78d44ce1-9d7b-4cb1-b9cb-2d402b1dfabb
# ╠═fdf9aa5e-a25a-42f3-a74c-cef5877f2a1d
# ╠═b0a75ff9-9ff7-4db6-ad82-76f183663dc0
# ╟─9c37f43b-dc3e-45fb-a99f-756c83f2f658
# ╟─428d1422-7521-42fd-993d-e617115edf14
# ╠═25804825-c9f6-4cfa-b209-918ad19d2ae1
# ╟─718f744e-e6af-4158-9418-7d2f4b757903
# ╟─962936da-c154-444f-8d48-ca1176dee1a1
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
