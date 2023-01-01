### A Pluto.jl notebook ###
# v0.19.14

using Markdown
using InteractiveUtils

# ╔═╡ 8a8be09a-49e9-4563-9afc-ae72bd05d26d
using BenchmarkTools, Colors, Manopt, Manifolds, PlutoUI, Random

# ╔═╡ f58f28fe-c3d8-11ec-0acb-e948605dd63f
md"""
# Stochastic Gradient Descent

This tutorial illustrates how to use the [`stochastic_gradient_descent`](https://manoptjl.org/stable/solvers/stochastic_gradient_descent.html)
solver and different [`DirectionUpdateRule`](https://manoptjl.org/stable/solvers/gradient_descent.html#Direction-Update-Rules-1)s in order to introduce
the average or momentum variant, see [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent).

Computationally, we look at a very simple but large scale problem,
the Riemannian Center of Mass or [Fréchet mean](https://en.wikipedia.org/wiki/Fréchet_mean):
for given points ``p_i ∈\mathcal M``, ``i=1,…,N`` this optimization problem reads

```math
\operatorname*{arg\,min}_{x∈\mathcal M} \frac{1}{2}\sum_{i=1}^{N}
  \operatorname{d}^2_{\mathcal M}(x,p_i),
```

which of course can be (and is) solved by a gradient descent, see the introductionary tutorial or [Statistics in Manifolds.jl](https://juliamanifolds.github.io/Manifolds.jl/stable/features/statistics.html).
If ``N`` is very large, evaluating the complete gradient might be quite expensive.
A remedy is to evaluate only one of the terms at a time and choose a random order for these.

We first initialize the packages
"""

# ╔═╡ 84528b83-f11d-42de-9e86-fc9f3d451500
md"and we define some colors from [Paul Tol](https://personal.sron.nl/~pault/)"

# ╔═╡ c104cad0-c489-42f8-b896-986811eebfef
begin
    black = RGBA{Float64}(colorant"#000000")
    TolVibrantOrange = RGBA{Float64}(colorant"#EE7733") # Start
    TolVibrantBlue = RGBA{Float64}(colorant"#0077BB") # a path
    TolVibrantTeal = RGBA{Float64}(colorant"#009988") # points
end;

# ╔═╡ 11ffe7e1-22f5-4098-9830-d56ff59745ee
md"We next generate a (little) large(r) data set"

# ╔═╡ f3a29d0e-91a5-437f-8681-5b72072b9cdf
begin
    n = 5000
    σ = π / 12
    M = Sphere(2)
    x = 1 / sqrt(2) * [1.0, 0.0, 1.0]
    Random.seed!(42)
    data = [exp(M, x,  σ * rand(M; vector_at=x)) for i in 1:n]
    localpath = join(splitpath(@__FILE__)[1:(end - 1)], "/") # files folder
    image_prefix = localpath * "/stochastic_gradient_descent"
    @info image_prefix
    render_asy = false # on CI or when you do not have asymptote, this should be false
end

# ╔═╡ 4f527a26-dc55-4eb9-ab1b-62a4712182d5
render_asy && asymptote_export_S2_signals(
    image_prefix * "/center_and_large_data.asy";
    points=[[x], data],
    colors=Dict(:points => [TolVibrantBlue, TolVibrantTeal]),
    dot_sizes=[2.5, 1.0],
    camera_position=(1.0, 0.5, 0.5),
)

# ╔═╡ 8baf553d-8d22-4f13-9741-43531b080998
render_asy && render_asymptote(image_prefix * "/center_and_large_data.asy"; render=2);

# ╔═╡ daf0dc47-cba2-4264-84d9-47137924addd
PlutoUI.LocalResource(image_prefix * "/center_and_large_data.png")

# ╔═╡ 30f0a209-afde-40c4-a8ab-3f40fd374d3c
md"""
Note that due to the construction of the points as zero mean tangent vectors, the mean should
be very close to our initial point `x`.

In order to use the stochastic gradient, we now need a function that returns the vector of gradients.
There are two ways to define it in `Manopt.jl`: either as a single function that returns a vector, or as a vector of functions.

The first variant is of course easier to define, but the second is more efficient when only evaluating one of the gradients.

For the mean, the gradient is

```math
 gradF(x) = \sum_{i=1}^N \operatorname{grad}f_i(x) \quad \text{where} \operatorname{grad}f_i(x) = -\log_x p_i
```

which we define in `Manopt.jl` in two different ways:
either as one function returning all gradients as a vector (see `gradF`), or – maybe more fitting for a large scale problem, as a vector of small gradient functions (see `gradf`)
"""

# ╔═╡ 12a60ec6-cd18-4eea-a0af-0cabd43686b8
F(M, x) = 1 / (2 * n) * sum(map(p -> distance(M, x, p)^2, data))

# ╔═╡ d76aa8ef-b234-40d6-940b-cfc5db38cee8
gradF(M, x) = [grad_distance(M, p, x) for p in data]

# ╔═╡ d41271a4-cbbe-46a3-b350-fce79d860671
gradf = [(M, x) -> grad_distance(M, p, x) for p in data];

# ╔═╡ 2d977c61-c556-4d58-8ca6-de0f7977cd30
md"""
The calls are only slightly different, but notice that accessing the 2nd gradient element
requires evaluating all logs in the first function.
So while you can use both `gradF` and `gradf` in the following call, the second one is (much) faster:
"""

# ╔═╡ 2b1ed0d6-d64c-41a2-b396-c3bb243af79c
x_opt1 = stochastic_gradient_descent(M, gradF, x)

# ╔═╡ a6d7707d-8d8b-4136-b058-60c1a33b182d
@benchmark stochastic_gradient_descent($M, $gradF, $x)

# ╔═╡ 201d9ab2-2848-472e-abd6-826301c328e3
x_opt2 = stochastic_gradient_descent(M, gradf, x)

# ╔═╡ ce5e6f7b-f101-401f-9470-74ae94b87096
@benchmark stochastic_gradient_descent($M, $gradf, $x)

# ╔═╡ b463184b-3b66-4c1d-810c-bb398d918e2a
x_opt2

# ╔═╡ f2093d02-a6e0-49f3-ad39-55d9d88e75d0
md"""This result is reasonably close. But we can improve it by using a `DirectionUpdateRule`, namely:

On the one hand [`MomentumGradient`](https://manoptjl.org/stable/solvers/gradient_descent.html#Manopt.MomentumGradient), which requires both the manifold and the initial value,  in order to keep track of the iterate and parallel transport the last direction to the current iterate.
You can also set a `vector_transport_method`, if `ParallelTransport()` is not
available on your manifold. Here, we simply do
"""

# ╔═╡ 22c66c8a-3db1-4584-8cb1-93fbbeb34d4e
x_opt3 = stochastic_gradient_descent(
    M, gradf, x; direction=MomentumGradient(M, x, StochasticGradient(zero_vector(M, x)))
)

# ╔═╡ dbefa2af-d68f-4918-8ab0-4af257425686
MG = MomentumGradient(M, x, StochasticGradient(zero_vector(M, x)));

# ╔═╡ 5d5122b6-ba71-48ef-b459-d676ad4cca0d
@benchmark stochastic_gradient_descent($M, $gradf, $x; direction=$MG)

# ╔═╡ d7526835-73d3-40e6-8f8d-1711519ee57a
md"""
And on the other hand the [`AverageGradient`](https://manoptjl.org/stable/solvers/gradient_descent.html#Manopt.AverageGradient) computes an average of the last `n` gradients, i.e.
"""

# ╔═╡ a23ee4d6-6e9b-413a-a3ee-75367c9ea943
x_opt4 = stochastic_gradient_descent(
    M, gradf, x; direction=AverageGradient(M, x, 10, StochasticGradient(zero_vector(M, x)))
);

# ╔═╡ b0daf769-f8ea-4d7d-9bd4-48748bc02346
AG = AverageGradient(M, x, 10, StochasticGradient(zero_vector(M, x)));

# ╔═╡ 8b7cbfc8-7ec5-4297-b8f1-d3691725578e
@benchmark stochastic_gradient_descent($M, $gradf, $x; direction=$AG)

# ╔═╡ 5a3cb657-b25b-4622-a5a1-c4e93a17204f
md"""
Note that the default `StoppingCriterion` is a fixed number of iterations which helps the comparison here.

For both update rules we have to internally specify that we are still in the stochastic setting (since both rules can also be used with the `IdentityUpdateRule` within [`gradient_descent`](file:///Users/ronny/Repositories/Julia/Manopt.jl/docs/build/solvers/gradient_descent.html).

For this not-that-large-scale example we can of course also use a gradient descent with `ArmijoLinesearch`, but it will be a little slower usually
"""

# ╔═╡ b60b65a4-a46f-4f12-a416-48a716fa7386
fullGradF(M, x) = sum(grad_distance(M, p, x) for p in data)

# ╔═╡ ed9d3864-9a24-4c9a-938b-c2d3c59bb97e
x_opt5 = gradient_descent(M, F, fullGradF, x; stepsize=ArmijoLinesearch())

# ╔═╡ 534bbae1-c777-460b-bdc3-9d4b78132644
AL = ArmijoLinesearch();

# ╔═╡ dba33908-3a1b-491a-abd8-497a00d85230
@benchmark gradient_descent($M, $F, $fullGradF, $x; stepsize=$AL)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
Manifolds = "1cead3c2-87b3-11e9-0ccd-23c62b72b94e"
Manopt = "0fc0a36d-df90-57f3-8f93-d78a9fc72bb5"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
BenchmarkTools = "~1.3.1"
Colors = "~0.12.8"
Manifolds = "~0.7.8"
Manopt = "~0.3.23"
PlutoUI = "~0.7.38"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.0"
manifest_format = "2.0"
project_hash = "9241760fb1a4c59f8a6206a3d8d66853b28424bd"

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
version = "1.1.1"

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

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "4c10eee4af024676200bc7752e536f858c6b8f93"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.1"

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
version = "0.5.2+0"

[[deps.CovarianceEstimation]]
deps = ["LinearAlgebra", "Statistics", "StatsBase"]
git-tree-sha1 = "a3e070133acab996660d31dcf479ea42849e368f"
uuid = "587fd27a-f159-11e8-2dae-1979310e6154"
version = "0.2.7"

[[deps.DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

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
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.Einsum]]
deps = ["Compat"]
git-tree-sha1 = "4a6b3eee0161c89700b6c1949feae8b851da5494"
uuid = "b7d42ee7-0b51-5a75-98ca-779d3107e4c0"
version = "0.4.1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

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
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

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
version = "2.28.0+0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NamedDims]]
deps = ["AbstractFFTs", "ChainRulesCore", "CovarianceEstimation", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "0856b62716585eb90cc1dada226ac9eab5f69aa5"
uuid = "356022a1-0364-5f58-8944-0da4b18d706f"
version = "0.2.47"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

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
git-tree-sha1 = "3b429f37de37f1fc603cc1de4a799dc7fbe4c0b6"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "670e559e5c8e191ded66fa9ea89c97f10376bb4c"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.38"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

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
version = "0.7.0"

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
git-tree-sha1 = "cd56bf18ed715e8b09f06ef8c6b781e6cdc49911"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c82aaa13b44ea00134f8c9c89819477bd3986ecd"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.3.0"

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
version = "1.0.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

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
version = "1.2.12+3"

[[deps.ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╟─f58f28fe-c3d8-11ec-0acb-e948605dd63f
# ╠═8a8be09a-49e9-4563-9afc-ae72bd05d26d
# ╟─84528b83-f11d-42de-9e86-fc9f3d451500
# ╠═c104cad0-c489-42f8-b896-986811eebfef
# ╟─11ffe7e1-22f5-4098-9830-d56ff59745ee
# ╠═f3a29d0e-91a5-437f-8681-5b72072b9cdf
# ╠═4f527a26-dc55-4eb9-ab1b-62a4712182d5
# ╠═8baf553d-8d22-4f13-9741-43531b080998
# ╠═daf0dc47-cba2-4264-84d9-47137924addd
# ╟─30f0a209-afde-40c4-a8ab-3f40fd374d3c
# ╠═12a60ec6-cd18-4eea-a0af-0cabd43686b8
# ╠═d76aa8ef-b234-40d6-940b-cfc5db38cee8
# ╠═d41271a4-cbbe-46a3-b350-fce79d860671
# ╟─2d977c61-c556-4d58-8ca6-de0f7977cd30
# ╠═2b1ed0d6-d64c-41a2-b396-c3bb243af79c
# ╠═a6d7707d-8d8b-4136-b058-60c1a33b182d
# ╠═201d9ab2-2848-472e-abd6-826301c328e3
# ╠═ce5e6f7b-f101-401f-9470-74ae94b87096
# ╠═b463184b-3b66-4c1d-810c-bb398d918e2a
# ╟─f2093d02-a6e0-49f3-ad39-55d9d88e75d0
# ╠═22c66c8a-3db1-4584-8cb1-93fbbeb34d4e
# ╠═dbefa2af-d68f-4918-8ab0-4af257425686
# ╠═5d5122b6-ba71-48ef-b459-d676ad4cca0d
# ╟─d7526835-73d3-40e6-8f8d-1711519ee57a
# ╠═a23ee4d6-6e9b-413a-a3ee-75367c9ea943
# ╠═b0daf769-f8ea-4d7d-9bd4-48748bc02346
# ╠═8b7cbfc8-7ec5-4297-b8f1-d3691725578e
# ╟─5a3cb657-b25b-4622-a5a1-c4e93a17204f
# ╠═b60b65a4-a46f-4f12-a416-48a716fa7386
# ╠═ed9d3864-9a24-4c9a-938b-c2d3c59bb97e
# ╠═534bbae1-c777-460b-bdc3-9d4b78132644
# ╠═dba33908-3a1b-491a-abd8-497a00d85230
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
