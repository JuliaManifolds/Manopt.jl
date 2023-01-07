### A Pluto.jl notebook ###
# v0.19.16

using Markdown
using InteractiveUtils

# ╔═╡ 727ca485-8350-4adc-9aa0-59bbb84a9205
using Manopt, Manifolds, Random, Colors, PlutoUI

# ╔═╡ 6bf76330-ad0e-11ec-0c00-894872624127
md"""
# Get Started: Optimize!

In this tutorial, we want to use `Manopt.jl` solve the optimization problem

```math
\operatorname*{argmin}_{p ∈ \mathcal M} f(p)
```

where ``\mathcal M`` is a [Riemannian manifold](https://juliamanifolds.github.io/Manifolds.jl/stable/interface.html#ManifoldsBase.Manifold) and ``f:\mathcal M → ℝ`` is the cost function.

We will take a loot at how to define the optimisation problem, that is the manifold, the cost functin and (in our example) a gradient to call one specific solver.

After that we will dive into general keyword arguments that are available for all solvers as well as specific keywords that are available for the solver we consider here.

We will finally consider a nonsmooth example with a second, a little bit more complicated solver.

This tutorial is a [Pluto 🎈 notebook](https://github.com/fonsp/Pluto.jl), so if you are reading the `Manopt.jl` documentation you can also [download](https://github.com/JuliaManifolds/Manopt.jl/raw/master/tutorials/Optimize!.jl) the notebook and run it yourself within Pluto.
"""

# ╔═╡ 960f171c-4f52-4104-a827-c6b918b7538d
md"""
## Setup

If you open this notebook in Pluto locally it switches between two modes.
If the tutorial is within the `Manopt.jl` repository, this notebook tries to use the local package in development mode.
Otherwise, the file uses the Pluto pacakge management version.
In this case, the includsion of images might be broken. unless you create a subfolder `optimize` and activate `asy`-rendering.
"""

# ╔═╡ a3c81cc3-02a9-41cd-8eb7-3d665d42c024
# hideall
_nb_mode = :auto;

# ╔═╡ b198b15d-e547-47f7-a274-9ca0bc2331d6
# hideall
if _nb_mode === :auto || _nb_mode === :development
	import Pkg
	curr_folder = pwd()
	parent_folder = dirname(curr_folder)
	manopt_file = joinpath(parent_folder,"src","Manopt.jl")
	# the tutorial is still in the package and not standalone
	_in_package =  endswith(curr_folder,"tutorials") && isfile(manopt_file)
	if _in_package
		eval(:(Pkg.develop(path=parent_folder)))  # directory of MyPkg
	end
else
	_in_package = false
end;

# ╔═╡ 4235a1ba-3cf2-49dc-9a26-32fafc7a7008
# hideall
begin
	localpath = join(splitpath(@__FILE__)[1:(end - 1)], "/") # files folder
	image_prefix = localpath * "/optimize"
    _in_package && (@info image_prefix)
    render_asy = false # on CI or when you do not have asymptote, this should be false
end;

# ╔═╡ 65489070-9066-46bb-b5b1-52732dbe9bc7
md"""
# Example

To get started with our example we first have to load the necessary packages.
"""

# ╔═╡ 3d010940-aa2e-44d3-988b-5d6b7bec54f0
# hideall
_in_package && Pkg.status(["Manopt", "Manifolds"]);

# ╔═╡ 177cc292-94d3-4344-857e-30483f592a55
md"""
Let's load a few colors from [Paul Tol](https://personal.sron.nl/~pault/).
"""

# ╔═╡ 0b405c42-19a5-480d-b1dc-0fb8811a48fa
begin
    black = RGBA{Float64}(colorant"#000000")
    TolVibrantOrange = RGBA{Float64}(colorant"#EE7733")
    TolVibrantBlue = RGBA{Float64}(colorant"#0077BB")
    TolVibrantTeal = RGBA{Float64}(colorant"#009988")
    TolVibrantMagenta = RGBA{Float64}(colorant"#EE3377")
end;

# ╔═╡ 803fc640-bbed-4700-8a1e-f414c6446eea
md"""
# The data set
"""

# ╔═╡ 278dbd3c-aa91-4d9d-ad49-c3b4b336efe2
md"""
We take a look at a srandom set of points.
"""

# ╔═╡ 7e0301fb-7465-410c-b47c-04686bf44ab1
begin
    n = 100
    σ = π / 8
    M = Sphere(2)
    x = 1 / sqrt(2) * [1.0, 0.0, 1.0]
    Random.seed!(42)
    data = [exp(M, x,  σ * rand(M; vector_at=x)) for i in 1:n]
end

# ╔═╡ 9b130a57-293d-429d-88b5-78bfacbf836f
asymptote_export_S2_signals(
    image_prefix * "/startDataAndCenter.asy";
    points=[[x], data],
    colors=Dict(:points => [TolVibrantBlue, TolVibrantTeal]),
    dot_size=3.5,
    camera_position=(1.0, 0.5, 0.5),
)

# ╔═╡ d2bc2e63-6ae7-4f54-b176-f74e66365b1a
render_asy && render_asymptote(image_prefix * "/startDataAndCenter.asy"; render=2)

# ╔═╡ 6ea36d50-96f0-46e1-8d90-529f0b23120d
PlutoUI.LocalResource(image_prefix * "/startDataAndCenter.png")

# ╔═╡ e21d6d03-4c61-457b-a9c7-fad5b4f369db
md"""
## Computing the Mean

To compute the mean on the manifold we use the characterization that the
Euclidean mean minimizes the sum of squared distances, and end up with the
following cost function. Its minimizer is called
[Riemannian Center of Mass](https://arxiv.org/abs/1407.2087).

> **Note.**
> There are more sophisticated methods tailored for the specific manifolds available in [Manifolds.jl](https://juliamanifolds.github.io/Manifolds.jl/) see [mean](https://juliamanifolds.github.io/Manifolds.jl/stable/features/statistics.html#Statistics.mean-Tuple{Manifold,AbstractArray{T,1}%20where%20T,AbstractArray{T,1}%20where%20T,ExtrinsicEstimation}).
"""

# ╔═╡ 515eaa1d-8307-45ba-ae63-be070dc2ff1c
F(M, y) = sum(1 / (2 * n) * distance.(Ref(M), Ref(y), data) .^ 2)

# ╔═╡ 00bc7b5b-7cc9-43bc-bf96-b5d2b85ddb9a
gradF(M, y) = sum(1 / n * grad_distance.(Ref(M), data, Ref(y)))

# ╔═╡ 3cc7ce2b-8ae6-4ba9-8298-2050fe7081eb
md"""
Note that the [grad_distance](https://manoptjl.org/stable/functions/gradients.html#Manopt.grad_distance) defaults to the case `p=2`, i.e. the
gradient of the squared distance. For details on convergence of the gradient
descent for this problem, see [^AfsariTronVidal2013].

The easiest way to call the gradient descent is now to call
[gradient_descent](https://manoptjl.org/stable/solvers/gradient_descent.html#Manopt.gradient_descent).
"""

# ╔═╡ c067f4d1-b54b-4228-85cf-70cdbbdc948b
gradient_descent(M, F, gradF, data[1])

# ╔═╡ 7aee0626-14ba-431e-af84-79c2dfc021da
md"""
In order to get more details, we further add the `debug=` keyword argument, which
act as a [decorator pattern](https://en.wikipedia.org/wiki/Decorator_pattern).

The following debug prints

```juliaREPL
# i | Last Change: | F(x): | x:
```
as well as the reason why the algorithm stopped at the end.

The elements passed to `debug=` are postprocessed, there are specifiic symbols and formats we can use. For example `:Iteration` to plot the iteratin, but we can also add a format for the print using `(:Change, "| Last change: %e3.8")`, i.e. the second string is a format for printf.

Note that here we use `PlutoUI` to see the output also here in the notebook
"""

# ╔═╡ 51daafb5-84a0-47fd-ac40-4d53888cd914
with_terminal() do
    global xMean = gradient_descent(
        M,
        F,
        gradF,
        data[1];
        debug=[
            :Iteration,
            (:Change, "change: %1.9f | "),
            (:Cost, " F(x): %1.11f | "),
            "\n",
            :Stop,
        ],
    )
end

# ╔═╡ 863bf8b8-272c-40d6-985f-0a7cf9454756
md"""
While this works fine and finds a point with a very small gradient norm, the default way to determine the stepsize (though using [`ArmijoLinesearch`](https://manoptjl.org/stable/plans/index.html#Manopt.ArmijoLinesearch)) might be a little bit conservative, since for a start we aim for robust defaults.

We can tweak the default values for the `contraction_factor` and the `sufficient_decrease`  of the Armijo linesearch to improve there.
We can further use the `:Stepsize` symbol in the `debug=` specification to also print the obtained step size by Armijo.
We get
"""

# ╔═╡ 38df2fb3-f742-4652-857c-baa403985ff8
with_terminal() do
    global xMean2 = gradient_descent(
        M,
        F,
        gradF,
        data[1];
        stepsize=ArmijoLinesearch(M; contraction_factor=0.999, sufficient_decrease=0.5),
        debug=[
            :Iteration,
            (:Change, "change: %1.9f | "),
            (:Cost, " F(x): %1.11f | "),
            (:Stepsize, " | s: %1.3f"),
            "\n",
            :Stop,
        ],
    )
end

# ╔═╡ 244eb6ea-0bdb-4443-8dc1-40419966198a
md"""
which finishes in 5 steaps, with (numerically) the same cost.
"""

# ╔═╡ eedfedbc-f305-48ff-8aa7-78b6aa6c4d02
F(M, xMean) - F(M, xMean2)

# ╔═╡ 887bd166-c278-48d3-bfb1-31c7026433b1
md"""
Note that other optimization tasks may have other speedup opportunities.

For even more precision, we can further require a smaller gradient norm.
This is done by changing the `StoppingCriterion` used, where several
criteria can be combined using `&` and/or `|`.
For example in the following case, we want to stop if either the gradient norm is _very_ small (`1e-15`) or if we reach 200 iterations, which can be seen as a fallback.
"""

# ╔═╡ a99a5603-6ef5-43e8-a082-54dd20226956
with_terminal() do
    global xMean3 = gradient_descent(
        M,
        F,
        gradF,
        data[1];
	    stepsize=ArmijoLinesearch(M; contraction_factor=0.999, sufficient_decrease=0.5),
        debug=[
            :Iteration,
            (:Change, "change: %1.9f | "),
            (:Cost, " F(x): %1.11f | "),
            (:Stepsize, " | s: %1.3f"),
            "\n",
            :Stop,
        ],
        stopping_criterion=StopWhenGradientNormLess(1e-15) | StopAfterIteration(200),
    )
end

# ╔═╡ 98028747-31dd-4bf8-b4b5-0959d5afb75c
md"""
Let's add this point to our data image
"""

# ╔═╡ fb07943f-54b4-4cb3-b1fd-f3ab06b4d033
asymptote_export_S2_signals(
    image_prefix * "/startDataCenterMean.asy";
    points=[[x], data, [xMean3]],
    colors=Dict(:points => [TolVibrantBlue, TolVibrantTeal, TolVibrantOrange]),
    dot_size=3.5,
    camera_position=(1.0, 0.5, 0.5),
)

# ╔═╡ 5e1e6db2-39da-4857-8745-eda6ae510fa8
render_asy && render_asymptote(image_prefix * "/startDataCenterMean.asy"; render=2);

# ╔═╡ c9e09455-1af2-40aa-aa05-7fa329b5eec7
PlutoUI.LocalResource(image_prefix * "/startDataCenterMean.png")

# ╔═╡ a17ddd91-0839-48fe-ab78-445226ee4ff9
md"""
## Computing the Median

> **Note.**
> There are more sophisticated methods tailored for the specific manifolds available in
> [Manifolds.jl](https://juliamanifolds.github.io/Manifolds.jl/) see [`median`](https://juliamanifolds.github.io/Manifolds.jl/stable/features/statistics.html#Statistics.median-Tuple{Manifold,AbstractArray{T,1}%20where%20T,AbstractArray{T,1}%20where%20T,CyclicProximalPointEstimation}).

Similarly to the mean, you can also define the median as the minimizer of the
distances, see for example [^Bačák2014], but since
this problem is not differentiable, we employ the Cyclic Proximal Point (CPP)
algorithm, described in the same reference. We define
"""

# ╔═╡ 439b700d-dce7-43c4-bb1e-1f263a3f54a4
F2(M, y) = sum(1 / (2 * n) * distance.(Ref(M), Ref(y), data))

# ╔═╡ 79921477-db82-424f-8f8e-9bade1cd63c3
proxes = Function[(M, λ, y) -> prox_distance(M, λ / n, di, y, 1) for di in data]

# ╔═╡ dc236508-283c-4047-a216-c570e35bc791
md"""
So we call the cyclic proximal point algorithm, this time with a recording, and activate the return of the complete options to access the recorded values. We further increase the display of the cost function to more digits.
"""

# ╔═╡ d2a8250e-7796-454b-a0bf-9970b1b9a2aa
with_terminal() do
    global o = cyclic_proximal_point(
        M,
        F2,
        proxes,
        data[1];
        debug=[
            :Iteration,
            " | ",
            :Change,
            " | ",
            (:Cost, "F(x): %1.12f"),
            " | ",
            :Iterate,
            "\n",
            50,
            :Stop,
        ],
        record=[:Iteration, :Change, :Cost, :Iterate],
        return_state=true,
    )
end

# ╔═╡ 90414c9b-b164-490c-b939-2472d8284887
xMedian = get_solver_result(o)

# ╔═╡ aa88f273-2a01-4d93-857b-8b9c19fcdd1a
md"""
where the differences to `gradient_descent` are as follows

* the third parameter is now an Array of proximal maps
* debug is reduced to only every 50th iteration
* we further activated a `RecordAction` using the `record=` optional
  parameter. These work very similarly to those in debug, but they
  collect their data in an array. The high level interface then returns two
  variables; the `values` do contain an array of recorded
  datum per iteration. Here a tuple containing the iteration, last change and
  cost respectively; see [RedordOptions](https://manoptjl.org/stable/plans/index.html#RecordSolverState-1) for details.

We can access the recorded values using `get_record`, that consists of a tuple per iteration and contains the iteration number, the change and the cost.
"""

# ╔═╡ c835b5ec-085e-4c9d-b777-76036515bcd1
values = get_record(o)

# ╔═╡ 9a19d3bb-4487-4469-a856-b1a0f3f540a7
asymptote_export_S2_signals(
    image_prefix * "/startDataCenterMedianAndMean.asy";
    points=[[x], data, [xMean], [xMedian]],
    colors=Dict(
        :points => [TolVibrantBlue, TolVibrantTeal, TolVibrantOrange, TolVibrantMagenta]
    ),
    dot_size=3.5,
    camera_position=(1.0, 0.5, 0.5),
)

# ╔═╡ e6e8cc14-a67b-4e6a-a83b-73f9ba1e1dcb
render_asy && render_asymptote(image_prefix * "/startDataCenterMedianAndMean.asy"; render=2)

# ╔═╡ 6cd0aaab-e0bf-428e-a84c-f456a32f8e36
md"""
In the following image the mean (orange), median (magenta) are shown.
"""

# ╔═╡ 97e1bec2-3577-423a-a012-ca1b5413b29a
PlutoUI.LocalResource(image_prefix * "/startDataCenterMedianAndMean.png")

# ╔═╡ 0ba01385-6502-4e66-afee-4c4af391c9b9
md"""
## Literature

[^Bačák2014]:
	> Bačák, M: __Computing Medians and Means in Hadamard Spaces.__
    > SIAM Journal on Optimization, Volume 24, Number 3, pp. 1542–1566,
    > doi: [10.1137/140953393](https://doi.org/10.1137/140953393),
    > arxiv: [1210.2145](https://arxiv.org/abs/1210.2145)
[^AfsariTronVidal2013]:
	> Afsari, B; Tron, R.; Vidal, R.: __On the Convergence of Gradient Descent for Finding the Riemannian Center of Mass__,
	> SIAM Journal on Control and Optimization, Volume 51, Issue 3, pp. 2230–2260.
	> doi: [10.1137/12086282X](https://doi.org/10.1137/12086282X),
	> arxiv: [1201.0925](https://arxiv.org/abs/1201.0925).
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
Manifolds = "1cead3c2-87b3-11e9-0ccd-23c62b72b94e"
Manopt = "0fc0a36d-df90-57f3-8f93-d78a9fc72bb5"
Pkg = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
Colors = "~0.12.10"
Manifolds = "~0.8.42"
Manopt = "0.3, 0.4"
PlutoUI = "~0.7.49"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.4"
manifest_format = "2.0"
project_hash = "d4c5e42e1e0e4be511f9bab7530080bc8ef4713b"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "69f7020bd72f069c219b5e8c236c1fa90d2cb409"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.2.1"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "195c5505521008abea5aee4f96930717958eac6f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.4.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "62e51b39331de8911e4a7ff6f5aaf38a5f4cc0ae"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.2.0"

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "14c3f84a763848906ac681f94cf469a851601d92"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.28"

[[deps.ArrayInterfaceStaticArraysCore]]
deps = ["Adapt", "ArrayInterfaceCore", "LinearAlgebra", "StaticArraysCore"]
git-tree-sha1 = "93c8ba53d8d26e124a5a8d4ec914c3a16e6a0970"
uuid = "dd5226c6-a4d4-4bc7-8575-46859f9c95b9"
version = "0.1.3"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e7ff6cadf743c098e08fca25c91103ee4303c9bb"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.6"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random", "SnoopPrecompile"]
git-tree-sha1 = "aa3edc8f8dea6cbfa176ee12f7c2fc82f0608ed3"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.20.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "600cc5508d66b78aae350f7accdb58763ac18589"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.10"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "00a2cccc7f098ff3b66806862d275ca3db9e6e5a"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.5.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.1+0"

[[deps.CovarianceEstimation]]
deps = ["LinearAlgebra", "Statistics", "StatsBase"]
git-tree-sha1 = "3c8de95b4e932d76ec8960e12d681eba580e9674"
uuid = "587fd27a-f159-11e8-2dae-1979310e6154"
version = "0.2.8"

[[deps.DataAPI]]
git-tree-sha1 = "e8119c1a33d267e16108be441a287a6981ba1630"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.14.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

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
git-tree-sha1 = "a7756d098cbabec6b3ac44f369f74915e8cfd70a"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.79"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.Einsum]]
deps = ["Compat"]
git-tree-sha1 = "4a6b3eee0161c89700b6c1949feae8b851da5494"
uuid = "b7d42ee7-0b51-5a75-98ca-779d3107e4c0"
version = "0.4.1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "9a0472ec2f5409db243160a8b030f94c380167a3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.6"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "6872f5ec8fd1a38880f027a26739d42dcda6691f"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.2"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "ba2d094a88b6b287bd25cfa86f301e7693ffae2f"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.7.4"

[[deps.HybridArrays]]
deps = ["LinearAlgebra", "Requires", "StaticArrays"]
git-tree-sha1 = "0de633a951f8b5bd32febc373588517aa9f2f482"
uuid = "1baab800-613f-4b0a-84e4-9cd3431bfbb9"
version = "0.4.13"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions", "Test"]
git-tree-sha1 = "709d864e3ed6e3545230601f94e11ebc65994641"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.11"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.Inflate]]
git-tree-sha1 = "5cd07aab533df5170988219191dfad0519391428"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.3"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "49510dfcb407e572524ba94aeae2fced1f3feb0f"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.8"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

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
git-tree-sha1 = "78d9909daf659c901ae6c7b9de7861ba45a743f4"
uuid = "2c470bb0-bcc8-11e8-3dad-c9649493f05e"
version = "0.5.3"

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

[[deps.LinearMaps]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics"]
git-tree-sha1 = "d1b46faefb7c2f48fdec69e6f3cc34857769bc15"
uuid = "7a12625a-238d-50fd-b39a-03d52299707e"
version = "3.8.0"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "946607f84feb96220f480e0422d3484c49c00239"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.19"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.Manifolds]]
deps = ["Colors", "Distributions", "Einsum", "Graphs", "HybridArrays", "Kronecker", "LinearAlgebra", "ManifoldsBase", "Markdown", "MatrixEquations", "Quaternions", "Random", "RecipesBase", "RecursiveArrayTools", "Requires", "SimpleWeightedGraphs", "SpecialFunctions", "StaticArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "57300c1019bad5c89f398f198212fbaa87ff6b4a"
uuid = "1cead3c2-87b3-11e9-0ccd-23c62b72b94e"
version = "0.8.42"

[[deps.ManifoldsBase]]
deps = ["LinearAlgebra", "Markdown", "Random"]
git-tree-sha1 = "c92e14536ba3c1b854676ba067926dbffe3624a9"
uuid = "3362f125-f0bb-47a3-aa74-596ffd7ef2fb"
version = "0.13.28"

[[deps.Manopt]]
deps = ["ColorSchemes", "ColorTypes", "Colors", "DataStructures", "Dates", "LinearAlgebra", "ManifoldsBase", "Markdown", "Printf", "Random", "Requires", "SparseArrays", "StaticArrays", "Statistics", "Test"]
path = "/Users/ronnber/Repositories/Julia/Manopt.jl"
uuid = "0fc0a36d-df90-57f3-8f93-d78a9fc72bb5"
version = "0.4.0"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MatrixEquations]]
deps = ["LinearAlgebra", "LinearMaps"]
git-tree-sha1 = "3b284e9c98f645232f9cf07d4118093801729d43"
uuid = "99c1a7ee-ab34-5fd5-8076-27c950a045f4"
version = "2.2.2"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[deps.NamedDims]]
deps = ["AbstractFFTs", "ChainRulesCore", "CovarianceEstimation", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "cb8ebcee2b4e07b72befb9def593baef8aa12f07"
uuid = "356022a1-0364-5f58-8944-0da4b18d706f"
version = "0.2.50"

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
git-tree-sha1 = "cf494dca75a69712a72b80bc48f59dcf3dea63ec"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.16"

[[deps.Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "6466e524967496866901a78fca3f2e9ea445a559"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.2"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "eadad7b14cf046de6eb41f13c9275e5aa2711ab6"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.49"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "97aa253e65b784fd13e83774cadc95b38011d734"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.6.0"

[[deps.Quaternions]]
deps = ["LinearAlgebra", "Random", "RealDot"]
git-tree-sha1 = "a3c34ce146e39c9e313196bb853894c133f3a555"
uuid = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
version = "0.7.3"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
deps = ["SnoopPrecompile"]
git-tree-sha1 = "18c35ed630d7229c5584b945641a73ca83fb5213"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.2"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterfaceCore", "ArrayInterfaceStaticArraysCore", "ChainRulesCore", "DocStringExtensions", "FillArrays", "GPUArraysCore", "IteratorInterfaceExtensions", "LinearAlgebra", "RecipesBase", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables", "ZygoteRules"]
git-tree-sha1 = "66e6a85fd5469429a3ac30de1bd491e48a6bac00"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "2.34.1"

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

[[deps.SnoopPrecompile]]
git-tree-sha1 = "f604441450a3c0569830946e5b33b78c928e1a85"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.1"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "a4ada03f999bd01b3a25dcaa30b2d929fe537e00"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.0"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "6954a456979f23d05085727adb17c4551c19ecd1"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.12"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "ab6083f09b3e617e34a956b43e9d51b824206932"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.1.1"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SymbolicIndexingInterface]]
deps = ["DocStringExtensions"]
git-tree-sha1 = "6b764c160547240d868be4e961a5037f47ad7379"
uuid = "2efcf032-c050-4f8e-a9bb-153293bab1f5"
version = "0.2.1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "c79322d36826aa2f4fd8ecfa96ddb47b174ac78d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.URIs]]
git-tree-sha1 = "ac00576f90d8a259f2c9d823e91d1de3fd44d348"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.1"

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
# ╟─6bf76330-ad0e-11ec-0c00-894872624127
# ╟─960f171c-4f52-4104-a827-c6b918b7538d
# ╠═a3c81cc3-02a9-41cd-8eb7-3d665d42c024
# ╠═b198b15d-e547-47f7-a274-9ca0bc2331d6
# ╠═4235a1ba-3cf2-49dc-9a26-32fafc7a7008
# ╟─65489070-9066-46bb-b5b1-52732dbe9bc7
# ╠═727ca485-8350-4adc-9aa0-59bbb84a9205
# ╠═3d010940-aa2e-44d3-988b-5d6b7bec54f0
# ╟─177cc292-94d3-4344-857e-30483f592a55
# ╠═0b405c42-19a5-480d-b1dc-0fb8811a48fa
# ╟─803fc640-bbed-4700-8a1e-f414c6446eea
# ╟─278dbd3c-aa91-4d9d-ad49-c3b4b336efe2
# ╠═7e0301fb-7465-410c-b47c-04686bf44ab1
# ╠═9b130a57-293d-429d-88b5-78bfacbf836f
# ╠═d2bc2e63-6ae7-4f54-b176-f74e66365b1a
# ╟─6ea36d50-96f0-46e1-8d90-529f0b23120d
# ╟─e21d6d03-4c61-457b-a9c7-fad5b4f369db
# ╠═515eaa1d-8307-45ba-ae63-be070dc2ff1c
# ╠═00bc7b5b-7cc9-43bc-bf96-b5d2b85ddb9a
# ╟─3cc7ce2b-8ae6-4ba9-8298-2050fe7081eb
# ╠═c067f4d1-b54b-4228-85cf-70cdbbdc948b
# ╟─7aee0626-14ba-431e-af84-79c2dfc021da
# ╠═51daafb5-84a0-47fd-ac40-4d53888cd914
# ╟─863bf8b8-272c-40d6-985f-0a7cf9454756
# ╠═38df2fb3-f742-4652-857c-baa403985ff8
# ╟─244eb6ea-0bdb-4443-8dc1-40419966198a
# ╠═eedfedbc-f305-48ff-8aa7-78b6aa6c4d02
# ╟─887bd166-c278-48d3-bfb1-31c7026433b1
# ╠═a99a5603-6ef5-43e8-a082-54dd20226956
# ╟─98028747-31dd-4bf8-b4b5-0959d5afb75c
# ╠═fb07943f-54b4-4cb3-b1fd-f3ab06b4d033
# ╠═5e1e6db2-39da-4857-8745-eda6ae510fa8
# ╟─c9e09455-1af2-40aa-aa05-7fa329b5eec7
# ╟─a17ddd91-0839-48fe-ab78-445226ee4ff9
# ╠═439b700d-dce7-43c4-bb1e-1f263a3f54a4
# ╠═79921477-db82-424f-8f8e-9bade1cd63c3
# ╟─dc236508-283c-4047-a216-c570e35bc791
# ╠═d2a8250e-7796-454b-a0bf-9970b1b9a2aa
# ╠═90414c9b-b164-490c-b939-2472d8284887
# ╟─aa88f273-2a01-4d93-857b-8b9c19fcdd1a
# ╠═c835b5ec-085e-4c9d-b777-76036515bcd1
# ╠═9a19d3bb-4487-4469-a856-b1a0f3f540a7
# ╠═e6e8cc14-a67b-4e6a-a83b-73f9ba1e1dcb
# ╟─6cd0aaab-e0bf-428e-a84c-f456a32f8e36
# ╟─97e1bec2-3577-423a-a012-ca1b5413b29a
# ╟─0ba01385-6502-4e66-afee-4c4af391c9b9
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
