### A Pluto.jl notebook ###
# v0.19.16

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ b198b15d-e547-47f7-a274-9ca0bc2331d6
using Pkg;

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
The following is a little bit of setup to save/include the generated images. If you are running the package locally and have Asymptote installed, you can set `render_asy` to true to generate the images.

If you downloaded only the notebook, the code runs but the images might not show.
"""

# ╔═╡ 65489070-9066-46bb-b5b1-52732dbe9bc7
md"""
# Example

To get started with our example we first have to load the necessary packages.
"""

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

# ╔═╡ 6cd0aaab-e0bf-428e-a84c-f456a32f8e36
md"""
In the following image the mean (orange), median (magenta) are shown.
"""

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

# ╔═╡ 31880e73-c585-4519-9d51-5523ea2f2c14
md"""
## Technical Details

This notebook tries to be able to run in both in a __development__ style mode, since for the documentation we want to make sure, the output refers to the most resent version, that is also with `Manopt.jl` in `]Pkg.develop` mode.

It should also be able to run with a __tutorial__ mode, where this notebook uses the internal package management of `Pluto.jl`.

A mix of both is __automatic__, which checks whether the folder containing this notebook is called “tutorials” and the parent folder contains `Manopt.jl`s main file. Note that also development checks this before but would error if the prequisists are not fulfilled.
"""

# ╔═╡ e023bf0e-1a5f-4c48-b0f7-a4234260138c
# hideall
nb_default_mode = :auto;

# ╔═╡ 14ddaae2-e0cd-445d-b38d-0d37ea1c6e31
# hideall
@bind nb_mode Select([:develop => "development 🛠️", :tutorial => "Tutorial 📖", :auto => "auto ⚙️"]; default=nb_default_mode)

# ╔═╡ 750d07d8-63a2-4507-95d2-6a219301bb79
# hideall
begin
	Pkg.activate(; temp=true)
    packages = [
  		PackageSpec(; name="Manifolds", version="0.8"),
   		PackageSpec(; name="Colors", version="0.12"),
   		PackageSpec(; name="PlutoUI", version="0.7"),
	]
	manopt_pkg = PackageSpec(; name="Manopt", version="0.3.51")
	if nb_mode === :develop || nb_mode === :auto
		curr_folder = pwd()
		parent_folder = dirname(curr_folder)
		manopt_file = joinpath(parent_folder,"src","Manopt.jl")
		if endswith(curr_folder,"tutorials") && isfile(manopt_file)
			# Manopt in dev mode
			Pkg.develop(path=parent_folder)
			nb_dev = true
		else
			nb_dev = false
			push!(packages, manopt_pkg)
			if nb_mode === :develop
				error("Development mode not possible, we are either not in the `tutorials/` folder or in `../src/` there is no `Manopt.jl` file,");
			end
		end
	else
		nb_dev = false
		push!(packages, manopt_pkg)
	end
	Pkg.add(packages)
end;

# ╔═╡ 4235a1ba-3cf2-49dc-9a26-32fafc7a7008
begin
	localpath = join(splitpath(@__FILE__)[1:(end - 1)], "/") # files folder
    image_prefix = localpath * "/optimize"
    nb_dev && (@info image_prefix)
    render_asy = false # on CI or when you do not have asymptote, this should be false
end;

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

# ╔═╡ 97e1bec2-3577-423a-a012-ca1b5413b29a
PlutoUI.LocalResource(image_prefix * "/startDataCenterMedianAndMean.png")

# ╔═╡ 3d010940-aa2e-44d3-988b-5d6b7bec54f0
# hideall
nb_dev && Pkg.status(["Manopt", "Manifolds"]);

# ╔═╡ Cell order:
# ╟─6bf76330-ad0e-11ec-0c00-894872624127
# ╟─960f171c-4f52-4104-a827-c6b918b7538d
# ╟─b198b15d-e547-47f7-a274-9ca0bc2331d6
# ╟─750d07d8-63a2-4507-95d2-6a219301bb79
# ╠═4235a1ba-3cf2-49dc-9a26-32fafc7a7008
# ╟─65489070-9066-46bb-b5b1-52732dbe9bc7
# ╠═727ca485-8350-4adc-9aa0-59bbb84a9205
# ╟─3d010940-aa2e-44d3-988b-5d6b7bec54f0
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
# ╟─31880e73-c585-4519-9d51-5523ea2f2c14
# ╟─e023bf0e-1a5f-4c48-b0f7-a4234260138c
# ╠═14ddaae2-e0cd-445d-b38d-0d37ea1c6e31
