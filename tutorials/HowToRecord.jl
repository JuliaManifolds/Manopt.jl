### A Pluto.jl notebook ###
# v0.19.16

using Markdown
using InteractiveUtils

# ╔═╡ 65e3c376-ad9f-11ec-003e-7f033e6865d8
md"""
# How to Record Data During the Iterations

The recording and debugging features make it possible to record nearly any data during the iterations.
This tutorial illustrates how to:

* record one value during the iterations;
* record multiple values during the iterations and access them afterwards;
* define an own `RecordAction` to perform individual recordings.

Several predefined recordings exist, for example `RecordCost()` or `RecordGradient()`, depending on the solver used.
For fields of the `State` this can be directly done using the [`RecordEntry(:field)`].
For other recordings, for example more advanced computations before storing a value, an own `RecordAction` can be defined.

We illustrate these using the gradient descent from the mean computation tutorial.

This tutorial is a [Pluto 🎈 notebook](https://github.com/fonsp/Pluto.jl), , so if you are reading the `Manopt.jl` documentation you can also [download](https://github.com/JuliaManifolds/Manopt.jl/raw/master/tutorials/HowToRecord.jl) the notebook and run it yourself within Pluto.
"""

# ╔═╡ 85cf1234-ae76-4cb7-9c14-366f25aa6f15
md"""
## Setup

If you open this notebook in Pluto locally it switches between two modes.
If the tutorial is within the `Manopt.jl` repository, this notebook tries to use the local package in development mode.
Otherwise, the file uses the Pluto pacakge management version.
"""

# ╔═╡ 94145efb-62fa-4b61-8e57-07d6c2ce66d8
# hideall
_nb_mode = :auto;

# ╔═╡ 02d0a5a5-8c12-4eec-8a3a-1269b5bb69c9
# hideall
begin
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
	using Manopt, Manifolds, Random, PlutoUI
end

# ╔═╡ cfd5a31a-2430-4e19-b1a8-ef2911109c0c
md"""
Now we can set up our small test example, which is just a gradient descent for the Riemannian center of mass, see the [Get Started: Optimize!](https://manoptjl.org/stable/tutorials/Optimize!/) tutorial for details.

Here we focus on ways to investigate the behaviour during iterations by using Recording techniques.
"""

# ╔═╡ c01983a2-33fc-4812-8071-229b7a0921e3
md"""
Since the loading is a little complicated, we show, which versions of packages were installed in the following.
"""

# ╔═╡ 40a09a75-9a7d-44bd-9ab6-4c77946c43ea
with_terminal() do
	Pkg.status()
end

# ╔═╡ 4331f82f-9778-4b12-8c2d-f007df8b213b
begin
    Random.seed!(42)
    m = 30
    M = Sphere(m)
    n = 800
    σ = π / 8
    x = zeros(Float64, m + 1)
    x[2] = 1.0
    data = [exp(M, x, σ * rand(M; vector_at=x)) for i in 1:n]
end

# ╔═╡ 604b880f-faf5-4d1c-a6b9-cd94e2a8bf46
F(M, y) = sum(1 / (2 * n) * distance.(Ref(M), Ref(y), data) .^ 2)

# ╔═╡ 066267a1-06e1-4eb7-a319-1b05f25b9eac
gradF(M, y) = sum(1 / n * grad_distance.(Ref(M), data, Ref(y)))

# ╔═╡ c822fe70-becd-4b0a-b66f-34366f98dfad
md"""
## Plain Examples

For the high level interfaces of the solvers, like [`gradient_descent`](https://manoptjl.org/stable/solvers/gradient_descent.html) we have to set `return_state` to `true` to obtain the whole options structure and not only the resulting minimizer.

Then we can easily use the `record=` option to add recorded values. This keyword accepts `RecordAction`s as well as several symbols as shortcuts, for example `:Cost` to record the cost, or if your options have a field `f`, `:f` would record that entry.
"""

# ╔═╡ fbc02960-7f7e-4500-92ac-30df7d058fa9
R = gradient_descent(M, F, gradF, data[1]; record=:Cost, return_state=true)

# ╔═╡ 7552c7ba-6d45-4ed9-856c-b00be28a84a0
md"""
From the returned options, we see that the `State` are encapsulated (decorated) with
`RecordSolverState`.

You can attach different recorders to some operations (`:Start`. `:Stop`, `:Iteration` at time of
 writing), where `:Iteration` is the default, so the following is the same as `get_record(R, :Iteation)`. We get
"""

# ╔═╡ bd4d1bce-7942-498f-a243-bc9090bb3b1c
get_record(R)

# ╔═╡ 0545973a-d77b-4cce-b834-38a9e08f9a17
md"""
To record more than one value, you can pass an array of a mix of symbols and `RecordAction` which formally introduces `RecordGroup`. Such a group records a tuple of values in every iteration.
"""

# ╔═╡ 68cb3e06-866d-4650-863a-61f965e3320f
R2 = gradient_descent(M, F, gradF, data[1]; record=[:Iteration, :Cost], return_state=true)

# ╔═╡ 4943d34f-0aff-4879-94a6-337ce42b2e36
md"""
Here, the symbol `:Cost` is mapped to using the `RecordCost` action. The same holds for `:Iteration` and `:Iterate` and any member field of the current `State`.
To access these you can first extract the group of records (that is where the `:Iteration`s are recorded – note the plural) and then access the `:Cost`
"""

# ╔═╡ 9d17df05-ca92-4a98-b101-c68ba8d3f7b9
get_record_action(R2, :Iteration)

# ╔═╡ edafb0ef-be70-4fed-97b1-eabeae9be5c2
md"""
`:Iteration` is the default here, i.e. something recorded through the iterations – and we can access the recorded data the same way as we specify them in the `record=` keyword, that is, using the indexing operation.
"""

# ╔═╡ 46a79917-e03e-4c99-9b13-c525b1abb585
get_record_action(R2)[:Cost]

# ╔═╡ ef47dee4-3ee9-45c2-9ebc-2598ea56aa75
md"""This can be also done by using a the high level interface `get_record`."""

# ╔═╡ a42b9205-2447-4b64-af23-9261c34ab284
get_record(R2, :Iteration, :Cost)

# ╔═╡ 44091233-12c7-4e2b-81c4-2b2d0aa488fa
md"""
Note that the first symbol again refers to the point where we record (not to the thing we record).
We can also pass a tuple as second argument to have our own order (not that now the second `:Iteration` refers to the recorded iterations).
"""

# ╔═╡ 98baba1a-d64d-4852-8f80-85f7248ece98
get_record(R2, :Iteration, (:Iteration, :Cost))

# ╔═╡ de36da12-d1bb-4b91-8d5f-7a2b64f70dd1
md"""
## A more Complex Example

To illustrate a complicated example let's record:
* the iteration number, cost and gradient field, but only every sixth iteration;
* the iteration at which we stop.

We first generate the problem and the options, to also illustrate the low-level works when not using `gradient_descent`.
"""

# ╔═╡ 5169f6ba-710c-4cef-8b99-9ce19300d634
p = DefaultManoptProblem(M, ManifoldGradientObjective(F, gradF))

# ╔═╡ 2015f505-cc26-4c1a-b512-3664fba1c0ed
o = GradientDescentState(
    M,
    copy(data[1]);
    stopping_criterion=StopAfterIteration(200) | StopWhenGradientNormLess(10.0^-9),
)

# ╔═╡ f079c326-d6bb-4710-9bd4-1133d74018d6
md"""
We first build a `RecordGroup` to group the three entries we want to record per iteration. We then put this into a `RecordEvery` to only record this every 6th iteration
"""

# ╔═╡ 3a915462-efac-4bc4-94d3-da0f400a2eaf
rI = RecordEvery(
    RecordGroup([
        :Iteration => RecordIteration(),
        :Cost => RecordCost(),
        :Gradient => RecordEntry(similar(data[1]), :X),
    ]),
    6,
)

# ╔═╡ cb2b3c30-d930-4de5-b919-39efc5e65fa1
md"""and a small option to record iterations"""

# ╔═╡ 930bb10c-cac0-48e2-99fa-f6faf3d0c519
sI = RecordIteration()

# ╔═╡ 995c264c-793e-4c23-9dfd-3eb7bc640e30
md"""
We now combine both into the `RecordSolverState` decorator. It acts completely the same as an `Option` but records something in every iteration additionally. This is stored in a dictionary of `RecordActions`, where `:Iteration` is the action (here the only every 6th iteration group) and the `sI` which is executed at stop.

Note that the keyword `record=` (in the high level interface `gradient_descent` only would fill the `:Iteration` symbol).
"""

# ╔═╡ 36965fe9-679d-49b8-8f0d-1cb4877cb5ab
r = RecordSolverState(o, Dict(:Iteration => rI, :Stop => sI))

# ╔═╡ abc01e8a-9ae5-483c-9cda-87a196d88918
md"""We now call the solver"""

# ╔═╡ 45ee91a6-0377-46e8-bdac-faea18b120e0
res = solve!(p, r)

# ╔═╡ 1c2c65f0-ccbd-4566-a8b3-d1449b683707
md"""
And we can check the recorded value at `:Stop` to see how many iterations were performed
"""

# ╔═╡ 8c6e70a2-3ebe-43a2-a96a-0219b64a1528
get_record(res, :Stop)

# ╔═╡ bba94b58-2815-4b83-8ee0-02baf9eefa77
md"""and the other values during the iterations are"""

# ╔═╡ 93af8680-a9d4-4c2d-8423-03eefb0fd752
get_record(res, :Iteration, (:Iteration, :Cost))

# ╔═╡ 5327d3b1-eb73-4725-bfa8-5d39b6993a6a
md"""
## Writing an own `RecordAction`s

Let's investigate where we want to count the number of function evaluations, again just to illustrate, since for the gradient this is just one evaluation per iteration.
We first define a cost, that counts its own calls.
"""

# ╔═╡ b3a8e43c-6c82-483f-9449-680ee21a0650
begin
    mutable struct MyCost{T}
        data::T
        count::Int
    end
    MyCost(data::T) where {T} = MyCost{T}(data, 0)
    function (c::MyCost)(M, x)
        c.count += 1
        return sum(1 / (2 * length(c.data)) * distance.(Ref(M), Ref(x), c.data) .^ 2)
    end
end

# ╔═╡ 05284884-e05c-4714-bf53-2da071c664f7
md"""
and we define the following RecordAction, which is a functor, i.e. a struct that is also a function. The function we have to implement is similar to a single solver step in signature, since it might get called every iteration:
"""

# ╔═╡ e0fe662d-edc5-4d1f-9fd6-987f37098cc4
begin
    mutable struct RecordCount <: RecordAction
        recorded_values::Vector{Int}
        RecordCount() = new(Vector{Int}())
    end
    function (r::RecordCount)(p::AbstractManoptProblem, ::AbstractManoptSolverState, i)
        if i > 0
            push!(r.recorded_values, get_cost_function(get_objective(p)).count)
        elseif i < 0 # reset if negative
            r.recorded_values = Vector{Int}()
        end
    end
end

# ╔═╡ 8eba9b9b-836b-48f3-b4fe-067bab605b87
md"""
Now we can initialize the new cost and call the gradient descent.
Note that this illustrates also the last use case – you can pass symbol-action pairs into the `record=`array.
"""

# ╔═╡ 26cac0ff-3339-45f9-8021-04adb4d061fa
F2 = MyCost(data)

# ╔═╡ 95b81875-fd5e-4e70-acbd-27c703b30393
md"""
Now for the plain gradient descent, we have to modify the step (to a constant stepsize) and remove the default check whether the cost increases (setting `debug` to `[]`).
We also only look at the first 20 iterations to keep this example small in recorded values. We call
"""

# ╔═╡ 4acd7072-f442-4433-8d98-77f772dcb35c
R3 = gradient_descent(
    M,
    F2,
    gradF,
    data[1];
    record=[:Iteration, :Count => RecordCount(), :Cost],
	stepsize = ConstantStepsize(1.0),
	stopping_criterion=StopAfterIteration(20),
	debug=[],
    return_state=true,
)

# ╔═╡ c3b78022-1055-4164-aee6-79f0580d72bd
md"""
For `:Cost` we already learned how to access them, the `:Count =>` introduces the following action to obtain the `:Count`. We can again access the whole sets of records
"""

# ╔═╡ c93b9f90-4069-4716-940b-17fcd8db055e
get_record(R3)

# ╔═╡ 8fdf9cbd-80be-4eac-9eff-2ca11f029e7a
md"""
this is equivalent to calling `R[:Iteration]`.
Note that since we introduced `:Count` we can also access a single recorded value using
"""

# ╔═╡ 4396a3c0-b9f6-4dc4-b5ae-f61cd5372671
R3[:Iteration, :Count]

# ╔═╡ 6456503a-6614-43f9-89c1-448e191c7769
md"""
and we see that the cost function is called once per iteration.
"""

# ╔═╡ 8a5ad31c-2a63-464e-b8da-c861f2aaa979
md"""
If we use this counting cost and run the default gradient descent with Armijo linesearch, we can infer how many Armijo linesearch backtracks are preformed:
"""

# ╔═╡ 3e357564-b283-44f5-af3d-0bd1a8c1303d
F3 = MyCost(data)

# ╔═╡ e167e3bc-9bff-44d1-9cfe-7986794a09cf
md"""
To not get too many entries let's just look at the first 20 iterations again
"""

# ╔═╡ 2ce0b70b-6f21-4039-a256-06efb19ae24d
R4 = gradient_descent(
    M,
    F3,
    gradF,
    data[1];
    record=[:Count => RecordCount()],
    return_state=true,

)

# ╔═╡ 15159704-4a1f-4110-add2-796a1ff640ab
get_record(R4)

# ╔═╡ 3851c68b-2245-4e59-b8f5-37c6319609f2
md"""
We can see that the number of cost function calls varies, depending on how many linesearch backtrack steps were required to obtain a good stepsize.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Manifolds = "1cead3c2-87b3-11e9-0ccd-23c62b72b94e"
Manopt = "0fc0a36d-df90-57f3-8f93-d78a9fc72bb5"
Pkg = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
Manifolds = "~0.8.42"
Manopt = "0.3, 0.4"
PlutoUI = "~0.7.49"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.4"
manifest_format = "2.0"
project_hash = "01f56121ecd861199cc08cded25d8f43fc1be763"

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
# ╟─65e3c376-ad9f-11ec-003e-7f033e6865d8
# ╟─85cf1234-ae76-4cb7-9c14-366f25aa6f15
# ╠═94145efb-62fa-4b61-8e57-07d6c2ce66d8
# ╠═02d0a5a5-8c12-4eec-8a3a-1269b5bb69c9
# ╟─cfd5a31a-2430-4e19-b1a8-ef2911109c0c
# ╟─c01983a2-33fc-4812-8071-229b7a0921e3
# ╠═40a09a75-9a7d-44bd-9ab6-4c77946c43ea
# ╠═4331f82f-9778-4b12-8c2d-f007df8b213b
# ╠═604b880f-faf5-4d1c-a6b9-cd94e2a8bf46
# ╠═066267a1-06e1-4eb7-a319-1b05f25b9eac
# ╟─c822fe70-becd-4b0a-b66f-34366f98dfad
# ╠═fbc02960-7f7e-4500-92ac-30df7d058fa9
# ╟─7552c7ba-6d45-4ed9-856c-b00be28a84a0
# ╠═bd4d1bce-7942-498f-a243-bc9090bb3b1c
# ╟─0545973a-d77b-4cce-b834-38a9e08f9a17
# ╠═68cb3e06-866d-4650-863a-61f965e3320f
# ╟─4943d34f-0aff-4879-94a6-337ce42b2e36
# ╠═9d17df05-ca92-4a98-b101-c68ba8d3f7b9
# ╟─edafb0ef-be70-4fed-97b1-eabeae9be5c2
# ╠═46a79917-e03e-4c99-9b13-c525b1abb585
# ╟─ef47dee4-3ee9-45c2-9ebc-2598ea56aa75
# ╠═a42b9205-2447-4b64-af23-9261c34ab284
# ╟─44091233-12c7-4e2b-81c4-2b2d0aa488fa
# ╠═98baba1a-d64d-4852-8f80-85f7248ece98
# ╟─de36da12-d1bb-4b91-8d5f-7a2b64f70dd1
# ╠═5169f6ba-710c-4cef-8b99-9ce19300d634
# ╠═2015f505-cc26-4c1a-b512-3664fba1c0ed
# ╟─f079c326-d6bb-4710-9bd4-1133d74018d6
# ╠═3a915462-efac-4bc4-94d3-da0f400a2eaf
# ╟─cb2b3c30-d930-4de5-b919-39efc5e65fa1
# ╠═930bb10c-cac0-48e2-99fa-f6faf3d0c519
# ╟─995c264c-793e-4c23-9dfd-3eb7bc640e30
# ╠═36965fe9-679d-49b8-8f0d-1cb4877cb5ab
# ╟─abc01e8a-9ae5-483c-9cda-87a196d88918
# ╠═45ee91a6-0377-46e8-bdac-faea18b120e0
# ╟─1c2c65f0-ccbd-4566-a8b3-d1449b683707
# ╠═8c6e70a2-3ebe-43a2-a96a-0219b64a1528
# ╟─bba94b58-2815-4b83-8ee0-02baf9eefa77
# ╠═93af8680-a9d4-4c2d-8423-03eefb0fd752
# ╟─5327d3b1-eb73-4725-bfa8-5d39b6993a6a
# ╠═b3a8e43c-6c82-483f-9449-680ee21a0650
# ╟─05284884-e05c-4714-bf53-2da071c664f7
# ╠═e0fe662d-edc5-4d1f-9fd6-987f37098cc4
# ╟─8eba9b9b-836b-48f3-b4fe-067bab605b87
# ╠═26cac0ff-3339-45f9-8021-04adb4d061fa
# ╟─95b81875-fd5e-4e70-acbd-27c703b30393
# ╠═4acd7072-f442-4433-8d98-77f772dcb35c
# ╟─c3b78022-1055-4164-aee6-79f0580d72bd
# ╠═c93b9f90-4069-4716-940b-17fcd8db055e
# ╟─8fdf9cbd-80be-4eac-9eff-2ca11f029e7a
# ╠═4396a3c0-b9f6-4dc4-b5ae-f61cd5372671
# ╟─6456503a-6614-43f9-89c1-448e191c7769
# ╟─8a5ad31c-2a63-464e-b8da-c861f2aaa979
# ╠═3e357564-b283-44f5-af3d-0bd1a8c1303d
# ╟─e167e3bc-9bff-44d1-9cfe-7986794a09cf
# ╠═2ce0b70b-6f21-4039-a256-06efb19ae24d
# ╠═15159704-4a1f-4110-add2-796a1ff640ab
# ╟─3851c68b-2245-4e59-b8f5-37c6319609f2
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
