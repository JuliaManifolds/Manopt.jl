### A Pluto.jl notebook ###
# v0.18.4

using Markdown
using InteractiveUtils

# ╔═╡ 954560de-faa3-47e9-94e7-194d45f7a0df
using Manopt, Manifolds, Random

# ╔═╡ 65e3c376-ad9f-11ec-003e-7f033e6865d8
md"""
# How to Record Data during the Iterations

The recording and debug possiblities make it possible to record nearly any data during the iterations.
This tutorial illustrates how to

* record one value during the iterations
* record multiple values during the iterations and access them afterwards
* define an own `RecordAction` to perform individual recordings.

Several predefined recordings exist, for example `RecordCost()` or `RecordGradient()`, depending on the solver used.
For fields of the `Options` this can be directly done using the [`RecordEntry(:field)
For other recordings, for example more advanced computations before storing a value, an own `RecordAction` can be defined.

We illustrate these using the gradient descent from the mean computation tutorial.
"""

# ╔═╡ 4331f82f-9778-4b12-8c2d-f007df8b213b
begin
    Random.seed!(42)
    m = 30
    M = Sphere(m)
    n = 800
    σ = π / 8
    x = zeros(Float64, m + 1)
    x[2] = 1.0
    data = [exp(M, x, random_tangent(M, x, Val(:Gaussian), σ)) for i in 1:n]
end

# ╔═╡ 604b880f-faf5-4d1c-a6b9-cd94e2a8bf46
F(M, y) = sum(1 / (2 * n) * distance.(Ref(M), Ref(y), data) .^ 2)

# ╔═╡ 066267a1-06e1-4eb7-a319-1b05f25b9eac
gradF(M, y) = sum(1 / n * grad_distance.(Ref(M), data, Ref(y)))

# ╔═╡ c822fe70-becd-4b0a-b66f-34366f98dfad
md"""
## Plain examples

For the high level interfaces of the solvers, like [`gradient_descent`](https://manoptjl.org/stable/solvers/gradient_descent.html) we have to set `return_options` to `true` to obtain the whole options structure and not only the resulting minimizer.

Then we can easily use the `record=` option to add recorded values. This kesword accepts `RecordAction`s as well as several Symbols as shortcuts, for example `:Cost` to record the cost or if your options have a field `f` `:f` would record that entry.
"""

# ╔═╡ fbc02960-7f7e-4500-92ac-30df7d058fa9
R = gradient_descent(M, F, gradF, data[1]; record=:Cost, return_options=true)

# ╔═╡ 7552c7ba-6d45-4ed9-856c-b00be28a84a0
md"""
From the returned options, we see that the `Options` are encapsulated (decorated) with 
`RecordOptions`.

You can attach different recorders to some operations (`:Start`. `:Stop`, `:Iteration` at time of
 writing), where `:Iteration` is the default, so the following is the same as `get_record(R, :Iteation)`. We get
"""

# ╔═╡ bd4d1bce-7942-498f-a243-bc9090bb3b1c
get_record(R)

# ╔═╡ 0545973a-d77b-4cce-b834-38a9e08f9a17
md"""
To record more than one value, you can pass a array of a mix of symbols and `RecordAction` which formally introduces `RecordGroup`. Such a Group records a tuple of values in every iteration.
"""

# ╔═╡ 68cb3e06-866d-4650-863a-61f965e3320f
R2 = gradient_descent(M, F, gradF, data[1]; record=[:Iteration, :Cost], return_options=true)

# ╔═╡ 4943d34f-0aff-4879-94a6-337ce42b2e36
md"""
Here, the Symbol `:Cost` is mapped to using the `RecordCost` action. The same holds for `:Iteration` and `:Iterate` and any member field of the current `Options`.
To access these you can first extract the group of records (that is where the `:Iteration`s are recorded – note the Pluraö) and then access the `:Cost`
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
We can also pass a Tuple as second argument to have our own order (not that now the second `:Iteration` refers to the recorded iteratons)
"""

# ╔═╡ 98baba1a-d64d-4852-8f80-85f7248ece98
get_record(R2, :Iteration, (:Iteration, :Cost))

# ╔═╡ de36da12-d1bb-4b91-8d5f-7a2b64f70dd1
md"""
## A more complex example

To illustrate a complicated example let's record
* the iteration number, cost and gradient field, but only every sixth iteration
* the iteration at which we stop

We first generate the problem and the options, to also illustrate the low-level works when not using `gradient_descent`.
"""

# ╔═╡ 5169f6ba-710c-4cef-8b99-9ce19300d634
p = GradientProblem(M, F, gradF)

# ╔═╡ 2015f505-cc26-4c1a-b512-3664fba1c0ed
o = GradientDescentOptions(
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
        :Gradient => RecordEntry(similar(data[1]), :gradient),
    ]),
    6,
)

# ╔═╡ cb2b3c30-d930-4de5-b919-39efc5e65fa1
md"""and a small option to record iterations"""

# ╔═╡ 930bb10c-cac0-48e2-99fa-f6faf3d0c519
sI = RecordIteration()

# ╔═╡ 995c264c-793e-4c23-9dfd-3eb7bc640e30
md"""
We now combine both into the `RecordOptions` decorator. It acts completely the same as an `Option` but records something in every iteration additionslly. This is stored in a dictionary of `RecordActions`, where `:Iteration` is the action (here the only every 6th iteration group) and the `sI` which is executed at stop.

Note that the keyword `record=` (in the high level interface `gradient_descent` only would fill the `:Iteration` symbol).
"""

# ╔═╡ 36965fe9-679d-49b8-8f0d-1cb4877cb5ab
r = RecordOptions(o, Dict(:Iteration => rI, :Stop => sI))

# ╔═╡ abc01e8a-9ae5-483c-9cda-87a196d88918
md"""We now call the solver"""

# ╔═╡ 45ee91a6-0377-46e8-bdac-faea18b120e0
res = solve(p, r)

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
We first define a cost, that counts it's own calls.
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
and we define the following RecordAction, which is a Functor, i.e. a struct that is also a function. The function we have to implement is similar to a single solver step in signature, since it might get called every iteration: 
"""

# ╔═╡ e0fe662d-edc5-4d1f-9fd6-987f37098cc4
begin
    mutable struct RecordCount <: RecordAction
        recorded_values::Vector{Int}
        RecordCount() = new(Vector{Int}())
    end
    function (r::RecordCount)(p::Problem, ::Options, i)
        if i > 0
            push!(r.recorded_values, p.cost.count)
        elseif i < 0 # reset if negative
            r.recorded_values = Vector{Int}()
        end
    end
end

# ╔═╡ 8eba9b9b-836b-48f3-b4fe-067bab605b87
md"""
Now we can initialize the new cost and call the gradient descent.
Note that this illustrates also the last use case – you can pass symbol-Action pairs into the `record=`array.
"""

# ╔═╡ 26cac0ff-3339-45f9-8021-04adb4d061fa
F2 = MyCost(data)

# ╔═╡ 4acd7072-f442-4433-8d98-77f772dcb35c
R3 = gradient_descent(
    M,
    F2,
    gradF,
    data[1];
    record=[:Iteration, :Count => RecordCount(), :Cost],
    return_options=true,
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
And we see that the cost function is called once per iteration.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Manifolds = "1cead3c2-87b3-11e9-0ccd-23c62b72b94e"
Manopt = "0fc0a36d-df90-57f3-8f93-d78a9fc72bb5"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
Manifolds = "~0.7.7"
Manopt = "~0.3.20"
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
git-tree-sha1 = "6e8fada11bb015ecf9263f64b156f98b546918c7"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "5.0.5"

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
git-tree-sha1 = "96b0bc6c52df76506efc8a441c6cf1adcb1babc4"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.42.0"

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
git-tree-sha1 = "c43e992f186abaf9965cc45e372f4693b7754b22"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.52"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

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

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "SpecialFunctions", "Test"]
git-tree-sha1 = "65e4589030ef3c44d3b90bdc5aac462b4bb05567"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.8"

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
git-tree-sha1 = "58f25e56b706f95125dcb796f39e1fb01d913a71"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.10"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Manifolds]]
deps = ["Colors", "Distributions", "Einsum", "Graphs", "HybridArrays", "Kronecker", "LinearAlgebra", "ManifoldsBase", "Markdown", "Random", "RecipesBase", "RecursiveArrayTools", "Requires", "SimpleWeightedGraphs", "SpecialFunctions", "StaticArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "e8fb7c69d6e67d377152fd882a20334535db050f"
uuid = "1cead3c2-87b3-11e9-0ccd-23c62b72b94e"
version = "0.7.7"

[[deps.ManifoldsBase]]
deps = ["LinearAlgebra", "Markdown"]
git-tree-sha1 = "058ba95cf4a41d4c7b88879f5b961352880ec919"
uuid = "3362f125-f0bb-47a3-aa74-596ffd7ef2fb"
version = "0.12.12"

[[deps.Manopt]]
deps = ["ColorSchemes", "ColorTypes", "Colors", "DataStructures", "Dates", "LinearAlgebra", "ManifoldsBase", "Markdown", "Printf", "Random", "Requires", "SparseArrays", "StaticArrays", "Statistics", "Test"]
git-tree-sha1 = "bcd147170ad2699518155ba9b12ad33f835c5a4d"
uuid = "0fc0a36d-df90-57f3-8f93-d78a9fc72bb5"
version = "0.3.20"

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

[[deps.NaNMath]]
git-tree-sha1 = "737a5957f387b17e74d4ad2f440eb330b39a62c5"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.0"

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

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

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
git-tree-sha1 = "f5dd036acee4462949cc10c55544cc2bee2545d6"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "2.25.1"

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
git-tree-sha1 = "c3d8ba7f3fa0625b062b82853a7d5229cb728b6b"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.1"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8977b17906b0a1cc74ab2e3a05faa16cf08a8291"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.16"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "25405d7016a47cf2bd6cd91e66f4de437fd54a07"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.16"

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
# ╟─65e3c376-ad9f-11ec-003e-7f033e6865d8
# ╠═954560de-faa3-47e9-94e7-194d45f7a0df
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
# ╟─3a915462-efac-4bc4-94d3-da0f400a2eaf
# ╟─cb2b3c30-d930-4de5-b919-39efc5e65fa1
# ╟─930bb10c-cac0-48e2-99fa-f6faf3d0c519
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
# ╠═4acd7072-f442-4433-8d98-77f772dcb35c
# ╟─c3b78022-1055-4164-aee6-79f0580d72bd
# ╠═c93b9f90-4069-4716-940b-17fcd8db055e
# ╟─8fdf9cbd-80be-4eac-9eff-2ca11f029e7a
# ╠═4396a3c0-b9f6-4dc4-b5ae-f61cd5372671
# ╟─6456503a-6614-43f9-89c1-448e191c7769
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
