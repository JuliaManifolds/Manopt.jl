### A Pluto.jl notebook ###
# v0.19.16

using Markdown
using InteractiveUtils

# ╔═╡ 02d0a5a5-8c12-4eec-8a3a-1269b5bb69c9
using Pkg;

# ╔═╡ 954560de-faa3-47e9-94e7-194d45f7a0df
using Manopt, Manifolds, Random

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

We first need to setup our environment.
First we have to decide whether to use the local Pluto package manegement or not.
The advantage of using that is a high reproducibility, but this variable might be set to false, if the last release introduced a breaking change or a new feature used in this tutorial – so it is only set to `false` for technical reasons.
If you are running the notebook yourself, we recommend to set it to `true`.
"""

# ╔═╡ 94145efb-62fa-4b61-8e57-07d6c2ce66d8
use_local = false

# ╔═╡ 9ab2a874-8f27-469a-8972-e346148babf4
use_local || Pkg.activate()

# ╔═╡ cfd5a31a-2430-4e19-b1a8-ef2911109c0c
md"""
Now we can set up our small test example, which is just a gradient descent for the Riemannian center of mass, see the [Get Started: Optimize!](https://manoptjl.org/stable/tutorials/Optimize!/) tutorial for details.

Here we focus on ways to investigate the behaviour during iterations by using Recording techniques.
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
            push!(r.recorded_values, get_cost_function(p).count)
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

# ╔═╡ Cell order:
# ╟─65e3c376-ad9f-11ec-003e-7f033e6865d8
# ╟─85cf1234-ae76-4cb7-9c14-366f25aa6f15
# ╠═94145efb-62fa-4b61-8e57-07d6c2ce66d8
# ╠═02d0a5a5-8c12-4eec-8a3a-1269b5bb69c9
# ╠═9ab2a874-8f27-469a-8972-e346148babf4
# ╟─cfd5a31a-2430-4e19-b1a8-ef2911109c0c
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
