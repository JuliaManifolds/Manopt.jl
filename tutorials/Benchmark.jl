### A Pluto.jl notebook ###
# v0.19.16

using Markdown
using InteractiveUtils

# ╔═╡ 1285b840-f8f5-496e-a48f-9734831f711a
using Pkg;

# ╔═╡ 62b0cf82-e00c-498b-a6d8-124ea9950ef4
using Manopt, Manifolds, Random, BenchmarkTools

# ╔═╡ d5e636f0-533a-11ec-168a-e118d53012cb
md"""
# Illustration of how to Use Mutating Gradient Functions

When it comes to time critital operations, a main ingredient in Julia is given by
mutating functions, i.e. those that compute in place without additional memory
allocations. In the following, we illustrate how to do this with `Manopt.jl`.

Let's start with the same function as in [Get Started: Optimize!](https://manoptjl.org/stable/tutorials/MeanAndMedian.html)
and compute the mean of some points, only that here we use the sphere ``\mathbb S^{30}``
and `n=800` points.

From the aforementioned example, the implementation looks like
"""

# ╔═╡ 1fcb3484-62eb-49c6-ac3c-ce3a77f80107
use_local=false

# ╔═╡ 66462459-be40-486d-af83-6ea7eeedca68
use_local || Pkg.activate()

# ╔═╡ 967e5caa-1d27-4c24-a882-6f7126053754
begin
    Random.seed!(42)
    m = 30
    M = Sphere(m)
    n = 800
    σ = π / 8
    x = zeros(Float64, m + 1)
    x[2] = 1.0
    data = [exp(M, x, σ * rand(M; vector_at=x)) for i in 1:n]
end;

# ╔═╡ 1436cdc7-1e84-4438-937d-2211856348de
md"""
## Classical Definition

The variant from the previous tutorial defines a cost ``F(x)`` and its gradient ``gradF(x)``
"""

# ╔═╡ 6f1b04cf-2a76-4660-916c-3b8b8472d3c0
F(M, x) = sum(1 / (2 * n) * distance.(Ref(M), Ref(x), data) .^ 2)

# ╔═╡ e487f38c-334e-4819-b775-a9a44632d7ff
gradF(M, x) = sum(1 / n * grad_distance.(Ref(M), data, Ref(x)))

# ╔═╡ e4a7b318-65e7-40bb-899d-8bdeeefbf304
md"""
We further set the stopping criterion to be a little more strict. Then we obtain
"""

# ╔═╡ 5a2d6d9f-20dc-41a2-bfce-32509a1ef106
begin
    sc = StopWhenGradientNormLess(1e-10)
    x0 = zeros(Float64, m + 1); x0[1] = 1/sqrt(2); x0[2] = 1/sqrt(2)
	m1 = gradient_descent(M, F, gradF, x0; stopping_criterion=sc)
end

# ╔═╡ 3eea2945-4cbc-41a3-ae5c-6bd1e81535bb
@benchmark gradient_descent($M, $F, $gradF, $x0; stopping_criterion=$sc)

# ╔═╡ 306c7706-f399-4104-9159-c2e6b0bca189
md"""
## In-place Computation of the Gradient

We can reduce the memory allocations by implementing the gradient as a [functor](https://docs.julialang.org/en/v1/manual/methods/#Function-like-objects).
The motivation is twofold: on one hand, we want to avoid variables from the global scope,
for example the manifold `M` or the `data`, being used within the function.
Considering to do the same for more complicated cost functions might also be worth it.

Here, we store the data (as reference) and one temporary memory in order to avoid
reallocation of memory per `grad_distance` computation. We have
"""

# ╔═╡ 4e464076-d477-4e38-aac6-79639c1cb9b5
begin
    struct grad!{TD,TTMP}
        data::TD
        tmp::TTMP
    end
    function (gradf!::grad!)(M, X, x)
        fill!(X, 0)
        for di in gradf!.data
            grad_distance!(M, gradf!.tmp, di, x)
            X .+= gradf!.tmp
        end
        X ./= length(gradf!.data)
        return X
    end
end

# ╔═╡ 15d4c49c-7178-4be1-8e83-00900e384831
md"""
Then we just have to initialize the gradient and perform our final benchmark.
Note that we also have to interpolate all variables passed to the benchmark with a `$`.
"""

# ╔═╡ 90f40202-69ec-4dba-b1d0-905c916a3642
begin
    gradF2! = grad!(data, similar(data[1]))
	m2 = deepcopy(x0)
    gradient_descent!(
        M, F, gradF2!, m2; evaluation=InplaceEvaluation(), stopping_criterion=sc
    )
end

# ╔═╡ 56f48b11-cb0d-40b1-a9e8-547339d54abf
@benchmark gradient_descent!(
        $M, $F, $gradF2!, m2; evaluation=$(InplaceEvaluation()), stopping_criterion=$sc
    ) setup = (m2 = deepcopy($x0))

# ╔═╡ 3e01469b-670b-47a6-a379-199b70c59207
md"""Note that the results `m1`and `m2` are of course the same."""

# ╔═╡ 10c7b9e5-024c-4815-be58-c524a7a08d25
distance(M, m1, m2)

# ╔═╡ Cell order:
# ╟─d5e636f0-533a-11ec-168a-e118d53012cb
# ╠═1fcb3484-62eb-49c6-ac3c-ce3a77f80107
# ╠═1285b840-f8f5-496e-a48f-9734831f711a
# ╠═66462459-be40-486d-af83-6ea7eeedca68
# ╠═62b0cf82-e00c-498b-a6d8-124ea9950ef4
# ╠═967e5caa-1d27-4c24-a882-6f7126053754
# ╟─1436cdc7-1e84-4438-937d-2211856348de
# ╠═6f1b04cf-2a76-4660-916c-3b8b8472d3c0
# ╠═e487f38c-334e-4819-b775-a9a44632d7ff
# ╟─e4a7b318-65e7-40bb-899d-8bdeeefbf304
# ╠═5a2d6d9f-20dc-41a2-bfce-32509a1ef106
# ╠═3eea2945-4cbc-41a3-ae5c-6bd1e81535bb
# ╟─306c7706-f399-4104-9159-c2e6b0bca189
# ╠═4e464076-d477-4e38-aac6-79639c1cb9b5
# ╟─15d4c49c-7178-4be1-8e83-00900e384831
# ╠═90f40202-69ec-4dba-b1d0-905c916a3642
# ╠═56f48b11-cb0d-40b1-a9e8-547339d54abf
# ╟─3e01469b-670b-47a6-a379-199b70c59207
# ╠═10c7b9e5-024c-4815-be58-c524a7a08d25
