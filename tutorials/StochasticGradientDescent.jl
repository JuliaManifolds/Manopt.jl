### A Pluto.jl notebook ###
# v0.19.16

using Markdown
using InteractiveUtils

# ╔═╡ b4fd5dc2-2fca-4b48-8293-b0fc92aaafa9
using Pkg

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

# ╔═╡ af01fd4c-82e2-495e-8475-49de5039bed5
use_local=false

# ╔═╡ 5037f0a4-a269-4724-a2b8-17960a1a3d55
use_local || Pkg.activate()

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
    #@info image_prefix
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
either as one function returning all gradients as a vector (see `gradF`), or – maybe more fitting for a large scale problem, as a vector of small gradient functions (see `gradf`)
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
    M, gradf, x; direction=MomentumGradient(M, x; direction=StochasticGradient(M; X=zero_vector(M, x)))
)

# ╔═╡ dbefa2af-d68f-4918-8ab0-4af257425686
MG = MomentumGradient(M, x; direction=StochasticGradient(M; X=zero_vector(M, x)));

# ╔═╡ 5d5122b6-ba71-48ef-b459-d676ad4cca0d
@benchmark stochastic_gradient_descent($M, $gradf, $x; direction=$MG)

# ╔═╡ d7526835-73d3-40e6-8f8d-1711519ee57a
md"""
And on the other hand the [`AverageGradient`](https://manoptjl.org/stable/solvers/gradient_descent.html#Manopt.AverageGradient) computes an average of the last `n` gradients, i.e.
"""

# ╔═╡ a23ee4d6-6e9b-413a-a3ee-75367c9ea943
x_opt4 = stochastic_gradient_descent(
    M, gradf, x; direction=AverageGradient(M, x; n=10, direction=StochasticGradient(M; X=zero_vector(M, x)))
);

# ╔═╡ b0daf769-f8ea-4d7d-9bd4-48748bc02346
AG = AverageGradient(M, x; n=10, direction=StochasticGradient(M; X=zero_vector(M, x)));

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
x_opt5 = gradient_descent(M, F, fullGradF, x; stepsize=ArmijoLinesearch(M))

# ╔═╡ f292c401-365e-4d86-8784-b060aea8012a
AL = ArmijoLinesearch(M);

# ╔═╡ dba33908-3a1b-491a-abd8-497a00d85230
@benchmark gradient_descent($M, $F, $fullGradF, $x; stepsize=$AL)

# ╔═╡ Cell order:
# ╟─f58f28fe-c3d8-11ec-0acb-e948605dd63f
# ╠═af01fd4c-82e2-495e-8475-49de5039bed5
# ╠═b4fd5dc2-2fca-4b48-8293-b0fc92aaafa9
# ╠═5037f0a4-a269-4724-a2b8-17960a1a3d55
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
# ╠═f292c401-365e-4d86-8784-b060aea8012a
# ╠═dba33908-3a1b-491a-abd8-497a00d85230
