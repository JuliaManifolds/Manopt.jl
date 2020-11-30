# # [Stochastic Gradient Descent](@id SGDTutorial)
#
# This tutorial illustrates how to use the [`stochastic_gradient_descent`](@ref)
# solver and different [`DirectionUpdateRule`](@ref)s in order to introduce
# the average or momentum variant, see [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent).
#
# Computationally we look at a very simple but large scale problem,
# the Riemannian Center of Mass or [Fréchet mean](https://en.wikipedia.org/wiki/Fréchet_mean):
# For given points ``p_i \in \mathcal M``, ``i=1,\ldots,N`` this optimization problem reads
#
# ```math
# \operatorname*{arg\,min}_{x\in\mathcal M} \frac{1}{2}\sum_{i=1}^{N}
#   \operatorname{d}_{\mathcal M}(x,p_i),
# ```
# which of course can be (and is) solved by a gradient descent, see the [introductionary tutorial](@ref Optimize).
# If ``N`` is very large it might be quite expensive to evaluate the complete gradient.
# A remedy is, to evaluate only one of the terms at a time and choose a random order for these.
#
# We first initialize the manifold (see [])
export_folder = joinpath( #src
    @__DIR__, #src
    "..", #src
    "..", #src
    "docs", #src
    "src", #src
    "assets", #src
    "images", #src
    "tutorials", #src
) #src
using Manopt, Manifolds, Random, Colors
# and we define some colors from [Paul Tol](https://personal.sron.nl/~pault/)
black = RGBA{Float64}(colorant"#000000")
TolVibrantOrange = RGBA{Float64}(colorant"#EE7733") # Start
TolVibrantBlue = RGBA{Float64}(colorant"#0077BB") # a path
TolVibrantTeal = RGBA{Float64}(colorant"#009988") # points
#
# And optain a large data set
n = 5000
σ = π / 12
M = Sphere(2)
x = 1 / sqrt(2) * [1.0, 0.0, 1.0]
Random.seed!(42)
data = [exp(M, x, random_tangent(M, x, Val(:Gaussian), σ)) for i in 1:n]
nothing #hide
# which looks like
#
asymptote_export_S2_signals( #src
    export_folder * "/centerAndLargeData.asy"; #src
    points=[[x], data], #src
    colors=Dict(:points => [TolVibrantOrange, TolVibrantTeal]), #src
    dot_sizes=[2.5, 1.25], #src
    camera_position=(1.0, 0.5, 0.5), #src
) #src
render_asymptote(export_folder * "/centerAndLargeData.asy"; render=2) #src
#md # ```julia
#md # asymptote_export_S2_signals("centerAndLargeData.asy";
#md #     points = [ [x], data],
#md #     colors=Dict(:points => [TolVibrantBlue, TolVibrantTeal]),
#md #     dot_sizes = [2.5, 1.0], camera_position = (1.,.5,.5)
#md # )
#md # render_asymptote("centerAndLargeData.asy"; render = 2)
#md # ```
#md #
#md # ![The data of noisy versions of $x$](../assets/images/tutorials/centerAndLargeData.png)
#
# In order to use the stochastic gradient, we now need a function that returns the vector of gradients.
# There are two ways to define it in `Manopt.jl`: as one function, that returns a vector or a vector of funtions.
#
# The first variant is of course easier to define, but the second is more efficient when only evaluating one of the gradients.
# For the mean we have as a gradient
#
# ```math
#  ∇F(x) = \sum_{i=1}^N ∇f_i \quad \text{where} ∇f_i(x) = \log_x p_i
# ```
#
# Which we define as
∇F(x) = [ log(M,x,p) for p ∈ data ]
∇f = [ x -> log(M, x, p) for p ∈ data ]
# The calls are only slightly different, but notice that accessing the 2nd gradient element
# requires evaluating all logs in the first function
[ ∇Fk = ∇F(p)[k], ∇f[k](p) ]
# With these we can directly get started with the stochastic gradient descent
x = stochastic_gradient_descent(M,∇F,p)
