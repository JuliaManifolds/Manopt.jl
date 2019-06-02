# # [Getting Started: Optimize!](@id Optimize) 
#
# This example illustrates how to set up and solve optimization problems and how
# to further get data from the algorithm using [`DebugOptions`](@ref) and
# [`RecordOptions`](@ref)
#
# To start from the quite general case: A __Solver__ is an algorithm that aims
# to solve
# 
# $\operatorname*{argmin}_{x\in\mathcal M} f(x)$
#
# where $\mathcal M$ is a [`Manifold`](@ref) and
# $f\colon\mathcal M \to \mathbb R$ is the cost function.
#
# In `Manopt.jl` a __Solver__ is an algorithm that requires a [`Problem`](@ref)
# `p` and [`Options`](@ref) `o`. While former contains __static__ data,
# most prominently the manifold $\mathcal M$ (usually as `p.M`) and the cost
# function $f$ (usually as `p.costFunction`), the latter contains __dynamic__
# data, i.e. things that usually change during the algorithm, are allowed to
# change, or specify the details of the algorithm to use. Together they form a
# `plan`. A `plan` uniquely determines the algorithm to use and provide all
# necessary information to run the algorithm.
#
# ## Example
#
# A gradient plan consists of a [`GradientProblem`](@ref) with the fields `M`,
# cost function $f$ as well as `gradient` storing the gradient function
# corresponding to $f$. Accessing both functions can be done directly but should
# be encapsulated using [`getCost`](@ref)`(p,x)` and [`getGradient`](@ref)`(p,x)`,
# where in both cases `x` is an [`MPoint`](@ref) on the [`Manifold`](@ref) `M`.
# Second, the [`GradientDescentOptions`](@ref) specify that the algorithm to solve
# the [`GradientProblem`](@ref) will be the [gradient
# descent](https://en.wikipedia.org/wiki/Gradient_descent) algorithm. It requires
# an initial value `o.x0`, a [`StoppingCriterion`](@ref) `o.stop`, a
# [`Stepsize`](@ref) `o.stepsize` and a retraction `o.retraction` and it
# internally stores the last evaluation of the gradient at `o.∇` for convenience.
# The only mandatory parameter is the initial value `x0`, though the defaults for
# both the stopping criterion ([`stopAfterIteration`](@ref)`(100)`) as well as the
# stepsize ([`ConstantStepsize`](@ref)`(1.)` are quite conservative, but are
# chosen to be as simple as possible.
#
# With these two at hand, running the algorithm just requires to call `xOpt = solve(p,o)`.
#
# In the following two examples we will see, how to use a higher level interface
# that allows to more easily activate for example a debug output or record values during the iterations
#
# ## The given Dataset
#
exportFolder = joinpath(@__DIR__,"..","..","docs","src","assets","images","tutorials") #src
using Manopt
using Random, Colors
# For a persistent random set we use
n = 100
σ = π/8
M = Sphere(2)
x = SnPoint(1/sqrt(2)*[1., 0., 1.])
Random.seed!(42)
data = addNoise.(Ref(M), repeat([x],n),Ref(σ))
nothing #hide
# and we define some colors from [Paul Tol](https://personal.sron.nl/~pault/)
black = RGBA{Float64}(colorant"#000000")
TolVibrantOrange = RGBA{Float64}(colorant"#EE7733")
TolVibrantBlue = RGBA{Float64}(colorant"#0077BB")
TolVibrantTeal = RGBA{Float64}(colorant"#009988")
TolVibrantMagenta = RGBA{Float64}(colorant"#EE3377")
nothing #hide
#
# Then our data looks like
#
asyResolution = 2
nothing #hide
renderAsymptote(exportFolder*"/startDataAndCenter.asy",asyExportS2Signals; #src
    render = asyResolution, #src
    points = [ [x], data], #src
    colors=Dict(:points => [TolVibrantBlue, TolVibrantTeal]), #src
    dotSize = 3.5, cameraPosition = (1.,.5,.5) #src
) #src
#md # ```julia
#md # renderAsymptote("startDataAndCenter.asy",asyExportS2Signals;
#md #     render = asyResolution,
#md #     points = [ [x], data],
#md #     colors=Dict(:points => [TolVibrantBlue, TolVibrantTeal]),
#md #     dotSize = 3.5, cameraPosition = (1.,.5,.5)
#md # )
#md # ```
#md # 
#md # ![The data of noisy versions of $x$](../assets/images/tutorials/startDataAndCenter.png)
#
# ## Computing the Mean
#
# To compute the mean on the manifold we use the characterization, that the
# Euclidean mean minimizes the sum of squared distances, and end up with the
# following cost function. Its minimizer is called
# [Riemannian Center of Mass](https://arxiv.org/abs/1407.2087).
#
F = y -> sum(1/(2*n) * distance.(Ref(M),Ref(y),data).^2)
∇F = y -> sum(1/n*gradDistance.(Ref(M),data,Ref(y)))
nothing #hide
#
# note that the [`gradDistance`](@ref) defaults to the case `p=2`, i.e. the
# gradient of the squared distance. For details on convergence of the gradient
# descent for this problem, see [[Afsari, Tron, Vidal, 2013](#AfsariTronVidal2013)]
#
# The easiest way to call the gradient descent is now to call
# [`steepestDescent`](@ref)
xMean = steepestDescent(M,F,∇F,data[1])
nothing; #hide
# but in order to get more details, we further add the `debug=` options, which
# act as a [decorator pattern](https://en.wikipedia.org/wiki/Decorator_pattern)
# using the [`DebugOptions`](@ref) and [`DebugAction`](@ref)s. The latter store
# values if that's necessary, for example for the [`DebugChange`](@ref) that prints
# the change during the last iteration. The following debug prints
#
# `# i | x: | Last Change: | F(x): ``
#
# as well as the reason why the algorithm stopped at the end.
# Here, the format shorthand and the [`DebugFactory`] are used, which returns a
# [`DebugGroup`](@ref) of [`DebugAction`](@ref) performed each iteration and the stop,
# respectively.
xMean = steepestDescent(M,F,∇F,data[1];
   debug = [:Iteration," | ", :x, " | ", :Change, " | ", :Cost, "\n", :Stop]
)
nothing #hide
#
renderAsymptote(exportFolder*"/startDataCenterMean.asy",asyExportS2Signals; #src
    render = asyResolution, #src
    points = [ [x], data, [xMean] ], #src
    colors=Dict(:points => [TolVibrantBlue, TolVibrantTeal, TolVibrantOrange]), #src
    dotSize = 3.5, cameraPosition = (1.,.5,.5) #src
) #src
#md # ```julia
#md # renderAsymptote("startDataCenterMean.asy",asyExportS2Signals;
#md #     render = asyResolution,
#md #     points = [ [x], data, [xMean] ],
#md #     colors=Dict(:points => [TolVibrantBlue, TolVibrantTeal, TolVibrantOrange]),
#md #     dotSize = 3.5, cameraPosition = (1.,.5,.5)
#md # )
#md # ```
#md # 
#md # ![The resulting mean (orange)](../assets/images/tutorials/startDataCenterMean.png)
#
# ## Computing the Median
#
# Similar to the mean you can also define the median as the minimizer of the
# distances, see for example [[Bačák, 2014](#Bačák2014)], but since
# this problem is not differentiable, we employ the Cyclic Proximal Point (CPP)
# algorithm, described in the same reference. We define
#
F2 = y -> sum( 1/(2*n) * distance.(Ref(M),Ref(y),data))
proxes = Function[ (λ,y) -> proxDistance(M,λ/n,di,y,1) for di in data ]
nothing #hide
# where the `Function` is a helper for global scope to infer the correct type.
#
# We then call the [`cyclicProximalPoint`](@ref) as
xMedian, values = cyclicProximalPoint(M,F2,proxes,data[1];
    debug = [:Iteration," | ", :x, " | ", :Change, " | ", :Cost, "\n", 50, :Stop],
    record = [:Iteration, :Change, :Cost]
)
nothing # hide
# where the differences to [`steepestDescent`](@ref) are as follows
# 
# * the third parameter is now an Array of proximal maps
# * debug is reduces to only every 50th iteration
# * we further activated a [`RecordAction`](@ref) using the `record=` optional
#   parameter. These work very similar to those in debug, but they
#   collect their data in an array. The high level interface then returns two
#   variables; the `values` do contain an array of recorded
#   datum per iteration. Here a Tuple containing the iteration, last change and
#   cost respectively; see [`RecordGroup`](@ref), [`RecordIteration`](@ref),
#   [`RecordChange`](@ref), [`RecordCost`](@ref) as well as the [`RecordFactory`](@ref)
#   for details. The `values` contains hence a tuple per iteration,
#   that itself consists of (by order of specification) the iteration number,
#   the last change and the cost function value.
#
# These recorded entries read
values
# The resulting median and mean for the data hence are
#
renderAsymptote(exportFolder*"/startDataCenterMedianAndMean.asy",asyExportS2Signals; #src
    render = asyResolution, #src
    points = [ [x], data, [xMean], [xMedian] ], #src
    colors=Dict(:points => [TolVibrantBlue, TolVibrantTeal, TolVibrantOrange, TolVibrantMagenta]), #src
    dotSize = 3.5, cameraPosition = (1.,.5,.5) #src
) #src
#md # ```julia
#md # renderAsymptote("startDataCenterMean.asy",asyExportS2Signals;
#md #     render = asyResolution,
#md #     points = [ [x], data, [xMean], [xMedian] ],
#md #     colors=Dict(:points => [TolVibrantBlue, TolVibrantTeal, TolVibrantOrange, TolVibrantMagenta]),
#md #     dotSize = 3.5, cameraPosition = (1.,.5,.5)
#md # )
#md # ```
#md # 
#md # ![The resulting mean (orange) and median (magenta)](../assets/images/tutorials/startDataCenterMedianAndMean.png)
#
# ## Literature
# 
# ```@raw html
# <ul>
# <li id="Bačák2014">[<a>Bačák, 2014</a>]
#   Bačák, M: <emph>Computing Medians and Means in Hadamard Spaces.</emph>,
#   SIAM Journal on Optimization, Volume 24, Number 3, pp. 1542–1566,
#   doi: <a href="https://doi.org/10.1137/140953393">10.1137/140953393</a>,
#   arxiv: <a href="https://arxiv.org/abs/1210.2145">1210.2145</a>.</li>
#   <li id="AfsariTronVidal2013">[<a>Afsari, Tron, Vidal, 2013</a>]
#    Afsari, B; Tron, R.; Vidal, R.: <emph>On the Convergence of Gradient
#    Descent for Finding the Riemannian Center of Mass</emph>,
#    SIAM Journal on Control and Optimization, Volume 51, Issue 3,
#    pp. 2230–2260. 
#    doi: <a href="https://doi.org/10.1137/12086282X">10.1137/12086282X</a>,
#    arxiv: <a href="https://arxiv.org/abs/1201.0925">1201.0925</a></li>
# </ul>
# ```