# [Stepsize and line search](@id Stepsize)

```@meta
CurrentModule = Manopt
```

Most iterative algorithms determine a direction along which the algorithm shall proceed and
determine a step size to find the next iterate. How advanced the step size computation can be implemented depends (among others) on the properties the corresponding problem provides.

Within `Manopt.jl`, the step size determination is implemented as a `functor` which is a subtype of [`Stepsize`](@ref) based on

```@docs
Stepsize
```

Usually, a constructor should take the manifold `M` as its first argument, for consistency,
to allow general step size functors to be set up based on default values that might depend
on the manifold currently under consideration.

Currently, the following step sizes are available

```@docs
AdaptiveWNGradient
ArmijoLinesearch
ConstantLength
CubicBracketingLinesearch
DecreasingLength
DistanceOverGradients
NonmonotoneLinesearch
Polyak
WolfePowellLinesearch
WolfePowellBinaryLinesearch
```

Additionally, initial stepsize guesses are handled by subtypes of `AbstractInitialLinesearchGuess`:

```@docs
ConstantInitialGuess
ArmijoInitialGuess
HagerZhangInitialGuess
```

Some step sizes use [`max_stepsize`](@ref) function as a rough upper estimate for the trust region size.
It is by default equal to injectivity radius of the exponential map but in some cases a different value is used.
For the `FixedRankMatrices` manifold an estimate from Manopt is used.
Tangent bundle with the Sasaki metric has 0 injectivity radius, so the maximum stepsize of the underlying manifold is used instead.
`Hyperrectangle` also has 0 injectivity radius and an estimate based on maximum of dimensions along each index is used instead.
For manifolds with corners, however, a line search capable of handling break points along the projected search direction should be used, and such algorithms do not call `max_stepsize`.

Internally these step size functions create a [`ManifoldDefaultsFactory`](@ref).
Internally these use

```@autodocs
Modules = [Manopt]
Pages = ["plans/stepsize/stepsize_message.jl", "plans/stepsize/linesearch.jl", "plans/stepsize/stepsize.jl"]
Private = true
Order = [:function, :type]
Filter = t -> !(t in [Manopt.get_message, Stepsize, AdaptiveWNGradient, ArmijoLinesearch, ConstantLength, CubicBracketingLinesearch, DecreasingLength, DistanceOverGradients, NonmonotoneLinesearch, Polyak, WolfePowellLinesearch, WolfePowellBinaryLinesearch ])
```


Some solvers have a different iterate from the one used for the line search.
Then the following state can be used to wrap these locally

```@docs
StepsizeState
```

The Hager-Zhang initial guess uses two helper functions to determine initial stepsize in the first iteration on manifolds which have "expected minimizer", for example the zero point on the Euclidean manifold:

```@docs
default_vector_norm
default_point_distance
```

## Literature

```@bibliography
Pages = ["stepsize.md"]
Canonical=false
```
