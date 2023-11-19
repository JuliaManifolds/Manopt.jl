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

```@autodocs
Modules = [Manopt]
Pages = ["plans/stepsize.jl"]
Order = [:type,:function]
Filter = t -> t != Stepsize
```

## Literature

```@bibliography
Pages = ["stepsize.md"]
Canonical=false
```