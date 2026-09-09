# A factory for manifold defaults

```@meta
CurrentModule = Manopt
```

Several components of a solver, like the [step size](stepsize.md),
might require a manifold to fill several defaults with reasonable values.
This would mean that when specifying your own step size in a [high-level interface](high-level-interface.md), one would have to repeat the manifold, since it is usually already the first of the arguments of that interface.

Therefore, elements like the [`ArmijoLinesearch`](@ref) allow you to provide only some of the optional and keyword arguments as well as to “skip” the manifold. Then a [`ManifoldDefaultsFactory`](@ref) is used to postpone the constructor call until later, when the manifold and other defaults are “filled in” automatically – since before passing a step size to a [solver state](state.md) we have the manifold available.

```@autodocs
Modules = [Manopt]
Pages = ["base/default_factory.jl"]
Order = [:type, :function]
Private = true
```
