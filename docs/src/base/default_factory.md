# A factory for manifold defaults

Several components of a solver, like the [stepsize](stepsize.md)
mit require a manifold to fill several default values with reasonable values.
This would mean, that when specifying an own step size in a [high level interface](high-level-interface.md), one would have to repeat the manifold, since it is usually already the first of the arguments of that interface.

Therefore, elements like the [`ArmijoLineseach`](@ref) do accept to provide only some of the optional and keyword arguments as well as to “skip” the manifold. Then a [`ManifoldDefaultFactory`](@ref) is used to postpone the constructor call until later, when the manifold and other defaults are “filled in” automatically – since before passing a step size to a [solver state](state.md) we have the manifold available.

```@autodocs
Modules = [Manopt]
Pages = ["base/default_factory.jl"]
Order = [:type, :function]
Private = true
```
