```@meta
CurrentModule = Manopt
```

# High-level interfaces

While the internal structure in a [manifold](@extref `ManifoldsBase.AbstractManifold`),
and an [objective](objective.md) wrapped in a [problem](problem.md) and a
[state](state.md) allows for a modular access, one should have an “easy access” that handles
several things (semi-) automatically.

These are referred to as the high-level-interfaces during this documentation.
For example [`gradient_descent`](@ref)`(M, f, grad_f; kwargs...)` builds the just-mentioned internal structures
based on providing a manifold `M`, a cost function `f` and a gradient `grad_f`.

By default these return the computed minimiser of the objective.

These interfaces are also unified to accept

* a `callbacks = ` keyword to attach [callback](state/callback.md) functions at specific points during a solver run
* a `debug = ` keyword to add [debug output](state/debug.md)
* an `evaluation = ` keyword to specify whether e.g. the gradient can be computed in-place, see the [function](function.md) section.
* a `record = ` keyword to [record elements](state/record.md)
* a `return_objective = ` keyword to return the [objective](objective.md) additionally to the minimiser or state – for example to access function call statistics.
* a `return_state = ` keyword to return the [full solver state](state.md) instead of just the minimizer, for example to access its fields afterwards
* an `objective_type = ` keyword to specify gradients or Hessians to be `:Euclidean` and automatically convert them to Riemannian ones

Additionally there are several keyword arguments, for example for solvers with sub solvers `sub_problem = ` and `sub_state = ` to specify which solvers to use in these. These are then passed to the [solver state](state.md) upon construction.
