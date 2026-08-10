# Commons

```@meta
CurrentModule = Manopt
```

The commons section of the documentation collects common elements used by more than one solver.

## Overview:

* [debug actions](debugs.md) that can be used in any solver, including the
  [`DebugActionFactory`](@ref DebugActionFactory(a::Vector)) that turns the input array to a `debug = ` keyword of a solver into
  the corresponding concrete debugs, especially when passing symbols like `:Cost`.
* [stopping criteria](stopping_criteria.md) that can be used with different solvers in the `stopping_criterion = ` keyword.