# Commons

```@meta
CurrentModule = Manopt
```

The commons section of the documentation collects common elements used by more than one solver.

## Overview:

* [common debug actions](debugs.md) that can be used in any solver, including the
  [`DebugActionFactory`](@ref DebugActionFactory(a::Vector)) that turns the input array to a `debug = ` keyword of a solver into
  the corresponding concrete debugs, especially when passing symbols like `:Cost`.