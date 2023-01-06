# [A Manifold Objective](@id ObjectiveSection)

The Objective describes that actual cost function and all its properties.

```@docs
AbstractManifoldObjective
```

Which has two main different possibilities for its containing functions concerning the evaluation mode – not necessarily the cost, but for example gradient in an [`AbstractManifoldGradientObjective`](@ref).

```@docs
AllocatingEvaluation
InplaceEvaluation
```

## Cost function

```@docs

```