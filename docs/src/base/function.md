```@meta
CurrentModule = Manopt
```

# A function

An [objective](objective.md) of an optimisation [problem](problem.md) may contain different
functions related to the objective. In the simplest case a cost function ``f(p)`` and its (Riemannian) gradient
``\operatorname{grad} f(p)`` which returns the tangent vector of the steepest ascent direction of
a differentiable function ``f``.
