# A manifold objective

```@meta
CurrentModule = Manopt
```

Within the optimization problem

```math
\operatorname*{argmin}_{p \in \mathcal M} f(p)
```

the objective describes the cost ``f(p)`` and its properties relations.
The general abstract type for these is

```@docs
AbstractManifoldObjective
```

For components of the objective, there might be different ways they can be evaluated,
most prominently, whether for example a function providing the gradient ``\operatorname{grad} f``
would allocate new memory or work inplace.

```@docs
AbstractEvaluationType
AllocatingEvaluation
AllocatingInplaceEvaluation
InplaceEvaluation
ParentEvaluationType
evaluation_type
```

The different types of objectives are listed on sub pages depending on their type of information or function

* [First Order Objectives](@ref first_order_objectives) for objectives that provide first order information such as gradients, subgradients or proximal maps
* [Second Order Objectives](@ref second_order_objectives) for objectives that provide second order information such as Hessians
* [Constrained Objectives](@ref constrained_objectives) for objectives that provide constraint information
* [Splitting-based Objectives](@ref splitting_based_objectives) for objectives that provide primal-dual or similar splitting based information
* [Objectives for Linear Systems](@ref objectives_for_linear_models) for objectives that provide linear systems usually in tangent spaces
* [Subproblem Objectives](@ref subproblem_objectives) for objectives that are used in subproblems and need access to the main objective
* [Vectorial Objectives](@ref vectorial_objectives) for objectives that provide vector valued
* [Decorators for Objectives](@ref decorators_for_objectives) for objectives that decorate other objectives, e.g., to provide caching or scaling
