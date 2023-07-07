# [Stopping Criteria](@id StoppingCriteria)

Stopping criteria are implemented as a `functor`, i.e. inherit from the base type

```@docs
StoppingCriterion
```

They can also be grouped, which is summarized in the type of a set of criteria

```@docs
StoppingCriterionSet
```

Then the stopping criteria `s` might have certain internal values to check against,
and this is done when calling them as a function `s(amp::AbstractManoptProblem, ams::AbstractManoptSolverState)`,
where the [`AbstractManoptProblem`](@ref) and the [`AbstractManoptSolverState`](@ref) together represent
the current state of the solver. The functor returns either `false` when the stopping criterion is not fulfilled or `true` otherwise.
One field all criteria should have is the `s.reason`, a string giving the reason to stop, see [`get_reason`](@ref).

## Stopping Criteria

The following generic stopping criteria are available. Some require that, for example,
the corresponding [`AbstractManoptSolverState`](@ref) have a field `gradient` when the criterion should check that.

Further stopping criteria might be available for individual solvers.

```@autodocs
Modules = [Manopt]
Pages = ["plans/stopping_criterion.jl"]
Order = [:type]
Filter = t -> t != StoppingCriterion && t != StoppingCriterionSet
```

## Functions for Stopping Criteria

There are a few functions to update, combine and modify stopping criteria, especially to update internal values even for stopping criteria already being used within an [`AbstractManoptSolverState`](@ref) structure.

```@autodocs
Modules = [Manopt]
Pages = ["plans/stopping_criterion.jl"]
Order = [:function]
```
```@docs
get_stopping_criterion
```