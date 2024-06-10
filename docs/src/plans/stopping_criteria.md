# [Stopping criteria](@id sec-stopping-criteria)

Stopping criteria are implemented as a `functor` and inherit from the base type

```@docs
StoppingCriterion
```

They can also be grouped, which is summarized in the type of a set of criteria

```@docs
StoppingCriterionSet
```

The stopping criteria `s` might have certain internal values/fields it uses to verify against.
This is done when calling them as a function `s(amp::AbstractManoptProblem, ams::AbstractManoptSolverState)`,
where the [`AbstractManoptProblem`](@ref) and the [`AbstractManoptSolverState`](@ref) together represent
the current state of the solver. The functor returns either `false` when the stopping criterion is not fulfilled or `true` otherwise.
One field all criteria should have is the `s.at_iteration`, to indicate at which iteration
the stopping criterion (last) indicated to stop. `0` refers to an indication _before_ starting the algorithm, while any negative number meant the stopping criterion is not (yet) fulfilled. To can access a string giving the reason of stopping see [`get_reason`](@ref).

## Generic stopping criteria

The following generic stopping criteria are available. Some require that, for example,
the corresponding [`AbstractManoptSolverState`](@ref) have a field `gradient` when the criterion should access that.

Further stopping criteria might be available for individual solvers.

```@autodocs
Modules = [Manopt]
Pages = ["plans/stopping_criterion.jl"]
Order = [:type]
Filter = t -> t != StoppingCriterion && t != StoppingCriterionSet
```

## Functions for stopping criteria

There are a few functions to update, combine, and modify stopping criteria, especially to update internal values even for stopping criteria already being used within an [`AbstractManoptSolverState`](@ref) structure.

```@autodocs
Modules = [Manopt]
Pages = ["plans/stopping_criterion.jl"]
Order = [:function]
```