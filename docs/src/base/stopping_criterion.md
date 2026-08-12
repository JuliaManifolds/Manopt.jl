
```@meta
CurrentModule = Manopt
```

# The Stopping Criterion

```@docs
StoppingCriterion
```

The stopping criterion is a function `(problem, state, k) -> Bool` that after the initialisation and
every iteration determines whether a solver should stop.
It is passed to a solver via the `stopping_criterion = ` keyword in the high-level interfaces.
Since the criterion is stored in the [solver state](state.md), state constructors also accept that keyword.

A stopping criterion should usually store the iteration number it last indicated to stop.
This is within `Manopt` the field `at_iteration`. It should reset its internal variables when called with a negative number like `k=-1`. It should store all necessary information to determine whether to stop and provide a human-reasonable reason why it stopped, see [`get_reason(stopping_criterion)`](@ref get_reason). This reason should return an empty string if the criterion has not yet indicated to stop.

The easiest example is the [`StopAfterIteration`](@ref), which is initialised to a maximal number of iterations and returns  `true` once the input `k` from above exceeds this threshold. This stopping criterion does not store anything else, since the reason only required the current iteration and the maximal one.

There is a list of [common stopping criteria](../commons/stopping_criteria.md) available.
Stopping criteria that a specialised to a single solver can be found on the corresponding solver page.

## Combining stopping criteria

For stopping criteria it is often useful to combine these

```@docs
StoppingCriterionSet
get_stopping_criteria
```

Which is a common supertype for the two specific cases [`StopWhenAll`](@ref) and [`StopWhenAny`](@ref).
These are mapped to the operators `&` and `|`, respectively, so that the combination of stopping criteria can be easily combined.


## Functions related to stopping criteria

Of course the main function to implement is the one of the new data structure
``(sc::StopWhenMyNewCritation)(problem, state, l)` itself.

```@docs
indicates_convergence
is_active_stopping_criterion
has_converged
get_active_stopping_criteria
get_reason
```