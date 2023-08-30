# [The Solver State](@id SolverStateSection)

```@meta
CurrentModule = Manopt
```

Given an [`AbstractManoptProblem`](@ref), that is a certain optimisation task,
the state specifies the solver to use. It contains the parameters of a solver and all
fields necessary during the algorithm, e.g. the current iterate, a [`StoppingCriterion`](@ref)
or a [`Stepsize`](@ref).

```@docs
AbstractManoptSolverState
get_state
Manopt.get_count
```

Since every subtype of an [`AbstractManoptSolverState`](@ref) directly relate to a solver,
the concrete states are documented together with the corresponding [solvers](@ref SolversSection).
This page documents the general functionality available for every state.

A first example is to access, i.e. obtain or set, the current iterate.
This might be useful to continue investigation at the current iterate, or to set up a solver for a next experiment, respectively.

```@docs
get_iterate
set_iterate!
get_gradient(s::AbstractManoptSolverState)
set_gradient!
```

An internal function working on the state and elements within a state is used to
pass messages from (sub) activities of a state to the corresponding [`DebugMessages`](@ref)

```@docs
get_message
```

Furthermore, to access the stopping criterion use

```@docs
get_stopping_criterion
```

## Decorators for AbstractManoptSolverState

A solver state can be decorated using the following trait and function to initialize

```@docs
dispatch_state_decorator
is_state_decorator
decorate_state!
```

A simple example is the

```@docs
ReturnSolverState
```

as well as [`DebugSolverState`](@ref) and [`RecordSolverState`](@ref).

## State Actions

A state action is a struct for callback functions that can be attached within
for example the just mentioned debug decorator or the record decorator.

```@docs
AbstractStateAction
```

Several state decorators or actions might store intermediate values like the (last) iterate to compute some change or the last gradient. In order to minimise the storage of these, there is a generic [`StoreStateAction`](@ref)
that acts as generic common storage that can be shared among different actions.

```@docs
StoreStateAction
get_storage
has_storage
update_storage!
PointStorageKey
VectorStorageKey
```

as well as two internal functions

```@docs
_storage_copy_vector
_storage_copy_point
```

## Abstract States

In a few cases it is useful to have a hierarchy of types. These are

```@docs
AbstractSubProblemSolverState
AbstractGradientSolverState
AbstractHessianSolverState
AbstractPrimalDualSolverState
```

For the sub problem state, there are two access functions

```@docs
get_sub_problem
get_sub_state
```