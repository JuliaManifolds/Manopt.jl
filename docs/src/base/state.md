# The Manopt.jl Solver state

The solver state represents all parameters that determine the solver's setup as well as interims memory, e.g. to avoid allocations or to keep certain variables in between iterations. These should also allow insight into how the solver is performing.

A state contains [callbacks](state/callback.md) and a [stopping criterion](stopping_criterion.md)
as mandatory elements and usually the iterate is stored in the field `p`.
A state can be [decorated](state/decorator.md) to add functionality

Further elements a state can use are

* a [stepsize](stepsize.md)
* a sub solver and sub state (TODO: Add link)

## Abstract state

```@autodocs
Modules = [Manopt]
Pages = ["base/state/abstract_state.jl"]
Order = [:type]
```

### Access functions

```@autodocs
Modules = [Manopt]
Pages = ["base/state/abstract_state.jl"]
Order = [:function]
Public=true
Private=false
```

### Internal functions

```@autodocs
Modules = [Manopt]
Pages = ["base/state/abstract_state.jl"]
Order = [:function]
Public=false
Private=true
```
