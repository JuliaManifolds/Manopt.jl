# [Record values](@id sec-record)

```@meta
CurrentModule = Manopt
```

To record values during the iterations of a solver run, there are in general two possibilities.
On the one hand, the high-level interfaces provide a `record=` keyword, that accepts several different inputs. For more details see [How to record](../tutorials/HowToRecord.md).

## [Record Actions & the solver state decorator](@id subsec-record-states)

```@autodocs
Modules = [Manopt]
Pages = ["plans/record.jl"]
Order = [:type]
```

## Access functions

```@autodocs
Modules = [Manopt]
Pages = ["plans/record.jl"]
Order = [:function]
Public = true
Private = false
```

## Internal factory functions

```@autodocs
Modules = [Manopt]
Pages = ["plans/record.jl"]
Order = [:function]
Public = false
Private = true
```

Further specific [`RecordAction`](@ref)s can be found when specific types of [`AbstractManoptSolverState`](@ref) define them on their corresponding site.

## Technical details

```@docs
initialize_solver!(amp::AbstractManoptProblem, rss::RecordSolverState)
step_solver!(p::AbstractManoptProblem, s::RecordSolverState, k)
stop_solver!(p::AbstractManoptProblem, s::RecordSolverState, k)
```