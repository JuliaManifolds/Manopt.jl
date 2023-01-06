
### [Record values](@id RecordSection)

To record values during the iterations of a solver run, there are in general two possibilities.
On the one hand, the high-level interfaces provide a `record=` keyword, that accepts several different inputs. For more details see [How to record](../tutorials/HowToRecord.md).

For example recording the gradient from the [`GradientDescentState`](@ref) is
automatically available, as explained in the [`gradient_descent`](@ref) solver.

## [Record Solver States](@id RecordSolverState)

```@autodocs
Modules = [Manopt]
Pages = ["plans/record.jl"]
Order = [:type, :function]
Private = true
```

see [recording values](@ref RecordSection) for details on the decorated solver.

Further specific [`RecordAction`](@ref)s can be found when specific types of [`AbstractManoptSolverState`](@ref) define them on their corresponding site.

## Technical Details: The Record Solver
```@autodocs
Modules = [Manopt]
Pages   = ["record_solver.jl"]
```