
### [Record values](@id RecordSection)

To record values during the iterations of a solver run, there are in general two possibilities.
On the one hand, the high-level interfaces provide a `record=` keyword, that accepts several different inputs. For more details see [How to record](@ref pluto/HowToRecord.md)

For example recording the gradient from the [`GradientDescentOptions`](@ref) is
automatically available, as explained in the [`gradient_descent`](@ref) solver.

## [Record Options](@id RecordOptions)

```@autodocs
Modules = [Manopt]
Pages = ["plans/record_options.jl"]
Order = [:type, :function]
Private = true
```

```@docs
getindex(ro::RecordOptions, s::Symbol)
getindex(::RecordGroup,::Any...)
```

see [recording values](@ref RecordSection) for details on the decorated solver.

Further specific [`RecordAction`](@ref)s can be found at the specific Options.

there's one internal helper that might be useful for you own actions, namely

```@docs
record_or_reset!
```

## Technical Details: The Record Solver
```@autodocs
Modules = [Manopt]
Pages   = ["record_solver.jl"]
```