
### [Record values](@id RecordSection)

The decorator to record certain values during the iterations can be activated by
decorating the [`Options`](@ref) with [`RecordOptions`](@ref) and implementing
your own [`RecordAction`](@ref)s.
For example recording the gradient from the [`GradientDescentOptions`](@ref) is
automatically available, as explained in the [`gradient_descent`](@ref) solver.

## [Record Options](@id RecordOptions)

```@autodocs
Modules = [Manopt]
Pages = ["plans/record_options.jl"]
Order = [:type, :function]
Private = false
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