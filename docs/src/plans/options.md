## Options

```@meta
CurrentModule = Manopt
```

For most algorithms a certain set of options can either be
generated beforehand of the function with keywords can be used.
Generally the type

```@docs
Options
get_options
```

Since the `Options` directly relate to a solver, they are documented with the
corresponding [Solvers](@ref).
You can always access the options (since they
might be decorated) by calling [`get_options`](@ref).

### Decorators for Options

Options can be decorated using the following trait and function to initialize

```@docs
dispatch_options_decorator
is_options_decorator
decorate_options
```

In general decorators often perform actions so we introduce

```@docs
AbstractOptionsAction
```

as well as a helper for storing values using keys, i.e.

```@docs
StoreOptionsAction
get_storage
has_storage
update_storage!
```

#### [Debug Options](@id DebugOptions)

```@autodocs
Modules = [Manopt]
Pages = ["plans/debug_options.jl"]
Order = [:type, :function]
```

see [DebugSolver](@ref DebugSolver) for details on the decorated solver.

Further specific [`DebugAction`](@ref)s can be found at the specific Options.

#### [Record Options](@id RecordOptions)

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

see [RecordSolver](@ref RecordSolver) for details on the decorated solver.

Further specific [`RecordAction`](@ref)s can be found at the specific Options.

there's one internal helper that might be useful for you own actions, namely

```@docs
record_or_reset!
```