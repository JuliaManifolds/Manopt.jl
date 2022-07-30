# [Options](@id OptionsSection)

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

Since the [`Options`](@ref) directly relate to a solver, they are documented with the
corresponding [solvers](@ref SolversSection).
You can always access the options (since they
might be decorated) by calling [`get_options`](@ref).

For easier access, and to abstract where these are actually stored, there exist

```@docs
get_iterate
get_gradient
```

## Decorators for Options

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