# Extension with JuMP.jl

Manopt can be used using the [JuMP.jl](https://jump.dev) interface, see the tutorial TODO:

## Setting s solver and its options

A main thing to choose is the solver to use. By default this is set to the [`GradientDescentState`](@ref). To change the solver you can set it with `set_attribute`. For exmple to use the [`quasi_Newton`](@ref) instead, use

```{julia}
model =  Model(Manopt.JuMP_Optimizer)
set_attribute(model, "descent_state_type", Manopt.QuasiNewtonState)
```

Any of the keywords of the solver you can set with `set_attribute)model, keyword, value)` for example to change the retraction to use, call

```{julia}
set_attribute(model, "retraction_method", ManifoldsBase.ProjectionRetraction())
```

## Interface functions

Several functions from the [Mathematical Optimization Interface](https://github.com/jump-dev/MathOptInterface.jl) (MOI) are
extended when both `Manopt.jl and [JuMP.jl](https://jump.dev) are loaded:

```@docs
Manopt.JuMP_Optimizer
Manopt.JuMP_ManifoldSet
```

## Internal functions

```@docs
JuMP.build_variable
JuMP.jump_function
JuMP.jump_function_type
JuMP.set_objective_function
MOI.add_constrained_variables
MOI.copy_to
MOI.empty!
MOI.dimension
MOI.Utilities.map_indices
MOI.supports_add_constrained_variables
MOI.get
MOI.is_valid
MOI.supports
MOI.supports_incremental_interface
MOI.set
```

## Internal wrappers and their functions

```@autodocs
Modules = [Base.get_extension(Manopt, :ManoptJuMPExt)]
Order   = [:type, :function]
```
