"""
    AbstractManoptSubProblemSolverState <: AbstractManoptSolverState

An abstract type for problems that involve a subsolver
"""
abstract type AbstractManoptSubProblemSolverState <: AbstractManoptSolverState end

"""
    set_manopt_parameter!(ams::AbstractManoptSolverState, element::Symbol, value)

Set a certain field/element from the [`AbstractManoptSolverState`](@ref) `ams` to `value.
This function should dispatch on `Val(element)`.
"""
function set_manopt_parameter!(ams::AbstractManoptSolverState, e::Symbol, v)
    return set_manopt_parameter!(ams, Val(e), v)
end

"""
    set_manopt_parameter!(ams::AbstractManoptProblem, element::Symbol, field::Symbol , value)

Set a certain field/element from the [`AbstractManoptProblem`](@ref) `ams` to `value.
This function should dispatch on `Val(element)`.

By default this passes on to the inner objective, see [`set_objective_parameter`](@ref)
"""
function set_manopt_parameter!(amp::AbstractManoptProblem, e::Symbol, f::Symbol, v)
    return set_manopt_parameter!(amp, Val(e), Val(f), v)
end
function set_manopt_parameter!(amp::AbstractManoptProblem, ev::Val, fv::Val, v)
    return set_manopt_parameter!(get_objective(amp), ev, fv, v)
end

"""
    set_manopt_parameter!(amo::AbstractManifoldObjective, element::Symbol, field::Symbol , value)

Set a certain field/element from the [`AbstractManifoldObjective`](@ref) `amo` to `value.
This function should dispatch on `Val(element)` and `Val{field}`.
"""
function set_manopt_parameter!(amo::AbstractManifoldObjective, e::Symbol, f::Symbol, v)
    return set_manopt_parameter!(amo, Val(e), Val(f), v)
end

function set_manopt_parameter!(amo::AbstractManifoldObjective, ::Val{:Cost}, fv, v)
    set_manopt_parameter!(get_cost_function(amo), fv, v)
    return amo
end

function set_manopt_parameter!(amo::AbstractManifoldObjective, ::Val{:Gradient}, fv, v)
    set_manopt_parameter!(get_gradient_function(amo), fv, v)
    return amo
end

"""
    set_manopt_parameter!(f, element::Symbol , value)

nIn general we assume that the parametzer we set refers to a functor/function.
This function should dispatch on `Val{element}`. by default this function just returns `f`.
"""
function set_manopt_parameter!(f, e::Symbol, v)
    return set_manopt_parameter!(f, Val(e), v)
end
function set_manopt_parameter!(f, ::Val, v)
    return f
end
