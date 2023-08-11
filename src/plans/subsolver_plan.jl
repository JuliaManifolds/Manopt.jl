"""
    AbstractSubProblemSolverState <: AbstractManoptSolverState

An abstract type for problems that involve a subsolver
"""
abstract type AbstractSubProblemSolverState <: AbstractManoptSolverState end

@doc raw"""
    get_sub_problem(ams::AbstractSubProblemSolverState)

Access the sub problem of a solver state that involves a sub optimisation task.
By default this returns `ams.sub_problem`.
"""
get_sub_problem(ams::AbstractSubProblemSolverState) = ams.sub_problem

@doc raw"""
    get_sub_state(ams::AbstractSubProblemSolverState)

Access the sub state of a solver state that involves a sub optimisation task.
By default this returns `ams.sub_state`.
"""
get_sub_state(ams::AbstractSubProblemSolverState) = ams.sub_state

"""
    set_manopt_parameter!(ams::AbstractManoptSolverState, element::Symbol, value)

Set a certain field/element from the [`AbstractManoptSolverState`](@ref) `ams` to `value.
This function should dispatch on `Val(element)`.
"""
function set_manopt_parameter!(ams::AbstractManoptSolverState, e::Symbol, v)
    return set_manopt_parameter!(ams, Val(e), v)
end
"""
    set_manopt_parameter!(amo::AbstractManoptSolverState, element::Symbol, field::Symbol , value)

Set a certain field/element from the [`AbstractManoptSolverState`](@ref) `amo` to `value.
This function should dispatch on `Val(element)` and `Val{field}`.
"""
function set_manopt_parameter!(ams::AbstractManoptSolverState, e::Symbol, f::Symbol, v)
    return set_manopt_parameter!(ams, Val(e), Val(f), v)
end

function set_manopt_parameter!(ams::AbstractManoptSolverState, ::Val{:Debug}, fv, v)
    return ams
end
function set_manopt_parameter!(dss::DebugSolverState, ::Val{:Debug}, fv, v)
    for d in values(dss.debugDictionary)
        set_manopt_parameter!(d, fv, v)
    end
    return dss
end
function set_manopt_parameter!(ams::AbstractSubProblemSolverState, v::Symbol, args...)
    set_manopt_parameter!(ams, Val(v), args...)
    return ams
end
function set_manopt_parameter!(
    ams::AbstractSubProblemSolverState, ::Val{:SubProblem}, args...
)
    set_manopt_parameter!(get_sub_problem(ams), args...)
    return ams
end
function set_manopt_parameter!(
    ams::AbstractSubProblemSolverState, ::Val{:SubState}, args...
)
    set_manopt_parameter!(get_sub_state(ams), args...)
    return ams
end

"""
    set_manopt_parameter!(ams::AbstractManoptProblem, element::Symbol, field::Symbol , value)

Set a certain field/element from the [`AbstractManoptProblem`](@ref) `ams` to `value.
This function should dispatch on `Val(element)`.

By default this passes on to the inner objective, see [`set_manopt_parameter!`](@ref)
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

In general we assume that the parameter we set refers to a functor/function.
This function should dispatch on `Val{element}`.
By default this function just returns `f`, so it assumes, when using just a function
instead of a fuctor, the parameter `element` is irrelevant for this case.
"""
function set_manopt_parameter!(f, e::Symbol, v)
    return set_manopt_parameter!(f, Val(e), v)
end
function set_manopt_parameter!(f, ::Val, v)
    return f
end
