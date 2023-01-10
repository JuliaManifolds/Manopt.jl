@doc raw"""
    decorate_state!(s::AbstractManoptSolverState)

decorate the [`AbstractManoptSolverState`](@ref)` s` with specific decorators.

# Optional Arguments
optional arguments provide necessary details on the decorators. A specific
one is used to activate certain decorators.

* `debug` – (`Array{Union{Symbol,DebugAction,String,Int},1}()`) a set of symbols
  representing [`DebugAction`](@ref)s, `Strings` used as dividers and a subsampling
  integer. These are passed as a [`DebugGroup`](@ref) within `:All` to the
  [`DebugSolverState`](@ref) decorator dictionary. Only excention is `:Stop` that is passed to `:Stop`.
* `record` – (`Array{Union{Symbol,RecordAction,Int},1}()`) specify recordings
  by using `Symbol`s or [`RecordAction`](@ref)s directly. The integer can again
  be used for only recording every ``i``th iteration.
* `return_state` - (`false`) indicate whether to wrap the options in a [`ReturnSolverState`](@ref),
  indicating that the solver should return options and not (only) the minimizer.

other keywords are ignored.

# See also

[`DebugSolverState`](@ref), [`RecordSolverState`](@ref), [`ReturnSolverState`](@ref)
"""
function decorate_state!(
    s::S;
    debug::Union{
        Missing, # none
        DebugAction, # single one for :All
        Array{DebugAction,1}, # a group that's put into :All
        Dict{Symbol,DebugAction}, # the most elaborate, a dictionary
        Array{<:Any,1}, # short hand for Factory.
    }=missing,
    record::Union{
        Missing, # none
        Symbol, # single action shortcut by symbol
        RecordAction, # single action
        Array{RecordAction,1}, # a group to be set in :All
        Dict{Symbol,RecordAction}, # a dictionary for precise settings
        Array{<:Any,1}, # a formated string with symbols or AbstractStateActions
    }=missing,
    return_state=false,
    kwargs..., # ignore all others
) where {S<:AbstractManoptSolverState}
    deco_s = ismissing(debug) ? s : DebugSolverState(s, debug)
    deco_s = ismissing(record) ? deco_s : RecordSolverState(deco_s, record)
    deco_s = (return_state) ? ReturnSolverState(deco_s) : deco_s
    return deco_s
end

@doc raw"""
    decorate_objective!(M, o::AbstractManifoldObjective)

decorate the [`AbstractManifoldObjective`](@ref)` o` with specific decorators.

# Optional Arguments

optional arguments provide necessary details on the decorators.
A specific one is used to activate certain decorators.

* `cache` – (`missing`) currently only supports the [`SimpleCacheObjective`](@ref)
  which is activated by either specifying the symbol `:Simple` or the tuple
  (`:Simple, kwargs...`) to pass down keyword arguments

other keywords are ignored.

# See also

[`objective_cache_factory`](@ref)
"""
function decorate_objective!(
    M::AbstractManifold, o::O; cache::Union{Missing,Symbol}=missing, kwargs...
) where {O<:AbstractManifoldObjective}
    deco_o = ismissing(cache) ? o : objective_cache_factory(M, o, cache)
    return deco_o
end

"""
    initialize_solver!(ams::AbstractManoptProblem, amp::AbstractManoptSolverState)

Initialize the solver to the optimization [`AbstractManoptProblem`](@ref) `amp` by
initializing the necessary values in the [`AbstractManoptSolverState`](@ref) `amp`.
"""
initialize_solver!(ams::AbstractManoptProblem, amp::AbstractManoptSolverState)

function initialize_solver!(p::AbstractManoptProblem, s::ReturnSolverState)
    return initialize_solver!(p, s.state)
end

"""
    solve!(p::AbstractManoptProblem, s::AbstractManoptSolverState)

run the solver implemented for the [`AbstractManoptProblem`](@ref)` p` and the
[`AbstractManoptSolverState`](@ref)` s` employing [`initialize_solver!`](@ref), [`step_solver!`](@ref),
as well as the [`stop_solver!`](@ref) of the solver.
"""
function solve!(p::AbstractManoptProblem, s::AbstractManoptSolverState)
    iter::Integer = 0
    initialize_solver!(p, s)
    while !stop_solver!(p, s, iter)
        iter = iter + 1
        step_solver!(p, s, iter)
    end
    return s
end

"""
    step_solver!(amp::AbstractManoptProblem, ams::AbstractManoptSolverState, i)

Do one iteration step (the `i`th) for an [`AbstractManoptProblem`](@ref)` p` by modifying
the values in the [`AbstractManoptSolverState`](@ref) `ams`.
"""
step_solver!(amp::AbstractManoptProblem, ams::AbstractManoptSolverState, i)
function step_solver!(p::AbstractManoptProblem, s::ReturnSolverState, i)
    return step_solver!(p, s.state, i)
end

"""
    stop_solver!(amp::AbstractManoptProblem, ams::AbstractManoptSolverState, i)

depending on the current [`AbstractManoptProblem`](@ref) `amp`, the current state of the solver
stored in [`AbstractManoptSolverState`](@ref) `ams` and the current iterate `i` this function
determines whether to stop the solver, which by default means to call
the internal [`StoppingCriterion`](@ref). `ams.stop`
"""
function stop_solver!(amp::AbstractManoptProblem, ams::AbstractManoptSolverState, i)
    return ams.stop(amp, ams, i)
end
function stop_solver!(p::AbstractManoptProblem, s::ReturnSolverState, i)
    return stop_solver!(p, s.state, i)
end
