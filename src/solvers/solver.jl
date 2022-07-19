@doc raw"""
    decorate_options(o)

decorate the [`Options`](@ref)` o` with specific decorators.

# Optional Arguments
optional arguments provide necessary details on the decorators. A specific
one is used to activate certain decorators.

* `debug` – (`Array{Union{Symbol,DebugAction,String,Int},1}()`) a set of symbols
  representing [`DebugAction`](@ref)s, `Strings` used as dividers and a subsampling
  integer. These are passed as a [`DebugGroup`](@ref) within `:All` to the
  [`DebugOptions`](@ref) decorator dictionary. Only excention is `:Stop` that is passed to `:Stop`.
* `record` – (`Array{Union{Symbol,RecordAction,Int},1}()`) specify recordings
  by using `Symbol`s or [`RecordAction`](@ref)s directly. The integer can again
  be used for only recording every ``i``th iteration.

# See also
[`DebugOptions`](@ref), [`RecordOptions`](@ref)
"""
function decorate_options(
    o::O;
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
        Array{<:Any,1}, # a formated string with symbols orAbstractOptionsActions
    }=missing,
) where {O<:Options}
    o = ismissing(debug) ? o : DebugOptions(o, debug)
    o = ismissing(record) ? o : RecordOptions(o, record)
    return o
end
"""
    initialize_solver!(p,o)

Initialize the solver to the optimization [`Problem`](@ref) by initializing all
values in the [`Options`](@ref)` o`.
"""
function initialize_solver! end

"""
    step_solver!(p,o,iter)

Do one iteration step (the `iter`th) for [`Problem`](@ref)` p` by modifying
the values in the [`Options`](@ref) `o`.
"""
function step_solver! end

"""
    get_solver_result(o)

Return the final result after all iterations that is stored within the
(modified during the iterations) [`Options`](@ref) `o`. By default it uses[`get_iterate`](@ref)
"""
function get_solver_result end

"""
    stop_solver!(p,o,i)

depending on the current [`Problem`](@ref) `p`, the current state of the solver
stored in [`Options`](@ref) `o` and the current iterate `i` this function determines
whether to stop the solver by calling the [`StoppingCriterion`](@ref).
"""
stop_solver!(p::Problem, o::Options, i::Int) = o.stop(p, o, i)

"""
    solve(p,o)

run the solver implemented for the [`Problem`](@ref)` p` and the
[`Options`](@ref)` o` employing [`initialize_solver!`](@ref), [`step_solver!`](@ref),
as well as the [`stop_solver!`](@ref) of the solver.
"""
function solve(p::Problem, o::Options)
    iter::Integer = 0
    initialize_solver!(p, o)
    while !stop_solver!(p, o, iter)
        iter = iter + 1
        step_solver!(p, o, iter)
    end
    return o
end
