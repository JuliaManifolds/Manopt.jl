#
#
# A general framework for solvers of problems on Manifolds
#
# This file introduces fallbacks for not yet implemented parts and the general
# function to run the solver
export decorateOptions
export initializeSolver!, doSolverStep!, getSolverResult, stopSolver!
export solve
@doc doc"""
    decorateOptions(o)

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
  be used for only recording every $i$th iteration.

# See also
[`DebugOptions`](@ref), [`RecordOptions`](@ref)
"""
function decorateOptions(o::O;
        debug::Union{Missing, # none
                    DebugAction, # single one for :All
                    Array{DebugAction,1}, # a group that's put into :All
                    Dict{Symbol,DebugAction}, # the most elaborate, a dictionary
                    Array{<:Any,1}, # short hand for Factory.
                    }=missing,
        record::Union{Missing, # none
                    RecordAction, # single action
                    Array{RecordAction,1}, # a group to be set in :All
                    Dict{Symbol,RecordAction}, # a dictionary for precise settings
                    Array{<:Any,1} # a formated string with symbols or Actions
                    }=missing,
    ) where {O <: Options}
    o = ismissing(debug) ? o : DebugOptions(o,debug)
    o = ismissing(record) ? o : RecordOptions(o,record)
    return o
end
"""
    initializeSolver!(p,o)

Initialize the solver to the optimization [`Problem`](@ref) by initializing all
values in the [`Options`](@ref)` o`.
"""
function initializeSolver!(p::P,o::O) where {P <: Problem, O <: Options}
    sig1 = string( typeof(p) )
    sig2 = string( typeof(o) )
    throw( ErrorException("Initialization of a solver corresponding to the $sig1 and $sig2 not yet implemented." ) )
end
"""
    doSolverStep!(p,o,iter)

Do one iteration step (the `iter`th) for [`Problem`](@ref)` p` by modifying
the values in the [`Options`](@ref) `o`.
"""
function doSolverStep!(p::P,o::O, iter) where {P <: Problem, O <: Options}
    sig1 = string( typeof(p) )
    sig2 = string( typeof(o) )
    throw( ErrorException("Initialization of a solver corresponding to the $sig1 and $sig2 not yet implemented." ) )
end
"""
    getSolverResult(p,o)

Return the final result after all iterations that is stored within the
(modified during the iterations) [`Options`](@ref) `o`.
"""
function getSolverResult(p::P,o::O) where {P <: Problem, O <: Options}
    sig1 = string( typeof(p) )
    sig2 = string( typeof(o) )
    throw( ErrorException("Initialization of a solver corresponding to the $sig1 and $sig2 not yet implemented." ) )
end
"""
    stopSolver!(p,o,i)

depending on the current [`Problem`](@ref) `p`, the current state of the solver
stored in [`Options`](@ref) `o` and the current iterate `i` this function determines
whether to stop the solver by calling the [`StoppingCriterion`](@ref).
"""
stopSolver!(p::P,o::O, i::Int) where {P <: Problem, O <: Options} = o.stop(p,o,i)

"""
    solve(p,o)

run the solver implemented for the [`Problem`](@ref)` p` and the
[`Options`](@ref)` o` employing [`initializeSolver!`](@ref), [`doSolverStep!`](@ref),
as well as the [`stopSolver!`](@ref) of the solver.
"""
function solve(p::P, o::O) where {P <: Problem, O <: Options}
    iter::Integer = 0
    initializeSolver!(p,o)
    while !stopSolver!(p,o,iter)
        iter = iter+1
        doSolverStep!(p,o,iter)
    end
    return o
end