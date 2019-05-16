#
#
# A general framework for solvers of problems on Manifolds
#
# This file introduces fallbacks for not yet implemented parts and the general
# function to run the solver
export decorateOptions
export initializeSolver!, doSolverStep!, getSolverResult, stopSolver!
export solve
"""
    decorateOptions(o)

decorate the [`Options`](@ref)` o` with specific decorators.

# Optional Arguments
optional arguments provide necessary details on the decorators. A specific
one is used to activate certain decorators.

* `debug` : (`Array{Symbol,1}()`) a set of symbols printed during the iterations,
at start or end, depending on the symbol. Providing at least on symbol activates
the [`DebugOptions`](@ref) decorator
* `debugEvery` : (`1`) print debug only every `debugEvery`th iteration
* `debugVerbosity` : (`3`) a level of debug verbosity between 1 (least) and 5 (most) output.
* `debugOutput` : (`Base.stdout`) an outputstream to put the debug to.
* `record` : (`NTuple{0,Symbol}()`)

# See also
[`DebugOptions`](@ref), [`RecordOptions`](@ref)
"""
function decorateOptions(o::O;
        debug::Union{Missing,DebugAction,Array{DebugAction,1},Dict{Symbol,DebugAction}}=missing,
        record::NTuple{N,Symbol} where N = NTuple{0,Symbol}(),
    ) where {O <: Options}
    if !ismissing(debug)
        o = DebugOptions(o,debug)
    end
    if length(record) > 0
        o = RecordOptions(o,record)
    end
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
    throw( ErrorException("Initialization of a solver corresponding to the problem $sig1 and options $sig2 not yet implemented." ) )
end
"""
    doSolverStep!(p,o,iter)

Do one iteration step (the `iter`th) for [`Problem`](@ref)` p` by modifying
the values in the [`Options`](@ref)` o`.
"""
function doSolverStep!(p::P,o::O, iter) where {P <: Problem, O <: Options}
    sig1 = string( typeof(p) )
    sig2 = string( typeof(o) )
    throw( ErrorException("Initialization of a solver corresponding to the problem $sig1 and options $sig2 not yet implemented." ) )
end
"""
    getSolverResult(p,o)

Return the final result after all iterations that is stored within the
(modified during the iterations) [`Options`](@ref)` o`.
"""
function getSolverResult(p::P,o::O) where {P <: Problem, O <: Options}
    sig1 = string( typeof(p) )
    sig2 = string( typeof(o) )
    throw( ErrorException("Initialization of a solver corresponding to the problem $sig1 and options $sig2 not yet implemented." ) )
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