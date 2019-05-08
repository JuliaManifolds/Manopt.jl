#
#
# A general framework for solvers of problems on Manifolds
#
# This file introduces fallbacks for not yet implemented parts and the general
# function to run the solver
export decorateOptions
export initializeSolver!, doSolverStep!, evaluateStoppingCriterion, getSolverResult, solve
export stopAtIteration, stopChangeLess, stopGradientNormLess
export stopWhenAll, stopWhenAny
#
# Stopping Criteria helper functions
#
"""
    stopAtIteration(maxIter)

given a maximal number `maxIter` this function returns a stopping criterion of
the form `(p,o,i) -> s,r`, where (p,o,i) are the [`Problem`](@ref)` p` its
[`Options`](@ref)` o` and the current iterate `i`, the tuple `s,r` indicates
* `s` whether to stop or not, i.e. whether `i > maxIter`
* `r` the string containing the reason to stop if `s` is true, otherwise the
  empty string.

This function can be combined with other stopping criteria using [`stopWhenAll`](@ref)
    or [`stopWhenAny`](@ref)

# See also
[`stopGradientNormLess`](@ref), [`stopChangeLess`](@ref)
"""
stopAtIteration(maxIter::Int) = (p,o,i) -> ( i > maxIter, (i>maxIter) ? "The algorithm reached its maximal number of iterations ($(maxIter)).\n" : "")
"""
    stopGradientNormLess(ε)

given a real-valued threshold `ε` function returns a stopping criterion of
the form `(p,o,i) -> s,r`, where (p,o,i) are the [`Problem`](@ref)` p` its
[`Options`](@ref)` o` and the current iterate `i`  tuple `s,r` indicates
* `s` whether to stop or not, i.e. whether the norm of the gradient is less than `ε``
* `r` the string containing the reason to stop if `s` is true, otherwise the
  empty string.

This function can be combined with other stopping criteria using [`stopWhenAll`](@ref)
    or [`stopWhenAny`](@ref)

# See also
[`stopAtIteration`](@ref), [`stopChangeLess`](@ref)
"""
stopGradientNormLess(ε::Float64) = (p,o,i) -> ( norm(p.M,o.x,o.∇) < ε,  norm(p.M,o.x,getGradient(p,o.x)) < ε ? "The algorithm reached approximately critical point; the gradient is less than $(ε)\n." : "")
"""
    stopGradientNormLess(ε)

given a real-valued threshold `ε` function returns a stopping criterion of
the form `(p,o,i) -> s,r`, where (p,o,i) are the [`Problem`](@ref)` p` its
[`Options`](@ref)` o` and the current iterate `i` and the returned tuple indicates
* `s` whether to stop or not, i.e. whether the distance from the current to the
  last iterate is less than `ε`
* `r` the string containing the reason to stop if `s` is true, otherwise the
  empty string.

This function can be combined with other stopping criteria using [`stopWhenAll`](@ref)
    or [`stopWhenAny`](@ref)

# See also
[`stopAtIteration`](@ref), [`stopGradientNormLess`](@ref)
"""
stopChangeLess(ε::Float64) = (p,o,i) -> ( distance(p.M, o.x, o.xOld) < ε, distance(p.M, o.x, o.xOld) < ε ? "The algorithm performed a step with a change less than $(ε)\n." : "")
"""
    stopWhenAll(f1,f2,...)

given a set of stopping criteria of the form `(p,o,i) -> s,r`, where `(p,o,i)`
are the [`Problem`](@ref)` p` its [`Options`](@ref)` o` and the current iterate
`i` each returning a `Bool s` whether to stop and a reason `r` (empty if
`s=false`) this function returns a similar function indicating to stop if _any_
of the functions indicates to stop. The reason is the concatenation of the
single reasons.

# See also
[`stopWhenAny`](@ref), [`stopAtIteration`](@ref), [`stopGradientNormLess`](@ref),
[`stopChangeLess`](@ref)
"""
stopWhenAll(kwargs...) = stopWhenAll([kwargs...])
"""
    stopWhenAll(F)

given an array of stopping criteria of the form `(p,o,i) -> s,r`, where `(p,o,i)`
are the [`Problem`](@ref)` p` its [`Options`](@ref)` o` and the current iterate
`i` each returning a `Bool s` whether to stop and a reason `r` (empty if
`s=false`) this function returns a similar function indicating to stop if _all_
of the functions indicate to stop. The reason is the concatenation of the
single reasons.

# See also
[`stopWhenAny`](@ref), [`stopAtIteration`](@ref), [`stopGradientNormLess`](@ref),
[`stopChangeLess`](@ref)
"""
stopWhenAll(F::Array{Function}) = stopWhen(F,all)
"""
    stopWhenAny(f1,f2,...)

given a set of stopping criteria of the form `(p,o,i) -> s,r`, where `(p,o,i)`
are the [`Problem`](@ref)` p` its [`Options`](@ref)` o` and the current iterate
`i` each returning a `Bool s` whether to stop and a reason `r` (empty if
`s=false`) this function returns a similar function indicating to stop if _any_
of the functions indicates to stop. The reason is the concatenation of the
single reasons.

# See also
[`stopWhenAll`](@ref), [`stopAtIteration`](@ref), [`stopGradientNormLess`](@ref),
[`stopChangeLess`](@ref)
"""
stopWhenAny(kwargs...) = stopWhenAny([kwargs...])
"""
    stopWhenAny(F)

given an array of stopping criteria of the form `(p,o,i) -> s,r`, where `(p,o,i)`
are the [`Problem`](@ref)` p` its [`Options`](@ref)` o` and the current iterate
`i` each returning a `Bool s` whether to stop and a reason `r` (empty if
`s=false`) this function returns a similar function indicating to stop if _any_
of the functions indicates to stop. The reason is the concatenation of the
single reasons.

# See also
[`stopWhenAll`](@ref), [`stopAtIteration`](@ref), [`stopGradientNormLess`](@ref),
[`stopChangeLess`](@ref)
"""
stopWhenAny(F::Array{Function}) = stopWhen(F,any)
function stopWhen(F::Array{Function},logicalConcat::Function)
    # evaluate functions and store in array (as a function)
    toArray = (p,o,i) -> [ [b[1], b[2]] for b in [ f(p,o,i) for f in F ] ]
    # turn array into the (Bool, String) return value our anonymous function should return
    toResult = X -> ( logicalConcat( [b[1] for b in X]),string( [b[2] for b in X]...) )
    return (p,o,i) -> toResult(toArray(p,o,i))
end
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
        debug::Array{Symbol,1} = Array{Symbol,1}(),
        debugEvery = 1,
        debugVerbosity = 3,
        debugOutput::IO=Base.stdout,
        record::NTuple{N,Symbol} where N = NTuple{0,Symbol}(),
    ) where {O <: Options}
    if length(debug) > 0
        o = DebugOptions(o,debug,debugEvery,debugVerbosity,debugOutput)
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
    evaluateStoppingCriterion(p,o,iter)

Evaluate, whether the stopping criterion for the [`Problem`](@ref)` p`
and the [`Options`](@ref)` o` after `iter`th iteration is met.
"""
function evaluateStoppingCriterion(p::P,o::O, iter) where {P <: Problem, O <: Options}
    return o.stoppingCriterion(p,o,iter); # fall back to default
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
    solve(p,o)

run the solver implemented for the [`Problem`](@ref)` p` and the
[`Options`](@ref)` o` employing [`initializeSolver!`](@ref), [`doSolverStep!`](@ref),
and [`evaluateStoppingCriterion`](@ref).
"""
function solve(p::P, o::O) where {P <: Problem, O <: Options}
    stop::Bool = false
    reason::String="";
    iter::Integer = 0
    initializeSolver!(p,o)
    while !stop
        iter = iter+1
        doSolverStep!(p,o,iter)
        stop, reason = evaluateStoppingCriterion(p,o,iter)
    end
    return o
end