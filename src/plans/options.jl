#
# While a problem consists of all things one has to know about the Optimization
# problem itself (independent of the solver), the options collect parameters,
# that steer the solver (indipendent of the problem at hand)
#
export Options
export ArmijoLineSearchOptions, ArmijoDescentDirectionLineSearchOptions, LineSearchOptions
export GradientDescentOptions, getStepSize
export CyclicProximalPointOptions
export ConjugateGradientOptions
export DebugDecoOptions
export evaluateStoppingCriterion
export getVerbosity, getOptions, setDebugFunction, setDebugOptions

"""
    Options
A general super type for all options.
"""
abstract type Options end
"""
    LineSearchOptions <: Options
A general super type for all options that refer to some line search
"""
abstract type LineSearchOptions <: Options end
"""
    SimpleLineSearchOptions <: LineSearchOptions
A line search without additional no information required, e.g. a constant step size.
"""
type SimpleLineSearchOptions <: LineSearchOptions end
"""
    ArmijoLineSearchOptions <: LineSearchOptions
A subtype of `LineSearchOptions` referring to an Armijo based line search,
especially with a search direction along the negative gradient.

# Fields
a default value is given in brackets. For `ρ` and `c`, only `c` can be left
out but not `ρ``.
* `x` : an [`MPoint`](@ref).
* ìnitialStepsize` : (1.0) and initial step size
* `retraction` : (exp) the rectraction used in line search
* `ρ` : exponent for line search reduction
* `c` : gain within Armijo's rule

*See also*: [`ArmijoLineSearch`](@ref), [`ArmijoDescentDirectionLineSearchOptions`](@ref)
"""
type ArmijoLineSearchOptions <: LineSearchOptions
    x::P where {P <: MPoint}
    initialStepsize::Float64
    retraction::Function
    ρ::Float64
    c::Float64
    ArmijoLineSearchOptions(x::P where {P <: MPoint}, s::Float64=1.0,r::Function=exp,ρ::Float64=0.5,c::Float64=0.0001) = new(x,s,r,ρ,c)
end
"""
    ArmijoDescentDirectionLineSearchOptions <: LineSearchOptions
A subtype of `LineSearchOptions` referring to an Armijo based line search,
searching along a specified direction.

# Fields
a default value is given in brackets. For `ρ` and `c`, only `c` can be left
out but not `ρ``.
* `x` : an [`MPoint`](@ref).
* ìnitialStepsize` : (1.0) and initial step size
* `retraction` : (exp) the rectraction used in line search
* `ρ` : (`0.5`) exponent for line search reduction
* `c` : (`0.0001`)gain within Armijo's rule
* `direction` : direction to search along

*Might be unified to `ArmijoLineSearchOptions` with Julia 0.7 and `missing`
values.*

*See also*:  [`ArmijoLineSearch`](@ref), [`ArmijoLineSearchOptions`](@ref)
"""
type ArmijoDescentDirectionLineSearchOptions <: LineSearchOptions
    x::P where {P <: MPoint}
    initialStepsize::Float64
    retraction::Function
    ρ::Float64
    c::Float64
    direction::T where {T <: TVector}
    ArmijoDescentDirectionLineSearchOptions(x::P where {P <: MPoint},d::T where {T <: TVector}, s::Float64=1.0,r::Function=exp,ρ::Float64=0.5,c::Float64=0.0001) = new(x,s,r,ρ,c,d)
    ArmijoDescentDirectionLineSearchOptions(o::ArmijoLineSearchOptions,d::T where {T <: TVector})  = new(o.x,o.initialStepsize,o.retraction,o.ρ,o.c,d)
end
"""
    GradientDescentOptions{P,L} <: Options
Describes a Gradient based descent algorithm, with

# Fields
a default value is given in brackets if a parameter can be left out in initialization.

* `x0` : an [`MPoint`](@ref) as starting point
* `stoppingCriterion` : a function s,r = @(o,iter,ξ,x,xnew) returning a stop
    indicator and a reason based on an iteration number, the gradient and the last and
    current iterates
* `retraction` : (exp) the rectraction to use
* `lineSearch` : a function performing the lineSearch, returning a step size
* `lineSearchOptions` : options the linesearch is called with.

*See also*: [`steepestDescent`](@ref)
"""
type GradientDescentOptions <: Options
    x0::P where {P <: MPoint}
    stoppingCriterion::Function
    retraction::Function
    lineSearch::Function
    lineSearchOptions::L where {L <: LineSearchOptions}
    # fallback do exp
    GradientDescentOptions(x0::P where {P<:MPoint},sC::Function,lS::Function,lSO::L where {L <: LineSearchOptions},retr::Function=exp) = new(x0,sC,retr,lS,lSO)
end

abstract type DirectionUpdateOptions end
"""
    SimpleDirectionUpdateOptions <: DirectionUpdateOptions
A simple update rule requires no information
"""
struct SimpleDirectionUpdateOptions <: DirectionUpdateOptions
end
struct HessianDirectionUpdateOptions <: DirectionUpdateOptions
end
doc"""
    ConjugateGradientOptions <: Options
specify options for a conjugate gradient descent algoritm, that solves a
[`GradientProblem`].

# Fields
* `x0` : Initial Point on the manifold
* `stoppingCriterion` : a stopping criterion
* `lineSearch` : a function to perform line search that is based on all
  information from `GradientProblem` and the `lineSearchoptions`
* `lineSearchOptions` : options for the `lineSearch`, e.g. parameters necessary
    within [`ArmijoLineSearch`](@ref).
* `directionUpdate` : a function @(M,g,gnew,d) computing the update `dnew` based on the
    current and last gradient as well as the last direction and
* `directionUpdateOptions` : options for the update, if needed (e.g. to provide the hessian with a function handle).

*See also*: [`conjugateGradientDescent`](@ref), [`GradientProblem`](@ref), [`ArmijoLineSearch`](@ref)
"""
type ConjugateGradientOptions <: Options
    x0::P where {P <: MPoint}
    stoppingCriterion::Function
    retraction::Function
    lineSearch::Function
    lineSearchOptions::L where {L <: LineSearchOptions}
    directionUpdate::Function
    directionUpdateOptions::D where {D <: DirectionUpdateOptions}
    ConjugateGradientOptions(x0::P where {P <: MPoint},
        sC::Function,
        lS::Function,
        lSO::L where {L<: LineSearchOptions},
        dU::Function,
        dUO::D where {D <: DirectionUpdateOptions}
        ) = ConjugateGradientOptions(x0,sC,exp,lS,lSO,dU,dUO)
end

abstract type EvalOrder end
type LinearEvalOrder <: EvalOrder end
type RandomEvalOrder <: EvalOrder end
type FixedRandomEvalOrder <: EvalOrder end
"""
    CyclicProximalPointOptions <: Options
stores options for the [`cyclicProximalPoint`](@ref) algorithm. These are the

# Fields
* `x0` : an [`MPoint`](@ref) to start
* `stoppingCriterion` : a function `@(iter,x,xnew,λ_k)` based on the current
    `iter`, `x` and `xnew` as well as the current value of `λ`.
* `λ` : (@(iter) -> 1/iter) a function for the values of λ_k per iteration/cycle
* `orderType` : (`LinearEvalOrder()`) how to cycle through the proximal maps.
    Other values are `RandomEvalOrder()` that takes a new random order each
    iteration, and `FixedRandomEvalOrder()` that fixes a random cycle for all iterations.

*See also*: [`cyclicProximalPoint`](@ref)
"""
type CyclicProximalPointOptions <: Options
    x0::P where {P <: MPoint}
    stoppingCriterion::Function
    λ::Function
    orderType::EvalOrder
    CyclicProximalPointOptions(x0::P where {P <: MPoint},sC::Function,λ::Function=(iter)-> 1.0/iter,o::EvalOrder=LinearEvalOrder()) = new(x0,sC,λ,o)
end
#
#
# Debug Decorator
#
#
type DebugDecoOptions <: Options
    options::O where {O<: Options}
    debugFunction::Function
    debugOptions::Dict{String,<:Any}
    verbosity::Int
end
#
#
# Corresponding Functions, getters&setters
#
#
"""
    evaluateStoppingCriterion(options,iter,ξx1,x2)
Evaluates the stopping criterion based on

# Input
* `o` – options of the solver (maybe decorated) `GradientDescentOptions`
* `iter` – the current iteration
* `ξ` – a tangent vector (the current gradient)
* `x`, `xnew` — two points on the manifold (last and current iterate)

# Output
Result of evaluating stoppingCriterion in the options, i.e.
* `true` if the algorithms stopping criteria are fulfilled and it should stop
* `false` otherwise.
"""
function evaluateStoppingCriterion{O<:Options, P <: MPoint, MT <: TVector, I<:Integer}(o::O,
                          iter::I,ξ::MT, x::P, xnew::P)
  evaluateStoppingCriterion(getOptions(o),iter,ξ,x,xnew)
end
function evaluateStoppingCriterion{O<:GradientDescentOptions, P <: MPoint, MT <: TVector, I<:Integer}(o::O,
                          iter::I,ξ::MT, x::P, xnew::P)
  o.stoppingCriterion(iter,ξ,x,xnew)
end
function evaluateStoppingCriterion{O<:Options, P <: MPoint, I<:Integer}(o::O,
                          iter::I, x::P, xnew::P,λ)
  evaluateStoppingCriterion(getOptions(o),iter,x,xnew,λ)
end
function evaluateStoppingCriterion{O<:CyclicProximalPointOptions, P <: MPoint, I<:Integer}(o::O,
                          iter::I, x::P, xnew::P,λ)
  o.stoppingCriterion(iter,x1,x2,λ)
end
"""
    getVerbosity(Options)

returns the verbosity of the options, if any decorator provides such, otherwise 0
    if more than one decorator has a verbosity, the maximum is returned
"""
function getVerbosity{O<:Options}(o::O)
  if getOptions(o) == o # no Decorator
      return 0
  end
  # else look into encapsulated
  return getVerbosity(getOptions(o))
end
# List here any decorator that has verbosity
function getVerbosity{O<:DebugDecoOptions}(o::O)
  if o.options == getOptions(o.options) # we do not have any further inner decos
      return o.verbosity;
  end
  # else maximum of all decorators
    return max(o.verbosity,getVerbosity(o.options));
end
"""
    getStepsize(p,o)

evaluates the stopping criterion of problem with respect to the current
iteration iter, the descent direction ξ, and two (successive) iterates x1, x2
of the algorithm.
"""
# simplest case – we start with GradientProblem and GradientOptions -> extract line search and its options
function getStepsize{P <: GradientProblem{M} where M <: Manifold, O <: GradientDescentOptions}(p::P,
                              o::O,vars...)
    return getStepsize(p,o.lineSearchOptions,o.lineSearch,vars...)
end
# for gradientLineSearch: Update initS and x and start
function getStepsize{gP <: GradientProblem{M} where M <: Manifold, O <: ArmijoLineSearchOptions, P <: MPoint}(p::gP,
                              o::O, f::Function, x::P, s::Float64)
    o.initialStepsize = s
    o.x = x;
    return getStepsize(p,o,f)
end
# for generalLineSearch - update DescentDir (how?) and continue
function getStepsize{gP <: GradientProblem{M} where M <: Manifold, P <: MPoint, O <: LineSearchOptions}(p::gP,
                              o::O, f::Function, x::P, s::Float64)
  o.initialStepsize = s;
  o.x = x;
  updateDescentDir!(o,x)
  return getStepsize(p,o,f)
end
# (finally) call lineSearchProcedure
function getStepsize{gP <: GradientProblem{M} where M <: Manifold, O <: Union{ArmijoLineSearchOptions,LineSearchOptions}}(p::gP, o::O, f::Function)
  return f(p,o)
end
# modifies o - updates descentDir - if I know how ;)
function updateDescentDir!{O <: LineSearchOptions, P <: MPoint}(o::O,x::P)
# how do the more general cases update?
#  o.descentDirection =
end

getOptions{O <: Options}(o::O) = o; # fallback and end
getOptions{O <: DebugDecoOptions}(o::O) = getOptions(o.options); #unpeel recursively

function setDebugFunction!{O<:Options}(o::O,f::Function)
    if getOptions(o) != o #decorator
        setDebugFunction!(o.options,f)
    end
end
function setDebugFunction!{O<:DebugDecoOptions}(o::O,f::Function)
    o.debugFunction = f;
end
function getDebugFunction{O<:Options}(o::O)
    if getOptions(o) != o #We have a decorator
        return getDebugFunction(o.options,f)
    end
end
function getDebugFunction{O<:DebugDecoOptions}(o::O,f::Function)
    return o.debugFunction;
end
function setDebugOptions!{O<:Options}(o::O,dO::Dict{String,<:Any})
    if getOptions(o) != o #decorator
        setDebugOptions(o.options,dO)
    end
end
function setDebugOptions!{O<:DebugDecoOptions}(o::O,dO::Dict{String,<:Any})
    o.debugOptions = dO;
end
function getDebugOptions{O<:Options}(o::O)
    if getOptions(o) != o #decorator
        return getDebugOptions(o.options)
    end
end
function getDebugOptions{O<:DebugDecoOptions}(o::O,dO::Dict{String,<:Any})
    return o.debugOptions;
end
function optionsHasDebug{O<:Options}(o::O)
    if getOptions(o) == o
        return false;
    else
        return optionsHaveDebug(o.options)
    end
end
optionsHasDebug{O<:DebugDecoOptions}(o::O) = true
