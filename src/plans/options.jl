#
# While a problem consists of all things one has to know about the Optimization
# problem itself (independent of the solver), the options collect parameters,
# that steer the solver (indipendent of the problem at hand)
#
export Options
export ArmijoLineSearchOptions, LineSearchOptions
export GradientDescentOptions, getStepSize
export CyclicProximalPointOptions
export ConjugateGradientOptions
export DebugDecoOptions
export DouglasRachfordOptions
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
mutable struct SimpleLineSearchOptions <: LineSearchOptions end
"""
    ArmijoLineSearchOptions <: LineSearchOptions
A subtype of `LineSearchOptions` referring to an Armijo based line search,
especially with a search direction along the negative gradient.

# Fields
a default value is given in brackets. For `ρ` and `c`, only `c` can be left
out but not `ρ``.
* `x` : an [`MPoint`](@ref).
* `direction` : (optional, can be `missing`) an [`TVector`](@ref).
* `initialStepsize` : (`1.0`) and initial step size
* `retraction` : ([`exp`](@ref) the rectraction used in line search
* `ρ` : exponent for line search reduction
* `c` : gain within Armijo's rule

# See also
[`ArmijoLineSearch`](@ref)
"""
mutable struct ArmijoLineSearchOptions <: LineSearchOptions
    x::P where {P <: MPoint}
    initialStepsize::Float64
    retraction::Function
    ρ::Float64
    c::Float64
    direction::Union{Missing,T where {T <: TVector}}
    ArmijoLineSearchOptions(x::P where {P <: MPoint}, s::Float64=1.0,r::Function=exp,ρ::Float64=0.5,c::Float64=0.0001) = new(x,s,r,ρ,c,missing)
    ArmijoLineSearchOptions(x::P where {P <: MPoint}, ξ::T where {T <: TVector}, s::Float64=1.0,r::Function=exp,ρ::Float64=0.5,c::Float64=0.0001) = new(x,s,r,ρ,c)
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

# See also
[`steepestDescent`](@ref)
"""
mutable struct GradientDescentOptions <: Options
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
"""
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

# See also
[`conjugateGradientDescent`](@ref), [`GradientProblem`](@ref), [`ArmijoLineSearch`](@ref)
"""
mutable struct ConjugateGradientOptions <: Options
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
mutable struct LinearEvalOrder <: EvalOrder end
mutable struct RandomEvalOrder <: EvalOrder end
mutable struct FixedRandomEvalOrder <: EvalOrder end
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

# See also
[`cyclicProximalPoint`](@ref)
"""
mutable struct CyclicProximalPointOptions <: Options
    x0::P where {P <: MPoint}
    stoppingCriterion::Function
    λ::Function
    orderType::EvalOrder
    CyclicProximalPointOptions(x0::P where {P <: MPoint},sC::Function,λ::Function=(iter)-> 1.0/iter,o::EvalOrder=LinearEvalOrder()) = new(M,x0,sC,λ,o)
end
"""
    DouglasRachfordOptions <: Options
"""
mutable struct DouglasRachfordOptions <: Options
    x0::P where {P <: MPoint}
    stoppingCriterion::Function
    λ::Function
    α::Function
    R::Function
    DouglasRachfordOptions(x0::P where {P <: MPoint}, sC::Function, λ::Function=(iter)->1.0, α::Function=(iter)->0.9, R=reflection) = new(M,x0,sC,λ,α,refl)
end
#
#
# Debug Decorator
#
#
mutable struct DebugDecoOptions <: Options
    options::O where {O<: Options}
    debugFunction::Function
    debugOptions::Dict{String,<:Any}
    verbosity::Int
end
getTrustRadius(o::DebugDecoOptions) = getTrustRadius(o.options);
function updateTrustRadius!(o::DebugDecoOptions,newΔ)
    o.options = updateTrustRadius!(o.options,newΔ)
end
#
#
# Trust Region OPtions
#
#
"""
    TrustRegionOptions <: Options
stores option values for a [`trustRegion`](@ref) solver

# Fields
* `maxTrustRadius` – maximal radius of the trust region
* `minΡAcceopt` – minimal value of `ρ` to still accept a new iterate
* `retraction` – the retration to use within
* `stoppingCriterion` – stopping criterion for the algorithm
* `TrustRadius` – current trust region radius
* `TrustRegionSubSolver` - function f(p,x,o) to solve the inner problem with `o` being
* [`TrustRegionSubOptions`](@ref) – options passed to the trustRegion sub problem solver
* `x` – initial value of the algorithm
"""
mutable struct TrustRegionOptions <: Options
    maxTrustRadius::Float64
    minΡAccept::Float64
    retraction::Function
    stoppingCriterion::Function
    TrustRadius::Float64
    TrustRegionSubSolver::Function
    TrustRegionSubOptions::Options
    x::P where {P <: MPoint}
    TrustRegionOptions(x,initΔ,maxΔ,minΡ,sC,retr,tRSubF,tRSubO) = new(maxΔ,minΡ,retr,sC,initΔ,tRSubF,tRSubO,x)
end
getTrustRadius(o::TrustRegionOptions) = o.TrustRadius;
function updateTrustRadius!(o::TrustRegionOptions,newΔ)
    o.TrustRadius = newΔ
end
"""
    TrustRegionSubOptions
Options for the internal subsolver of the [`trustRegion`](@ref) algorithm.

# Fields
- `TrustRadius` : initial radius of the trust region
- `stoppingCriterion` : a function determining when to stop based
  on `(iter,x,η,ηnew)`.

# See also
[`trustRegion`](@ref), [`TrustRegionOptions`](@ref)
"""
mutable struct TrustRegionSubOptions <: Options
    TrustRadius::Float64
    stoppingCriterion::Function
end
getTrustRadius(o::TrustRegionSubOptions) = o.TrustRadius;
function updateTrustRadius!(o::TrustRegionSubOptions,newΔ)
    o.TrustRadius = newΔ
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
function evaluateStoppingCriterion(o::O,iter::I,ξ::MT, x::P, xnew::P) where {O<:Options, P <: MPoint, MT <: TVector, I<:Integer}
  evaluateStoppingCriterion(getOptions(o),iter,ξ,x,xnew)
end
function evaluateStoppingCriterion(o::O,iter::I,ξ::MT, x::P, xnew::P) where {O<:GradientDescentOptions, P <: MPoint, MT <: TVector, I<:Integer}
  o.stoppingCriterion(iter,ξ,x,xnew)
end
# fallback: Unpeel
function evaluateStoppingCriterion(o::O,v...) where {O<:Options}
  evaluateStoppingCriterion(getOptions(o),v...)
end
function evaluateStoppingCriterion(o::O,iter::I, x::P, xnew::P,λ) where {O<:CyclicProximalPointOptions, P <: MPoint, I<:Integer}
  o.stoppingCriterion(iter, x, xnew, λ)
end
function evaluateStoppingCriterion(o::O, iter::I, η::T, x::P, xnew::P) where {O<:TrustRegionOptions, P <: MPoint, T <: TVector, I<:Integer}
  o.stoppingCriterion(iter,η,x,xnew)
end
function evaluateStoppingCriterion(o::O, iter::I, x::P, η::T, ηnew::T) where {O<:TrustRegionSubOptions, P <: MPoint, T <: TVector, I<:Integer}
  o.stoppingCriterion(iter,x,η,ηnew)
end
"""
    getVerbosity(Options)

returns the verbosity of the options, if any decorator provides such, otherwise 0
    if more than one decorator has a verbosity, the maximum is returned
"""
function getVerbosity(o::O) where {O<:Options}
  if getOptions(o) == o # no Decorator
      return 0
  end
  # else look into encapsulated
  return getVerbosity(getOptions(o))
end
# List here any decorator that has verbosity
function getVerbosity(o::O) where {O<:DebugDecoOptions}
  if o.options == getOptions(o.options) # we do not have any further inner decos
      return o.verbosity;
  end
  # else maximum of all decorators
    return max(o.verbosity,getVerbosity(o.options));
end
# simplest case – we start with GradientProblem and GradientOptions -> extract line search and its options
"""
    getStepsize(p,o)

evaluates the stopping criterion of problem with respect to the current
iteration iter, the descent direction ξ, and two (successive) iterates x1, x2
of the algorithm.
"""
function getStepsize(p::P,o::O,vars...) where {P <: GradientProblem{M} where M <: Manifold, O <: GradientDescentOptions}
    return getStepsize(p,o.lineSearchOptions,o.lineSearch,vars...)
end
# for gradientLineSearch: Update initS and x and start
function getStepsize(p::gP,o::O, f::Function, x::P, s::Float64) where {gP <: GradientProblem{M} where M <: Manifold, O <: ArmijoLineSearchOptions, P <: MPoint}
    o.initialStepsize = s
    o.x = x;
    return getStepsize(p,o,f)
end
# for generalLineSearch - update DescentDir (how?) and continue
function getStepsize(p::gP,o::O, f::Function, x::P, s::Float64) where {gP <: GradientProblem{M} where M <: Manifold, P <: MPoint, O <: LineSearchOptions}
  o.initialStepsize = s;
  o.x = x;
  updateDescentDir!(o,x)
  return getStepsize(p,o,f)
end
# (finally) call lineSearchProcedure
function getStepsize(p::gP, o::O, f::Function) where {gP <: GradientProblem{M} where M <: Manifold, O <: Union{ArmijoLineSearchOptions,LineSearchOptions}}
  return f(p,o)
end
# modifies o - updates descentDir - if I know how ;)
function updateDescentDir!(o::O,x::P) where {O <: LineSearchOptions, P <: MPoint}
# how do the more general cases update?
#  o.descentDirection =
end

getOptions(o::O) where {O <: Options} = o; # fallback and end
getOptions(o::O) where {O <: DebugDecoOptions} = getOptions(o.options); #unpeel recursively

function setDebugFunction!(o::O,f::Function) where {O<:Options}
    if getOptions(o) != o #decorator
        setDebugFunction!(o.options,f)
    end
end
function setDebugFunction!(o::O,f::Function) where {O<:DebugDecoOptions}
    o.debugFunction = f;
end
function getDebugFunction(o::O) where {O<:Options}
    if getOptions(o) != o #We have a decorator
        return getDebugFunction(o.options,f)
    end
end
function getDebugFunction(o::O,f::Function) where {O<:DebugDecoOptions}
    return o.debugFunction;
end
function setDebugOptions!(o::O,dO::Dict{String,<:Any}) where {O<:Options}
    if getOptions(o) != o #decorator
        setDebugOptions(o.options,dO)
    end
end
function setDebugOptions!(o::O,dO::Dict{String,<:Any}) where {O<:DebugDecoOptions}
    o.debugOptions = dO;
end
function getDebugOptions(o::O) where {O<:Options}
    if getOptions(o) != o #decorator
        return getDebugOptions(o.options)
    end
end
function getDebugOptions(o::O,dO::Dict{String,<:Any}) where {O<:DebugDecoOptions}
    return o.debugOptions;
end
function optionsHasDebug(o::O) where {O<:Options}
    if getOptions(o) == o
        return false;
    else
        return optionsHaveDebug(o.options)
    end
end
optionsHasDebug(o::O) where {O<:DebugDecoOptions} = true
