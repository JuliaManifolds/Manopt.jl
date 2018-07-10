#
# While a problem consists of all things one has to know about the Optimization
# problem itself (independent of the solver), the options collect parameters,
# that steer the solver (indipendent of the problem at hand)
#
export Options
export GradientLineSearchOptions, LineSearchOptions
export GradientDescentOptions, getStepSize
export CyclicProximalPointOptions
export DebugDecoOptions
export evaluateStoppingCriterion
export getVerbosity, getOptions, setDebugFunction, setDebugOptions

abstract type Options end
abstract type LineSearchOptions <: Options end
type GradientLineSearchOptions <: LineSearchOptions
    x::MPoint
    initialStepsize::Float64
    rho::Float64
    c::Float64
    retraction::Function
    GradientLineSearchOptions(x::MP where {MP <: MPoint}) = GradientLineSearchOptions(x,1.0);
    GradientLineSearchOptions(x::MP where {MP <: MPoint}, s::Float64) = GradientLineSearchOptions(x,s,0.5,0.0001);
    GradientLineSearchOptions(x::MP where {MP <: MPoint}, s::Float64,rho::Float64,c::Float64) = GradientLineSearchOptions(x,s,rho,c,exp)
    GradientLineSearchOptions(x::MP where {MP <: MPoint}, s::Float64,rho::Float64,c::Float64,r::Function) = new(x,s,rho,c,r)
end
type DescentLineSearchOptions <: LineSearchOptions
    x::MPoint
    initialStepsize::Float64
    rho::Float64
    c::Float64
    retraction::Function
    descentDirection::TVector
    DescentLineSearchOptions(o::GradientLineSearchOptions,dir::TVector) = new(o.x,o.initialStepsize,o.rho,o.c,o.retraction,dir)
end

type GradientDescentOptions <: Options
    x0::MPoint
    stoppingCriterion::Function
    retraction::Function
    lineSearch::Function
    lineSearchOptions::L where {L <:LineSearchOptions}
    # fallback do exp
    GradientDescentOptions(x0::MP where {MP <:MPoint},sC::Function,retr::Function,lS::Function,lSO::LSO where {LSO <: LineSearchOptions}) = new(x0,sC,retr,lS,lSO)
    GradientDescentOptions(x0::MP where {MP <:MPoint},sC::Function,lS::Function,lSO::LSO where {LSO <: LineSearchOptions}) = new(x0,sC,exp,lS,lSO)
end

abstract type EvalOrder end
type LinearEvalOrder <: EvalOrder end
type RandomEvalOrder <: EvalOrder end
type FixedRandomEvalOrder <: EvalOrder end

type CyclicProximalPointOptions{MP <: MPoint} <: Options
    x0::MP
    stoppingCriterion::Function
    λ::Function
    orderType::EvalOrder
    CyclicProximalPointOptions{MP}(
        x0::MP,
        sC::Function,
        λ::Function,
        order::EO where { EO <: EvalOrder } = LinearEvalOrder()
        ) where {MP <: MPoint} = new(x0,sC,λ,order)
    CyclicProximalPointOptions{MP}(x0::MP,sC::Function) where {MP <: MPoint} = CyclicProximalPointOptions(x0,sC, (iter)-> 1.0/iter)
end
#
#
# Debug Decorator
#
#
type DebugDecoOptions{O<: Options} <: Options
    options::O
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
function evaluateStoppingCriterion{O<:Options, MP <: MPoint, MT <: TVector, I<:Integer}(o::O,
                          iter::I,ξ::MT, x::MP, xnew::MP)
  evaluateStoppingCriterion(getOptions(o),iter,ξ,x,xnew)
end
function evaluateStoppingCriterion{O<:GradientDescentOptions, MP <: MPoint, MT <: TVector, I<:Integer}(o::O,
                          iter::I,ξ::MT, x::MP, xnew::MP)
  o.stoppingCriterion(iter,ξ,x,xnew)
end
function evaluateStoppingCriterion{O<:Options, MP <: MPoint, I<:Integer}(o::O,
                          iter::I, x::MP, xnew::MP,λ)
  evaluateStoppingCriterion(getOptions(o),iter,x,xnew,λ)
end
function evaluateStoppingCriterion{O<:CyclicProximalPointOptions, MP <: MPoint, I<:Integer}(o::O,
                          iter::I, x::MP, xnew::MP,λ)
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
function getStepsize{P <: GradientProblem{M} where M <: Manifold, O <: GradientLineSearchOptions, MP <: MPoint}(p::P,
                              o::O, f::Function, x::MP, s::Float64)
    o.initialStepsize = s
    o.x = x;
    return getStepsize(p,o,f)
end
# for generalLineSearch - update DescentDir (how?) and continue
function getStepsize{P <: GradientProblem{M} where M <: Manifold, MP <: MPoint, O <: LineSearchOptions}(p::P,
                              o::O, f::Function, x::MP, s::Float64)
  o.initialStepsize = s;
  o.x = x;
  updateDescentDir!(o,x)
  return getStepsize(p,o,f)
end
# (finally) call lineSearchProcedure
function getStepsize{P <: GradientProblem{M} where M <: Manifold, O <: Union{GradientLineSearchOptions,LineSearchOptions}}(p::P, o::O, f::Function)
  return f(p,o)
end
# modifies o - updates descentDir - if I know how ;)
function updateDescentDir!{O <: LineSearchOptions, MP <: MPoint}(o::O,x::MP)
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
