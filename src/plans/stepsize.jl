#
# Collection of step sizes
#
export ConstantStepsize, DecreasingStepsize
export Linesearch, ArmijoLinesearch
export getStepsize!, getInitialStepsize, getLastStepsize
#
# Simple ones
#
"""
    ConstantStepsize <: Stepsize

A functor that always returns a fixed step size.

# Fields
* `length` – constant value for the step size.

# Constructor

    ConstantStepSize(s)

initialize the stepsie to a constant `s`
"""
mutable struct ConstantStepsize <: Stepsize
    length::Float64
    ConstantStepsize(s::Real) = new(s)
end
(s::ConstantStepsize)(p::P,o::O,i::Int) where {P <: Problem, O <: Options} = s.length
getInitialStepsize(s::ConstantStepsize) = s.length

@doc doc"""
    DecreasingStepsize()

A functor that represents several decreasing step sizes

# Fields
* `length` – (`1`)the initial step size $l$.
* `factor` – (`1`)(a value $f$ to multiply the initial step size with every iteration
* `subtrahend` – (`0`)a value $a$ that is subtracted every iteration
* `exponent` – (`1`) a value $e$ the current iteration numbers $e$th exponential
  is taken of

In total the complete formulae reads for the $i$th iterate as

$ s_i = \frac{(l-i\cdot a)f^i}{i^e}$

and hence the default simplifies to just $ s_i = \frac{l}{i}
# Constructor

    ConstantStepSize(l,f,a,e)

initialiszes all fields above, where none of them is mandatory.
"""
mutable struct DecreasingStepsize <: Stepsize
    length::Float64
    factor::Float64
    subtrahend::Float64
    exponent::Float64
    DecreasingStepsize(l::Real=1.0, f::Real=1.0, a::Real=0.0, e::Real=1.0) = new(l,f,a,e)
end
(s::DecreasingStepsize)(p::P,o::O,i::Int) where {P <: Problem, O <: Options} = (s.length - i*s.subtrahend)*(s.factor^i)/(i^(s.exponent))
getInitialStepsize(s::DecreasingStepsize) = s.length

#
# Linesearch
#
"""
    Linesearch <: Stepsize

An abstract functor to represent line search type step size deteminations, see
[`Stepsize`](@ref) for details. One example is the [`ArmijoLinesearch`](@ref)
functor.

Compared to simple step sizes, the linesearch functors provide an interface of
the form `(p,o,i,η) -> s` with an additional (but optional) fourth parameter to
proviade a search direction; this should default to something reasonable, e.g.
the negative gradient.
"""
abstract type Linesearch <: Stepsize end

@doc doc"""
    ArmijoLineseach <: Linesearch

A functor representing Armijo line seach including the last runs state, i.e. a
last step size.

# Fields
* `initialStepsize` – (`1.0`) and initial step size
* `retraction` – ([`exp`](@ref Manopt.exp)) the rectraction used in line search
* `contractionFactor` – (`0.95`) exponent for line search reduction
* `sufficientDecrease` – (`0.1`) gain within Armijo's rule
* `lastStepSize` – 

# Constructor

    ArmijoLineSearch(i,retr,c,s)

construct an Armijo lineseach, where the last step size is also initialized to i.
"""
mutable struct ArmijoLinesearch <: Linesearch
    initialStepsize::Float64
    retraction::Function
    contractionFactor::Float64
    sufficientDecrease::Float64
    stepsizeOld::Float64
    ArmijoLinesearch(
        s::Float64=1.0,
        r::Function=exp,
        contractionFactor::Float64=0.95,
        sufficientDecrease::Float64=0.1) = new(s, r, contractionFactor, sufficientDecrease,s)
end
function (a::ArmijoLinesearch)(p::P,o::O,i::Int, η::T=-getGradient(p,o.x)) where {P <: GradientProblem{mT} where mT <: Manifold, O <: Options, T <: TVector}
  # for local shortness
  F = p.costFunction
  gradient = getGradient(p,o.x)
  s = a.stepsizeOld
  retr = a.retraction
  e = F(o.x)
  eNew = e-1
  if e > eNew
    xNew = retr(p.M,o.x,s*η)
    eNew = F(xNew) + a.sufficientDecrease*s*dot(p.M, o.x, η, gradient)
    if e >= eNew
      s = s/a.contractionFactor
    end
  end
  while (e < eNew) && (s > typemin(Float64))
    s = s*a.contractionFactor
    xNew = retr(p.M,o.x,s*η)
    eNew = F(xNew) + a.sufficientDecrease*s*dot(p.M, o.x, η, gradient)
  end
  a.stepsizeOld = s
  return s
end
getInitialStepsize(a::ArmijoLinesearch) = a.initialStepsize

#
# Access functions
#
@traitfn getStepsize!(p::P, o::O,vars...) where {P <: Problem, O <: Options; IsOptionsDecorator{O}} = getStepsize!(p, o.options,vars...)
@traitfn getStepsize!(p::P, o::O,vars...) where {P <: Problem, O <: Options; !IsOptionsDecorator{O}} = o.stepsize(p,o,vars...)

@traitfn getInitialStepsize(p::P,o::O,vars...) where {P <: Problem, O <: Options; !IsOptionsDecorator{O}} = getInitialStepsize(o.stepsize)
@traitfn getInitialStepsize(p::P, o::O,vars...) where {P <: Problem, O <: Options; IsOptionsDecorator{O}} = getInitialStepsize(p, o.options)

@traitfn getLastStepsize(p::P, o::O,vars...) where {P <: Problem, O <: Options; IsOptionsDecorator{O}} = getLastStepsize(p, o.options,vars...)
@traitfn getLastStepsize(p::P, o::O,vars...) where {P <: Problem, O <: Options; !IsOptionsDecorator{O}} = getLastStepsize(p,o,o.stepsize,vars...)

getLastStepsize(p::P,o::O,s::S,vars...) where {P <: Problem, O <: Options,S <: Stepsize} = s(p,o,vars...)
getLastStepsize(p::P,o::O,s::ArmijoLinesearch,vars...) where {P <: Problem, O <: Options} = s.stepsizeOld
