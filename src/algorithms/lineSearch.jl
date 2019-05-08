#
# Manopt.jl – line Search
#
# methods related to line search
#
# ---
# Manopt.jl – Ronny Bergmann – 2018-06-25
export ArmijoLineSearch, ArmijoLinesearchOptions, Armijo
"""
    Armijo(initialStepsize,retraction,contractionFactor,sufficientDecrease)

return the option for ArmijoLineSearch with parameters
* `initialStepsize` : (`1.0`) and initial step size
* `retraction` : ([`exp`](@ref Manopt.exp)) the rectraction used in line search
* `contractionFactor` : (`0.95`) exponent for line search reduction
* `sufficientDecrease` : (`0.1`) gain within Armijo's rule
"""
Armijo(i=1.0,r=exp,c=0.95,s=0.1) = (ArmijoLineSearch, ArmijoLinesearchOptions(i,r,c,s))

"""
    ArmijoLinesearchOptions <: StepsizeOptions
A subtype of `StepsizeOptions` referring to an Armijo based line search,
especially with a search direction along the negative gradient.

# Fields
a default value is given in brackets.
* `initialStepsize` : (`1.0`) and initial step size
* `retraction` : (`exp`) the rectraction used in line search
* `contractionFactor` : (`0.95`) exponent for line search reduction
* `sufficientDecrease` : (`0.1`) gain within Armijo's rule

# See also
[`ArmijoLineSearch`](@ref)
"""
mutable struct ArmijoLinesearchOptions <: LinesearchOptions
    initialStepsize::Float64
    retraction::Function
    contractionFactor::Float64
    sufficientDecrease::Float64
    ArmijoLinesearchOptions(
        s::Float64=1.0,
        r::Function=exp,
        contractionFactor::Float64=0.95,
        sufficientDecrease::Float64=0.1) = new(s, r, contractionFactor, sufficientDecrease)
end

"""
    ArmijoLineSearch(p,o,aO[, descentDirection])

compute the step size with respect to Armijo's rule for a
[`GradientProblem`](@ref)` P`, some (undecorated) [`Options`](@ref)` o`,
and the corresponding [`ArmijoLinesearchOptions`](@ref)` aO`.

The optional argument `descxentDirection` can be used to search along a certain
direction. If not provided [`getGradient`](@ref)`(p,o.x)` is used.
"""
function ArmijoLineSearch(p::GradientProblem{Mc},
    o::O,
    aO::ArmijoLinesearchOptions,
    descentDirection::TVector = -getGradient(p,o.x)
    )::Float64 where {Mc<:Manifold, O <: Options}
  # for local shortness
  F = p.costFunction
  gradient = -getGradient(p,o.x)
  s = o.stepsize
  retr = aO.retraction
  e = F(o.x)
  eNew = e-1
  if e > eNew
    xNew = retr(p.M,o.x,s*descentDirection)
    eNew = F(xNew) - aO.sufficientDecrease*s*dot(p.M,o.x,descentDirection,gradient)
    if e >= eNew
      s = s/aO.contractionFactor
    end
  end
  while (e < eNew) && (s > typemin(Float64))
    s = s*aO.contractionFactor
    xNew = retr(p.M,o.x,s*descentDirection)
    eNew = F(xNew) - aO.sufficientDecrease*s*dot(p.M,o.x,descentDirection,gradient)
  end
  return s
end

#
# Redefine getStepsize for Armijo to not perform a linesearch initially
#
function getInitialStepsize(p::P,o::O, lo::ArmijoLinesearchOptions) where {P <: GradientProblem{M} where M <: Manifold, O <: GradientDescentOptions}
    return lo.initialStepsize
end
