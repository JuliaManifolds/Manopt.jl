#
# A simple steepest descent algorithm implementation
#
export initializeSolver!, doSolverStep!, getSolverResult
export steepestDescent
@doc doc"""
    steepestDescent(M, F, ∇F, x)
perform a steepestDescent $x_{k+1} = \mathrm{retr}_{x_k} s_k\nabla f(x_k)$ with
different choices of $s_k$ available (see `stepsize` option below).

# Input
* `M` : a manifold $\mathcal M$
* `F` : a cost function $F\colon\mathcal M\to\mathbb R$ to minimize
* `∇F`: the gradient $\nabla F\colon\mathcal M\to T\mathcal M$ of F
* `x` : an initial value $x\in\mathcal M$

# Optional
* `stepsize` : ([`ConstantStepsize`](@ref)`(1.)`) specify a [`Stepsize`](@ref)
  functor.
* `retraction` : (`exp`) a `retraction(M,x,ξ)` to use.
* `stoppingCriterion` : (`[`stopWhenAny`](@ref)`(`[`stopAfterIteration`](@ref)`(200), `[`stopWhenGradientNormLess`](@ref)`(10.0^-8))`)
  a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop.

and the ones that are passed to [`decorateOptions`](@ref) for decorators.

# Output
* `xOpt` – the resulting (approximately critical) point of gradientDescent
* `record` - if activated (using the `record` key, see [`RecordOptions`](@ref)
  an array containing the recorded values.
"""
function steepestDescent(M::mT,
    F::Function, ∇F::Function, x::MP;
    stepsize::Stepsize = ConstantStepsize(1.0),
    retraction::Function = exp,
    stoppingCriterion::StoppingCriterion = stopWhenAny( stopAfterIteration(200), stopWhenGradientNormLess(10.0^-8)),
    kwargs... #collect rest
  ) where {mT <: Manifold, MP <: MPoint}
  p = GradientProblem(M,F,∇F)
  o = GradientDescentOptions(x,stoppingCriterion,stepsize,retraction)

  o = decorateOptions(o; kwargs...)
  resultO = solve(p,o)
  if hasRecord(resultO)
      return getSolverResult(p,getOptions(resultO)), getRecord(resultO)
  end
  return getSolverResult(p,resultO)
end
#
# Solver functions
#
function initializeSolver!(p::P,o::O) where {P <: GradientProblem, O <: GradientDescentOptions}
    o.∇ = zeroTVector(p.M,o.x)
end
function doSolverStep!(p::P,o::O,iter) where {P <: GradientProblem, O <: GradientDescentOptions}
    # update
    o.∇ = getGradient(p,o.x)
    o.x = o.retraction(p.M, o.x , -getStepsize!(p,o,iter) * o.∇)
end
getSolverResult(p::P,o::O) where {P <: GradientProblem, O <: GradientDescentOptions} = o.x

#
# Specific records and Debugs
#
record(p::P,o::O,::Val{:Stepsize},iter) where {P <: GradientProblem, O <: GradientDescentOptions} = o.Stepsize
recordType(p::P,o::O,::Val{:Stepsize}) where {P <: GradientProblem, O <: GradientDescentOptions} = Float64
record(p::P,o::O,::Val{:Gradient},iter) where {P <: GradientProblem, O <: GradientDescentOptions}= o.∇
recordType(p::P,o::O,::Val{:Gradient}) where {P <: GradientProblem, O <: GradientDescentOptions} = typeof(o.∇)

@doc doc"""
    DebugGradient <: DebugAction

debug for the gradient evaluated at the current iterate

# Constructors
    DebugGradient([long=false,p=print])

display the short (`false`) or long (`true`) default text for the gradient.

    DebugGradient(prefix[, p=print])

display the a `prefix` in front of the gradient.
"""
mutable struct DebugGradient <: DebugAction
    print::Function
    prefix::String
    DebugGradientNorm(long::Bool=false,print::Function=print) = new(print,
        long ? "Gradient: " : "∇F(x):")
    DebugGradientNorm(prefix::String,print::Function=print) = new(print,prefix)
end
(d::DebugGradient)(p::ProximalProblem,o::CyclicProximalPointOptions,i::Int) = d.print((i>=0) ? d.prefix*""*string(getproperty(o,:∇)) : "")

@doc doc"""
    DebugGradientNorm <: DebugAction

debug for gradient evaluated at the current iterate.

# Constructors
    DebugGradientNorm([long=false,p=print])

display the short (`false`) or long (`true`) default text for the gradient norm.

    DebugGradientNorm(prefix[, p=print])

display the a `prefix` in front of the gradientnorm.
"""
mutable struct DebugGradientNorm <: DebugAction
    print::Function
    prefix::String
    DebugGradientNorm(long::Bool=false,print::Function=print) = new(print,
        long ? "Norm of the Gradient: " : "|∇F(x)|:")
    DebugGradientNorm(prefix::String,print::Function=print) = new(print,prefix)
end
(d::DebugGradientNorm)(p::P,o::O,i::Int) where {P <: GradientProblem, O <: GradientDescentOptions} = d.print((i>=0) ? d.prefix*"$(norm(p.M,o.x,o.∇))" : "")

@doc doc"""
    DebugStepsize <: DebugAction

debug for the current step size.

# Constructors
    DebugStepsize([long=false,p=print])

display the short (`false`) or long (`true`) default text for the step size.

    DebugStepsize(prefix[, p=print])

display the a `prefix` in front of the step size.
"""
mutable struct DebugStepsize <: DebugAction
    print::Function
    prefix::String
    DebugGradientNorm(long::Bool=false,print::Function=print) = new(print,
        long ? "step size:" : "s:")
    DebugGradientNorm(prefix::String,print::Function=print) = new(print,prefix)
end
(d::DebugStepsize)(p::P,o::O,i::Int) where {P <: GradientProblem, O <: GradientDescentOptions} = d.print((i>0) ? d.prefix*"$(getLastStepsize(p,o,i))" : "")
