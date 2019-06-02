#
# A simple steepest descent algorithm implementation
#
export initializeSolver!, doSolverStep!, getSolverResult
export steepestDescent
export DebugGradient, DebugGradientNorm, DebugStepsize
@doc doc"""
    steepestDescent(M, F, ∇F, x)
perform a steepestDescent $x_{k+1} = \mathrm{retr}_{x_k} s_k\nabla f(x_k)$ with
different choices of $s_k$ available (see `stepsize` option below).

# Input
* `M` – a manifold $\mathcal M$
* `F` – a cost function $F\colon\mathcal M\to\mathbb R$ to minimize
* `∇F`: the gradient $\nabla F\colon\mathcal M\to T\mathcal M$ of F
* `x` – an initial value $x\in\mathcal M$

# Optional
* `stepsize` – ([`ConstantStepsize`](@ref)`(1.)`) specify a [`Stepsize`](@ref)
  functor.
* `retraction` – (`exp`) a `retraction(M,x,ξ)` to use.
* `stoppingCriterion` – (`[`stopWhenAny`](@ref)`(`[`stopAfterIteration`](@ref)`(200), `[`stopWhenGradientNormLess`](@ref)`(10.0^-8))`)
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
    o.∇ = getGradient(p,o.x)
end
function doSolverStep!(p::P,o::O,iter) where {P <: GradientProblem, O <: GradientDescentOptions}
    o.∇ = getGradient(p,o.x)
    o.x = o.retraction(p.M, o.x , -getStepsize!(p,o,iter) * o.∇)
end
getSolverResult(p::P,o::O) where {P <: GradientProblem, O <: GradientDescentOptions} = o.x