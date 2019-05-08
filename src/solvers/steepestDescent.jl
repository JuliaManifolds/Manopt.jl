#
# A simple steepest descent algorithm implementation
#
export initializeSolver!, doSolverStep!, evaluateStoppingCriterion, getSolverResult
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
* `stepsize` : ([`ConstantStepsize`](@ref)`(1.)`) specify a stepsize,
  consisting of a `Tuple{Function,`[`StepsizeOptions`](@ref)`}` where the first
  maps `(p,o,sO)->s`, i.e. a [`GradientProblem`](@ref)` p`, its [`Options`](@ref)` o`
  and the [`StepsizeOptions`](@ref)` sO` to a new step size, where the second
  tuple element are the initial values for `sO`.
* `retraction` : (`exp`) a `retraction(M,x,ξ)` to use.
* `stoppingCriterion` : (`[`stopWhenAny`](@ref)`(`[`stopAtIteration`](@ref)`(200), `[`stopGradientNormLess`](@ref)`(10.0^-8))`)
  a function indicating when to stop.

and the ones that are passed to [`decorateOptions`](@ref) for decorators.

# Output
* `xOpt` – the resulting (approximately critical) point of gradientDescent
* `record` - if activated (using the `record` key, see [`RecordOptions`](@ref)
  an array containing the recorded values.
"""
function steepestDescent(M::mT,
    F::Function, ∇F::Function, x::MP;
    stepsize::Tuple{Function,StepsizeOptions} = ConstantStepsize(1.),
    retraction::Function = exp,
    stoppingCriterion::Function = stopWhenAny( stopAtIteration(200), stopGradientNormLess(10.0^-8)),
    kwargs... #collect rest
  ) where {mT <: Manifold, MP <: MPoint}
  p = GradientProblem(M,F,∇F)
  o = GradientDescentOptions(x,stoppingCriterion,stepsize[1],stepsize[2],retraction)

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
    o.xOld = o.x
    o.∇ = zeroTVector(p.M,o.x)
    o.∇Old = o.∇
    o.stepsize = getInitialStepsize(p,o)
    o.stepsizeOld = o.stepsize
end
function doSolverStep!(p::P,o::O,iter) where {P <: GradientProblem, O <: GradientDescentOptions}
    # save last
    o.xOld = o.x
    o.∇Old = o.∇
    o.stepsizeOld = o.stepsize
    # update
    o.∇ = getGradient(p,o.x)
    o.stepsize = getStepsize(p,o)
    o.x = o.retraction(p.M, o.x , -o.stepsize * o.∇)
end
evaluateStoppingCriterion(p::P,o::O,iter) where {P <: GradientProblem, O <: GradientDescentOptions} = o.stoppingCriterion(p,o,iter)
getSolverResult(p::P,o::O) where {P <: GradientProblem, O <: GradientDescentOptions} = o.x

#
# Specific records and Debugs
#
record(p::P,o::O,::Val{:Stepsize},iter) where {P <: GradientProblem, O <: GradientDescentOptions} = o.Stepsize
recordType(p::P,o::O,::Val{:Stepsize}) where {P <: GradientProblem, O <: GradientDescentOptions} = Float64
record(p::P,o::O,::Val{:Gradient},iter) where {P <: GradientProblem, O <: GradientDescentOptions}= o.∇
recordType(p::P,o::O,::Val{:Gradient}) where {P <: GradientProblem, O <: GradientDescentOptions} = typeof(o.∇)

debug(p::P,o::O,::Val{:Gradient},iter, out::IO=Base.stdout) where {P <: GradientProblem, O <: GradientDescentOptions} = print(out,"Gradient: $(o.∇)")
debug(p::P,o::O,::Val{:GradientNorm},iter, out::IO=Base.stdout) where {P <: GradientProblem, O <: GradientDescentOptions} = print(out,"Norm of gradient: $(norm(p.M,o.x,o.∇))")
debug(p::P,o::O,::Val{:Stepsize},iter, out::IO=Base.stdout) where {P <: GradientProblem, O <: GradientDescentOptions} = print(out,"Stepsize: $(o.stepsize)")